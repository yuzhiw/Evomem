#!/usr/bin/env python3
"""hbs_learn.py — 论文 Eq.4 的 HBS 参数学习（开发集训练，测试阶段冻结）。

论文 App A.2：
    对开发集每个查询 q（类型 T_q），令 C(q) 为 FG 候选节点、R*(q) ⊆ C(q) 为
    "能推出参考答案的最小来源可溯支持集"，用 support-evidence ranking loss 训练

        L_HBS = -Σ_{(q,T_q)} log [ Σ_{n∈R*} exp S(n|pa(n),q) / Σ_{n∈C} exp S(n|pa(n),q) ]
                + λ_reg Σ_T ||θ_T - θ_0||²

    训练完的参数在静态测试阶段冻结（EDPL 之后才可能微调）。

    S(n|pa(n),q) = α·Φ_struct + β·sim + γ·c_cal + δ·f_rec        (Eq.2)
    Φ_struct     = w_{T_q}^T φ(e_ij)，φ 为 creation/update/deletion 的 one-hot

实现要点（与论文口径对齐）：
  * R*(q) 用 benchmark 提供的 answer_session_ids 界定：来源会话 ∈ answer_session_ids
    的版本节点即"支撑参考答案的最小证据"。这是 benchmark 标注，不是答案内容。
  * 开发集默认取 LongMemEval 全量 500 题中**不在 120 题测试子集**里的题（严格 disjoint），
    每类取 N 题（默认 8），共 48 题。
  * 只做正则 EAT / 无 LLM 调用、无文档级 embedding（只 embed 候选节点文本），因此很便宜。
  * 输出 JSON：{"theta": {T: [α,β,γ,δ]}, "w": {T: [w_creation, w_update, w_deletion]}}

用法：
    CUDA_VISIBLE_DEVICES="" python3 hbs_learn.py --test-set data/lme_120_sample.json \
        --full /path/longmemeval_s --per-type 8 --out hbs_theta_dev.json
"""
import argparse, json, os, sys, math, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import numpy as np

from hierarchyv2.trimem_ar_v2 import EntityFactGraph
from hierarchyv2.fact_extractor_v2 import extract_facts_v2
from hierarchyv2.date_utils import parse_date, fmt_date
from hierarchyv2 import knowledge_memory_v2 as KM


def build_efg(item):
    """只用 EAT 正则构建 EFG（无 LLM、无文档 embedding）——HBS 打分只需要版本链。"""
    efg = EntityFactGraph()
    sessions = item.get('haystack_sessions', [])
    dates = item.get('haystack_dates', [])
    for si, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        dt = dates[si] if si < len(dates) else ''
        p = parse_date(dt)
        dt_str = fmt_date(p) if p else ''
        for turn in session:
            if isinstance(turn, dict) and turn.get('role', 'user') == 'user' and turn.get('content', '').strip():
                for f in extract_facts_v2(turn['content'].strip(), dt_str, f'session_{si}'):
                    efg.add_fact(f)
    return efg


def collect_candidates(query, efg, qtype, now, max_candidates=None):
    """复刻 ta_hbs_search 的候选收集（实体 → 属性 → 版本），返回 (version, feats) 列表。

    feats = (phi(3), sim, c_cal, f_rec)；Φ_struct 由 w_T 与 phi 点积在外部算。
    """
    max_candidates = max_candidates or KM.HBS_MAX_CANDIDATES
    import re
    q_lower = query.lower()
    q_words = set(re.findall(r'[a-zA-Z]{3,}', q_lower)) - KM._STOP
    q_entities = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
    common_ents = {'How', 'What', 'When', 'Where', 'Why', 'Which', 'Who', 'The', 'This', 'That',
                   'These', 'Those', 'My', 'Your', 'His', 'Her', 'Its', 'Our', 'Their', 'Me',
                   'You', 'He', 'She', 'It', 'We', 'They'}
    q_entities = [e.lower() for e in q_entities if e not in common_ents]
    matched = [e for e in q_entities if e in efg.entities]
    if not matched:
        for name in efg.entities:
            if name.lower() in q_lower or any(qw in name.lower() for qw in q_words):
                matched.append(name)
                if len(matched) >= 5:
                    break
    L1 = matched if matched else list(efg.entities.keys())[:10]

    A = []
    for ent in L1[:5]:
        A.extend(list(efg.attribute_edges.get(ent, []))[:5])
    F_init = []
    for attr_key in A[:6]:
        for vid in efg.version_chains.get(attr_key, []):
            v = efg.get_version(vid)
            if v:
                F_init.append(v)
    if len(F_init) > max_candidates:
        F_init = F_init[:max_candidates]
    if not F_init:
        return []

    from hierarchyv2.embedding import embed
    q_emb = np.array(embed([query])).flatten()
    texts = [f"{v.get('object', '')} {v.get('confidence', '')}" for v in F_init]
    embs = np.array(embed(texts))
    out = []
    for v, ve in zip(F_init, embs):
        sim = float(np.dot(q_emb, ve) / (np.linalg.norm(q_emb) * np.linalg.norm(ve) + 1e-10))
        c_cal = KM.bayesian_confidence_calibrate(v.get('confidence', 'medium'), v.get('weight', 1.0))
        f_rec = KM.compute_recency(v.get('time', ''), now)
        et = KM.classify_edge_type(v, efg)
        phi = [0.0, 0.0, 0.0]
        phi[KM.EDGE_TYPE_INDEX[et]] = 1.0
        out.append((v, (phi, sim, c_cal, f_rec)))
    return out


def score(feats, theta, w):
    phi, sim, c_cal, f_rec = feats
    a, b, g, d = theta
    return a * float(np.dot(w, phi)) + b * sim + g * c_cal + d * f_rec


def learn(dev_items, iters=60, lr=0.05, l2=0.01, verbose=True):
    """对开发集做 Eq.4 的梯度下降，返回学得的 {T: θ} 与 {T: w}。"""
    # 预计算每个 dev 题的候选特征（一次构建，多次迭代复用）
    grouped = {}
    t0 = time.time()
    for it in dev_items:
        q, qt = it['question'], it['question_type']
        vt = _vke(qt)
        now = parse_date(it.get('question_date', '')[:10]) or None
        efg = build_efg(it)
        cands = collect_candidates(q, efg, vt, now)
        if not cands:
            continue
        ans_sids = set(it.get('answer_session_ids') or [])
        hsids = it.get('haystack_session_ids') or []
        keep_idx = {i for i, sid in enumerate(hsids) if sid in ans_sids}
        R = [(v, f) for v, f in cands if _session_index(v) in keep_idx]
        C = [(v, f) for v, f in cands]
        if not R:
            continue
        grouped.setdefault(vt, []).append((R, C))
    if verbose:
        print(f'[learn] dev items with usable candidates: '
              f'{sum(len(v) for v in grouped.values())} ({time.time()-t0:.0f}s)', flush=True)
        for k, v in grouped.items():
            print(f'   {k}: {len(v)} questions', flush=True)

    theta, wts = {}, {}
    for T, items in grouped.items():
        th = list(KM.DEFAULT_WEIGHTS.get(T, (0.25, 0.35, 0.20, 0.20)))
        th0 = list(th)
        w = list(KM.EDGE_WEIGHTS.get(T, (1/3, 1/3, 1/3)))
        for _it in range(iters):
            g_th = [0.0] * 4
            g_w = [0.0] * 3
            for R, C in items:
                sR = np.array([score(f, th, w) for _, f in R])
                sC = np.array([score(f, th, w) for _, f in C])
                pR = np.exp(sR - sR.max()); pR /= pR.sum()
                pC = np.exp(sC - sC.max()); pC /= pC.sum()
                # 期望特征
                eR = np.zeros(4); eC = np.zeros(4)
                for p, (_, f) in zip(pR, R):
                    phi, sim, c_cal, f_rec = f
                    eR += p * np.array([float(np.dot(w, phi)), sim, c_cal, f_rec])
                for p, (_, f) in zip(pC, C):
                    phi, sim, c_cal, f_rec = f
                    eC += p * np.array([float(np.dot(w, phi)), sim, c_cal, f_rec])
                g_th += -(eR - eC)
                # w 的梯度：dS/dw_k = α·φ_k
                phiR = np.zeros(3); phiC = np.zeros(3)
                for p, (_, f) in zip(pR, R):
                    phiR += p * np.array(f[0])
                for p, (_, f) in zip(pC, C):
                    phiC += p * np.array(f[0])
                g_w += -th[0] * (phiR - phiC)
            n = len(items)
            for j in range(4):
                th[j] -= lr * (g_th[j] / n + 2 * l2 * (th[j] - th0[j]))
                th[j] = max(0.05, min(0.70, th[j]))
            for k in range(3):
                w[k] -= lr * (g_w[k] / n)
                w[k] = max(0.0, w[k])
            _s = sum(w) or 1.0
            w = [x / _s for x in w]   # w 保持归一化（论文：权重向量）
        theta[T] = [round(x, 4) for x in th]
        wts[T] = [round(x, 4) for x in w]
        if verbose:
            print(f'  [{T}] θ {th0} -> {theta[T]} | w {wts[T]}', flush=True)
    return theta, wts


def _vke(qt):
    return {'knowledge-update': 'KU', 'temporal-reasoning': 'TR', 'multi-session': 'MS',
            'single-session-preference': 'PR', 'single-session-user': 'SS',
            'single-session-assistant': 'SS'}.get(qt, qt)


def _session_index(v):
    s = str(v.get('source', ''))
    if s.startswith('session_'):
        try:
            return int(s.split('_')[1])
        except Exception:
            return -1
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-set', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '..', 'data', 'lme_120_sample.json'))
    ap.add_argument('--full', required=True, help='LongMemEval 全量 json（500 题）')
    ap.add_argument('--per-type', type=int, default=8)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='hbs_theta_dev.json')
    a = ap.parse_args()

    test_ids = {x['question_id'] for x in json.load(open(a.test_set))}
    full = json.load(open(a.full))
    pool = [x for x in full if x['question_id'] not in test_ids]
    rng = np.random.RandomState(a.seed)
    by_t = {}
    for x in pool:
        by_t.setdefault(x['question_type'], []).append(x)
    dev = []
    for t, xs in sorted(by_t.items()):
        idx = rng.permutation(len(xs))[:a.per_type]
        dev += [xs[i] for i in idx]
    print(f'[dev] {len(dev)} questions from {len(pool)} held-out (disjoint from test '
          f'{len(test_ids)}), per type {a.per_type}', flush=True)

    theta, wts = learn(dev, iters=a.iters, lr=a.lr, l2=a.l2)
    json.dump({'theta': theta, 'w': wts,
               'meta': {'dev_n': len(dev), 'per_type': a.per_type, 'iters': a.iters,
                        'lr': a.lr, 'l2': a.l2, 'seed': a.seed,
                        'test_set': os.path.basename(a.test_set)}},
              open(a.out, 'w'), indent=1)
    print(f'[saved] {a.out}', flush=True)


if __name__ == '__main__':
    main()
