#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_retrieval_ablation.py — 检索侧消融 runner（ARIS 阶段3→4）

用法（每配置独立进程，保证状态隔离）：
    python3 run_retrieval_ablation.py --dataset lme --config full
    python3 run_retrieval_ablation.py --dataset lme --config wo_profiling
    python3 run_retrieval_ablation.py --dataset lme --config wo_hbs
    python3 run_retrieval_ablation.py --dataset lme --config wo_edpl
    （同上换成 --dataset locomo）

输出：
    outputs/retrieval_ablation/results_<dataset>_<config>.json
    （含逐题结果 + 汇总 + per-category + 审计字段）

开关实现（monkey-patch，不改动 run_item_v2 源码）：
    wo_profiling : trimem_ar_v2.profile_query -> 恒等全 1 画像（统一管线，全组件激活）
    wo_hbs       : knowledge_memory_v2.ta_hbs_search -> []（束搜关闭；
                   ⚠️ 2026-09-14 接线后主路径确实会调 ta_hbs_search，
                   但 vke_retrieve 还有“全量关键词扫描”兜底，所以 wo_hbs ≠ full，
                   它现在是一次真的“无束搜”消融，结果需重跑而非复用旧值）
    wo_edpl      : run_item_v2._register_error_pattern -> no-op + 清空 km.json
    wo_vke       : V2_WIRE_VKE=0（关闭 VKE 块注入 + HBS/剪枝/贝叶斯）
                   —— 仅摘 trimem 的 'VKE' 组件已不足够（内联注入不在那条路径上）
"""
import sys, os, json, time, argparse

BASE = os.environ.get('EVOMEM_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'hierarchyv2'))

KM_PATH = os.path.join(BASE, 'hierarchyv2', 'km.json')
LME120 = os.environ.get('LME_INPUT', os.path.join(BASE, 'data', 'lme_120_sample.json'))
LOC120_IN = os.path.join(BASE, 'hierarchyv2', 'outputs', 'retrieval_ablation', 'locomo120_input.json')
OUT_DIR = os.path.join(BASE, 'hierarchyv2', 'outputs', 'retrieval_ablation')


def clear_km():
    json.dump({'patterns': [], 'failures': [], 'pattern_v2': []},
              open(KM_PATH, 'w'), indent=2)


def apply_switch(config):
    """返回审计描述；在进程内 import 后调用"""
    from hierarchyv2 import run_item_v2 as riv2
    from hierarchyv2 import trimem_ar_v2
    from hierarchyv2 import knowledge_memory_v2 as kmv2
    if config == 'wo_profiling':
        def _uniform(q):
            return {'f_s': 1, 't_s': 1, 'd_i': 1}
        trimem_ar_v2.profile_query = _uniform
        riv2.profile_query = _uniform
        return 'profile_query -> uniform {f_s:1,t_s:1,d_i:1}'
    if config == 'wo_hbs':
        kmv2.ta_hbs_search = lambda *a, **k: []
        return 'ta_hbs_search -> [] (no beam)'
    if config == 'wo_edpl':
        riv2._register_error_pattern = lambda *a, **k: (None, None)
        return '_register_error_pattern -> no-op; km cleared'
    if config == 'wo_ipi':
        _orig_act = trimem_ar_v2.activate_components
        trimem_ar_v2.activate_components = lambda profile: _orig_act(profile) - {'IPI'}
        return 'activate_components -> IPI removed'
    if config == 'wo_eat':
        _orig_act2 = trimem_ar_v2.activate_components
        # EAT 参与 FEI 分支门控（'FEI' in active or 'EAT' in active），去掉 EAT 后强制保留 FEI，避免连带关掉基础检索
        trimem_ar_v2.activate_components = lambda profile: (_orig_act2(profile) - {'EAT'}) | {'FEI'}
        return 'activate_components -> EAT removed, FEI forced on'
    if config == 'wo_vke':
        # 2026-09-14 接线后：真正的 VKE 消融开关是 run_item_v2.WIRE_VKE
        # （仅减 activate_components 的 'VKE' 只关掉 trimem 的内联块，关不掉 _vke_context_block）
        _orig_act3 = trimem_ar_v2.activate_components
        trimem_ar_v2.activate_components = lambda profile: _orig_act3(profile) - {'VKE'}
        riv2.WIRE_VKE = False
        riv2.WIRE_HBS_WEIGHTS = False
        return 'activate_components -> VKE removed; WIRE_VKE/WIRE_HBS_WEIGHTS -> False'
    return 'no switch (full)'


def run(dataset, config):
    from hierarchyv2 import run_item_v2 as riv2
    if dataset == 'lme':
        data = json.load(open(LME120))
        tag = f'LongMemEval-{len(data)}'
    else:
        data = json.load(open(LOC120_IN))
        tag = 'LoCoMo-120'

    audit = apply_switch(config)
    clear_km()
    print(f'===== {config} @ {tag} ({len(data)} items) =====', flush=True)
    print(f'[switch] {audit}', flush=True)

    results = []
    correct = total = 0
    t_start = time.time()
    SUFFIX = os.environ.get('OUT_SUFFIX', '')
    ckpt_path = os.path.join(OUT_DIR, f'checkpoint_{dataset}_{config}{SUFFIX}.json')
    for idx, item in enumerate(data):
        qid = item.get('question_id', '?')
        qt = item.get('question_type', '?')
        t0 = time.time()
        try:
            r = riv2.run_item_v2(item)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {'correct': False, 'invalid': True, 'type': qt, 'predicted': ''}
        et = time.time() - t0
        if not r.get('invalid'):
            total += 1
            if r.get('correct'):
                correct += 1
        results.append({
            'idx': idx + 1, 'qid': qid, 'type': qt,
            'correct': bool(r.get('correct')), 'invalid': bool(r.get('invalid')),
            'predicted': str(r.get('predicted', ''))[:200],
            'elapsed': round(et, 2),
        })
        print(f'[{idx+1}/{len(data)}] {qid} ({qt}) -> {"PASS" if r.get("correct") else "FAIL"} {et:.1f}s', flush=True)
        if (idx + 1) % 20 == 0:
            acc = correct / total * 100 if total else 0
            print(f'  partial {idx+1}/{len(data)}: {correct}/{total} = {acc:.1f}%', flush=True)
            json.dump({'partial': True, 'idx': idx + 1, 'correct': correct, 'total': total,
                       'results': results}, open(ckpt_path, 'w'), ensure_ascii=False, indent=1)

    acc = correct / total * 100 if total else 0
    elapsed = time.time() - t_start
    # per-category
    by_cat = {}
    for r in results:
        by_cat.setdefault(r['type'], {'total': 0, 'correct': 0})
        if not r['invalid']:
            by_cat[r['type']]['total'] += 1
            by_cat[r['type']]['correct'] += int(r['correct'])
    cat_stats = {k: {**v, 'acc': round(v['correct'] / v['total'] * 100, 1) if v['total'] else None}
                 for k, v in by_cat.items()}

    summary = {
        'dataset': dataset, 'dataset_tag': tag, 'config': config,
        'switch_audit': audit,
        'total': total, 'correct': correct, 'accuracy': round(acc, 1),
        'elapsed_min': round(elapsed / 60, 1),
        'cat_stats': cat_stats,
        'n_items': len(results),
        'n_invalid': sum(1 for r in results if r['invalid']),
    }
    out = os.path.join(OUT_DIR, f'results_{dataset}_{config}{SUFFIX}.json')
    json.dump({'summary': summary, 'results': results}, open(out, 'w'), ensure_ascii=False, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print('=' * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f'saved -> {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['lme', 'locomo'], required=True)
    ap.add_argument('--config', choices=['full', 'wo_profiling', 'wo_hbs', 'wo_edpl', 'wo_ipi', 'wo_eat', 'wo_vke'], required=True)
    a = ap.parse_args()
    run(a.dataset, a.config)
