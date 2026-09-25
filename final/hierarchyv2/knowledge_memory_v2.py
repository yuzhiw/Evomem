#!/usr/bin/env python3
"""
knowledge_memory_v2.py — VKE (Versioned Knowledge Evolution) System v2

Five dynamic mechanisms:
1. TA-HBS (Type-Aware Heuristic Beam Search): beam search on EFG with adaptive beam width
2. Adaptive Weight Learning: error-driven weight adjustment (α, β, γ, δ)
3. Hyperbolic Decay Pruning: keep_score = f_conf · f_src · f_rec · f_acc
4. Bayesian Confidence Calibration: Beta-Binomial posterior update
5. Type-Aware Context Selection: format selection by question type
"""
import json, os, re, time, math
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

import sys
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from hierarchyv2.config import _STOP, _MS_COMMON_NAMES, KM_PATH
from hierarchyv2.embedding import embed
from hierarchyv2.llm_utils import call_llm


# ═══════════════════════════════════════════════════════════════════
# Type-Aware Heuristic Beam Search (TA-HBS)
# ═══════════════════════════════════════════════════════════════════

# Question types and their beam widths
QUESTION_TYPES = ['KU', 'TR', 'SS', 'PR', 'MS']
BEAM_WIDTHS = {'KU': 5, 'TR': 3, 'SS': 1, 'PR': 3, 'MS': 5}

# Default beam weights θ_Tq = (α, β, γ, δ)
DEFAULT_WEIGHTS = {
    'KU': (0.25, 0.35, 0.20, 0.20),
    'TR': (0.20, 0.15, 0.25, 0.40),
    'SS': (0.10, 0.50, 0.30, 0.10),
    'PR': (0.15, 0.40, 0.15, 0.30),
    'MS': (0.30, 0.30, 0.20, 0.20),
}

# Pruning thresholds θ_Tq
PRUNE_THRESHOLDS = {'KU': 0.2, 'TR': 0.6, 'SS': 0.6, 'PR': 0.2, 'MS': 0.2}

# Hyperbolic decay rate η_Tq
DECAY_RATES = {'KU': 0.003, 'TR': 0.008, 'SS': 0.02, 'PR': 0.003, 'MS': 0.008}

# ═══════════════════════════════════════════════════════════════════════════
# 2026-09-18 cod1（论文对齐 P2）— Eq.2 结构对齐项 Φ_struct(e_ij, T_q) = w_{T_q}^T φ(e_ij)
#   论文：“φ is a one-hot encoding of the creation, update, and deletion edge types
#   and w_{T_q} is a query-type-specific weight vector.”
#   原实现 phi_struct 恒为常量 0.5（论文机制 = 死值）。cod1 改为真的按入边类型打分：
#   入边类型由 EFG 版本链上下文推断（首版本=creation / 值变更=update / 值失效=deletion）。
#   w_T 可由开发集学习得到（见 hbs_learn.py），未学习时用 DEFAULT_EDGE_WEIGHTS。
# ═══════════════════════════════════════════════════════════════════════════
EDGE_TYPES = ['creation', 'update', 'deletion']
EDGE_TYPE_INDEX = {t: i for i, t in enumerate(EDGE_TYPES)}
EDGE_WEIGHTS = {
    'KU': (0.25, 0.55, 0.20),
    'TR': (0.25, 0.45, 0.30),
    'SS': (0.60, 0.30, 0.10),
    'PR': (0.30, 0.50, 0.20),
    'MS': (0.30, 0.50, 0.20),
}

# 学习到的论文参数 θ_T=(α,β,γ,δ) 与 w_T（由 hbs_learn.py 写入 JSON，测试阶段冻结）
_HBS_THETA = {'theta': {}, 'w': {}, 'loaded': False}


def load_hbs_theta(path: Optional[str] = None) -> Dict:
    """加载开发集学得的 HBS 参数（论文 App A.2：learned on dev split, frozen for test）。

    未提供或文件不存在时保持默认参数（行为退化到 DEFAULT_WEIGHTS / EDGE_WEIGHTS）。
    """
    global _HBS_THETA
    p = path or os.environ.get('HBS_THETA_PATH', '')
    if p and os.path.exists(p):
        try:
            d = json.load(open(p))
            _HBS_THETA = {'theta': d.get('theta', {}), 'w': d.get('w', {}), 'loaded': True}
            print(f'  [HBS] learned theta loaded from {p} '
                  f'({len(_HBS_THETA["theta"])} types)', flush=True)
        except Exception as e:
            print(f'  [HBS] theta load failed: {e}', flush=True)
    return _HBS_THETA


def theta_for(question_type: str, base_weights=None):
    """返回 (θ_T=(α,β,γ,δ), w_T)，学习值优先，否则默认值。"""
    _b = base_weights or DEFAULT_WEIGHTS.get(question_type, (0.25, 0.35, 0.20, 0.20))
    t = _HBS_THETA['theta'].get(question_type)
    th = tuple(t) if t else tuple(_b)
    w = _HBS_THETA['w'].get(question_type) or EDGE_WEIGHTS.get(question_type, (1/3, 1/3, 1/3))
    return th, tuple(w)


def classify_edge_type(v: Dict, efg) -> str:
    """推断版本节点的入边类型（论文 App A.6：creation / update / deletion）。

    - 值失效（空/N/A/none）        → deletion
    - 该 (subject, predicate) 链的首个版本 → creation
    - 其余（值被后续新值取代）      → update
    """
    obj = str(v.get('object', '')).strip()
    if obj == '' or obj.lower() in ('n/a', 'none', 'null', 'unknown'):
        return 'deletion'
    attr_key = (v.get('subject', ''), v.get('predicate', ''))
    chain = getattr(efg, 'version_chains', {}).get(attr_key, [])
    if not chain or chain[0] == v.get('id'):
        return 'creation'
    return 'update'


def compute_phi_struct(v: Dict, question_type: str, efg, w_T=None) -> float:
    """论文 Eq.2：Φ_struct(e_ij, T_q) = w_{T_q}^T φ(e_ij)，φ 为边类型 one-hot。"""
    if w_T is None:
        _, w_T = theta_for(question_type)
    et = classify_edge_type(v, efg)
    phi = [0.0, 0.0, 0.0]
    phi[EDGE_TYPE_INDEX[et]] = 1.0
    return float(sum(w_T[i] * phi[i] for i in range(len(phi))))


def classify_question_type_vke(query: str) -> str:
    """
    Classify query into VKE question types:
      KU = knowledge-update (current state, ongoing)
      TR = temporal-reasoning (time-based, intervals)
      SS = single-session (precise fact about one event)
      PR = preference (likes, favorites)
      MS = multi-session (counting across sessions)
    """
    q_lower = query.lower()
    
    # Temporal-reasoning
    if re.search(r'\b(how long|how many days|duration|between.*and|from.*to|when|what date|which year)\b', q_lower):
        if re.search(r'\b(when|what date|what time|which year)\b', q_lower):
            return 'TR'
        return 'TR'
    
    # Knowledge-update
    if any(w in q_lower for w in ['currently', 'so far', 'most recent', 'latest', 'now', 'how often', 'current', 'change']):
        return 'KU'
    if re.search(r'\b(has|have|does)\s+[a-z]+\s+(change|update|increase|decrease)\b', q_lower):
        return 'KU'
    
    # Preference
    if any(w in q_lower for w in ['favorite', 'prefer', 'preference', 'like', 'recommend', 'suggest']):
        return 'PR'
    
    # Multi-session (counting)
    if 'how many' in q_lower or 'how much' in q_lower:
        return 'MS'
    
    # Single-session (default for static facts)
    return 'SS'


# 2026-09-14 接线：主路径（run_item_v2 → TriMemAR_v2 → 本模块）现在会真正调用
# ta_hbs_search / vke_retrieve / should_prune / bayesian_confidence_calibrate。
# 调用条件：仅当题型 ∈ {knowledge-update, temporal-reasoning, multi-session,
# single-session-preference} 且 EFG 中存在该实体的版本链时（见
# WIRING_AND_CALL_CONDITIONS_20260914.md）。single-session-user/assistant 不调用。
#
# ⚠️ 实现与论文的分歧（接线时未改算法，只改调用与 now 传递，勿在论文里过度声明）：
#  1) [已修] phi_struct 原为常量 0.5 —— cod1 已改为论文 A.2 的 w_T^T φ（边类型 one-hot）；
#     权重更新仍走 adapt_weights（误差驱动增量），不是 KL 优化；w_T 可由 hbs_learn.py 在开发集上学；
#  2) 权重更新走 adapt_weights（误差类型 → 固定增量），不是 KL 优化；
#  3) c_cal 用 Beta 后验均值 (α0+k·κ)/(α0+k·κ+β0)，与论文 A.2 的
#     c_cal=(κ+s_c)/(κ+n_c) 形式不同（KAPPA=3 作为证据强度乘子）。
# 最近一次 vke_retrieve 的剪枝统计（日志/审计用）
LAST_VKE_STATS = {'candidates': 0, 'after_prune': 0, 'rescued': False}

HBS_MAX_CANDIDATES = int(os.environ.get('HBS_MAX_CANDIDATES', '200'))


def ta_hbs_search(query: str, efg, question_type: str,
                  weights: Optional[Tuple[float, float, float, float]] = None,
                  max_depth: int = 3,
                  now: Optional[datetime] = None,
                  max_candidates: Optional[int] = None) -> List[Dict]:
    """
    Type-Aware Heuristic Beam Search over EFG.
    
    Args:
        query: The question
        efg: EntityFactGraph instance
        question_type: One of KU, TR, SS, PR, MS
        weights: Optional (α, β, γ, δ) weights (uses default if None)
        max_depth: Maximum beam search depth
        now: 查询日期（用于 f_rec 时间衰减）。
             **必须由调用方传入题目 question_date**，否则 f_rec 会按服务器当前时间算，
             与 benchmark 的时间轴错位（2026-09-14 审计发现的缺陷）。
        max_candidates: 候选版本上限（默认 HBS_MAX_CANDIDATES=200），
             防止长会话下逐节点 embed 造成成本失控。
        
    Returns:
        List of selected version dicts, sorted by type-aware context selection
    """
    if max_candidates is None:
        max_candidates = HBS_MAX_CANDIDATES
    from hierarchyv2.trimem_ar_v2 import EntityFactGraph
    
    if weights is None:
        weights = DEFAULT_WEIGHTS.get(question_type, (0.25, 0.35, 0.20, 0.20))

    # 2026-09-18 cod1：θ_T 与 w_T 优先取开发集学得值（论文 App A.2），否则默认
    weights, _w_T = theta_for(question_type, weights)
    alpha, beta, gamma, delta = weights
    beam_width = BEAM_WIDTHS.get(question_type, 3)
    
    # ─── Step 1: Entity Retrieval ──────────────────────────────
    q_lower = query.lower()
    q_words = set(re.findall(r'[a-zA-Z]{3,}', q_lower)) - _STOP
    
    # Extract named entities from query
    q_entities = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
    common_ents = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
        'The','This','That','These','Those','My','Your','His','Her',
        'Its','Our','Their','Me','You','He','She','It','We','They'}
    q_entities = [e.lower() for e in q_entities if e not in common_ents]
    
    # Also check for entities in efg
    matched_entities = [e for e in q_entities if e in efg.entities]
    
    # If no entity matched by name, try keyword-based entity matching
    if not matched_entities:
        for entity_name in efg.entities:
            en_lower = entity_name.lower()
            if en_lower in q_lower or any(qw in en_lower for qw in q_words):
                matched_entities.append(entity_name)
                if len(matched_entities) >= 5:
                    break
    
    L1 = matched_entities if matched_entities else list(efg.entities.keys())[:10]
    
    # ─── Step 2: Expand Attributes ─────────────────────────────
    A = []  # attribute keys
    for entity in L1[:5]:
        attrs = list(efg.attribute_edges.get(entity, []))[:5]
        A.extend(attrs)
    
    # If no attribute matches, check by keyword overlap
    if not A:
        for entity in efg.entities:
            for pred, vid in efg._entity_to_versions.get(entity, []):
                v = efg.get_version(vid)
                if v:
                    obj_words = set(re.findall(r'[a-zA-Z]{3,}', v.get('object_lower', '')))
                    if q_words & obj_words:
                        A.append((entity, pred))
    
    # ─── Step 3: Expand Versions → Beam Search ─────────────────
    F_init = []  # initial version pool
    for attr_key in A[:beam_width * 2]:
        entity, pred = attr_key
        versions = efg.version_chains.get(attr_key, [])
        for vid in versions:
            v = efg.get_version(vid)
            if v:
                F_init.append(v)
    
    # 候选上限（2026-09-14 接线新增）：长版本链下逐节点 embed 会线性放大成本
    if len(F_init) > max_candidates:
        F_init = F_init[:max_candidates]

    # Compute query embedding for similarity scoring
    q_emb = np.array(embed([query])).flatten() if query else None
    
    V_sel = []
    for d in range(max_depth):
        if not F_init:
            break
        
        C = []  # candidates with scores
        for v in F_init:
            # Φ_struct: 结构对齐（论文 Eq.2: w_T^T φ(e_ij)，边类型 one-hot）
            # 2026-09-18 cod1：原为常量 0.5，现按入边类型（creation/update/deletion）打分
            phi_struct = compute_phi_struct(v, question_type, efg, _w_T)
            
            # sim: semantic similarity
            sim = 0.0
            if q_emb is not None:
                v_text = f"{v.get('object', '')} {v.get('confidence', '')}"
                v_emb = np.array(embed([v_text])).flatten()
                sim = float(np.dot(q_emb, v_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(v_emb) + 1e-10))
            
            # c_cal: calibrated confidence
            conf_str = v.get('confidence', 'medium')
            c_cal = bayesian_confidence_calibrate(conf_str, v.get('weight', 1.0))
            
            # f_rec: temporal recency（now=query date，见 ta_hbs_search docstring）
            f_rec = compute_recency(v.get('time', ''), now)
            
            # Composite score
            S = alpha * phi_struct + beta * sim + gamma * c_cal + delta * f_rec
            C.append((S, v))
        
        # Beam: keep top-B
        C.sort(key=lambda x: -x[0])
        F_init = [v for s, v in C[:beam_width]]
        V_sel.extend(F_init)
    
    # ─── 去重（2026-09-14 接线修复 A1）────────────────────────
    # 原实现每层都用 V_sel.extend(F_init)，同一版本节点会在 max_depth 层里
    # 被重复 append（默认 3 层 → 每个存活节点出现 3 次）：这会让 PR 分支的
    # frequency rank 把“1 次提及”写成“3x mentioned”（伪造证据），KU/MS 上下文
    # 也重复填充。在返回前按版本身份去重并保持首次出现顺序。
    _seen_keys = set()
    _dedup = []
    for _v in V_sel:
        _k = _v.get('id')
        if _k is None:
            _k = (str(_v.get('subject', '')), str(_v.get('predicate', '')),
                  _v.get('object_lower', _v.get('object', '')), str(_v.get('time', '')))
        if _k in _seen_keys:
            continue
        _seen_keys.add(_k)
        _dedup.append(_v)
    V_sel = _dedup

    # ─── Step 4: Type-Aware Context Selection ──────────────────
    V_sel = type_aware_context_selection(V_sel, question_type)
    
    return V_sel


def type_aware_context_selection(V_sel: List[Dict], question_type: str) -> List[Dict]:
    """
    Format selected versions based on question type.
    
    KU → chronological sort (full history)
    TR → filter by window δ=3 sessions
    SS → top-k by confidence
    PR → frequency rank
    MS → merge all versions
    """
    if not V_sel:
        return V_sel
    
    if question_type == 'KU':
        # Chronological sort
        V_sel.sort(key=lambda v: v.get('time', ''))
        return V_sel
    
    elif question_type == 'TR':
        # Filter by temporal window (keep versions within ±3 sessions)
        # Since session count is limited, keep all but sort by time
        V_sel.sort(key=lambda v: v.get('time', ''))
        return V_sel[:min(len(V_sel), 10)]
    
    elif question_type == 'SS':
        # Top-k by confidence
        def _conf_val(v):
            return {'high': 3.0, 'medium': 2.0, 'low': 1.0}.get(v.get('confidence', 'medium'), 1.0)
        V_sel.sort(key=lambda v: -_conf_val(v))
        return V_sel[:min(len(V_sel), 5)]
    
    elif question_type == 'PR':
        # Frequency rank (most common objects first)
        obj_counts = defaultdict(int)
        for v in V_sel:
            obj_counts[v.get('object_lower', '')] += 1
        V_sel.sort(key=lambda v: -obj_counts.get(v.get('object_lower', ''), 0))
        return V_sel
    
    elif question_type == 'MS':
        # Merge all versions (sort by time for dedup)
        return V_sel
    
    return V_sel


# ═══════════════════════════════════════════════════════════════════
# Adaptive Weight Learning
# ═══════════════════════════════════════════════════════════════════

def adapt_weights(error_type: str, question_type: str,
                  current_weights: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                  ) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Adapt beam weights based on error type.
    
    Error → adjustment:
      retrieval miss → (+0.05, 0, 0, +0.05)
      hallucination  → (0, 0, +0.05, 0)
      temporal error → (0, 0, 0, +0.10)
      precision error → (-0.05, 0, +0.05, 0)

    2026-09-14 接线：EDPL（run_item_v2.classify_error）产出的是
    retrieval_miss / format_error / reasoning_error / precision_error，
    与本表原键名不完全一致，故补充同名映射（专家裁定，非新机制）：
      reasoning_error → 同 hallucination（证据在但结论错 → 提高置信度权重）
      format_error    → 同 precision_error（内容对但形式错 → 提高精确度权重）

    All weights clamped to [0.05, 0.70]
    """
    if current_weights is None:
        current_weights = dict(DEFAULT_WEIGHTS)
    
    adjustments = {
        'retrieval_miss': (0.05, 0.0, 0.0, 0.05),
        'hallucination': (0.0, 0.0, 0.05, 0.0),
        'temporal_error': (0.0, 0.0, 0.0, 0.10),
        'precision_error': (-0.05, 0.0, 0.05, 0.0),
        # EDPL 直接可用（映射见 docstring）
        'reasoning_error': (0.0, 0.0, 0.05, 0.0),
        'format_error': (-0.05, 0.0, 0.05, 0.0),
    }
    
    adj = adjustments.get(error_type, (0.0, 0.0, 0.0, 0.0))
    if isinstance(question_type, str) and question_type not in current_weights:
        # 允许传入 benchmark 题型（knowledge-update 等）→ 归一化为 VKE 类型
        _qmap = {'knowledge-update': 'KU', 'temporal-reasoning': 'TR',
                 'multi-session': 'MS', 'single-session-preference': 'PR',
                 'single-session-user': 'SS', 'single-session-assistant': 'SS'}
        question_type = _qmap.get(question_type, question_type)
    
    if question_type in current_weights:
        old = current_weights[question_type]
        new = tuple(
            max(0.05, min(0.70, old[i] + adj[i]))
            for i in range(4)
        )
        current_weights[question_type] = new
    
    return current_weights


# ═══════════════════════════════════════════════════════════════════
# Hyperbolic Decay Pruning
# ═══════════════════════════════════════════════════════════════════

def compute_recency(time_str: str, now: Optional[datetime] = None) -> float:
    """Compute recency score f_rec(t, t_now) = 1/(1+η|t_now-t|) with η=0.003."""
    if not time_str:
        return 0.5
    
    if now is None:
        now = datetime.now()
    
    try:
        t = datetime.strptime(time_str[:10], '%Y-%m-%d') if len(time_str) >= 10 else datetime.strptime(time_str[:7], '%Y-%m') if len(time_str) >= 7 else None
        if t:
            days_diff = abs((now - t).days)
            return 1.0 / (1.0 + 0.003 * days_diff)
    except:
        pass
    
    return 0.5


def compute_keep_score(v: Dict, now: Optional[datetime] = None,
                       threshold_type: str = 'SS') -> float:
    """
    Compute keep_score for hyperbolic decay pruning.
    
    keep_score = f_conf(c) · f_src(w) · f_rec(t, t_now) · f_acc(t_last)
    
    f_conf(c) = {0.3, 0.7, 1.0} for {low, medium, high}
    f_src(w) = 1 - exp(-w)
    f_rec(t, t_now) = 1/(1+η|t_now-t|)
    f_acc(t_last) = 1/(1+0.01|t_now-t_last|)
    """
    # f_conf: confidence-based weight
    conf = v.get('confidence', 'medium')
    f_conf = {'low': 0.3, 'medium': 0.7, 'high': 1.0}.get(conf, 0.7)
    
    # f_src: source weight (w = corroborating sources)
    w = v.get('weight', 1.0)
    f_src = 1.0 - math.exp(-w)
    
    # f_rec: temporal recency
    f_rec = compute_recency(v.get('time', ''), now)
    
    # f_acc: access recency (when was it last accessed/confirmed)
    last_access = v.get('last_access', v.get('time', ''))
    if now is not None:
        try:
            t_last = datetime.strptime(last_access[:10], '%Y-%m-%d') if len(last_access) >= 10 else now
            days_since_access = abs((now - t_last).days)
            f_acc = 1.0 / (1.0 + 0.01 * days_since_access)
        except:
            f_acc = 0.5
    else:
        f_acc = 0.5
    
    keep_score = f_conf * f_src * f_rec * f_acc
    return keep_score


def should_prune(v: Dict, question_type: str = 'SS',
                 now: Optional[datetime] = None) -> bool:
    """
    Determine if a version node should be pruned.
    
    Prune if keep_score ≤ θ_Tq.
    θ_Tq ∈ {0.2, 0.6} — {KU, PR, MS} use 0.2, {TR, SS} use 0.6
    """
    threshold = PRUNE_THRESHOLDS.get(question_type, 0.2)
    score = compute_keep_score(v, now, question_type)
    return score <= threshold


# ═══════════════════════════════════════════════════════════════════
# Bayesian Confidence Calibration
# ═══════════════════════════════════════════════════════════════════

# Beta prior parameters (α_0, β_0) for {low, medium, high}
PRIORS = {
    'low': (2, 5),
    'medium': (5, 5),
    'high': (8, 2),
}
KAPPA = 3  # evidence strength multiplier


def bayesian_confidence_calibrate(confidence: str, k: float = 1.0) -> float:
    """
    2026-09-18 cod1（论文对齐 P3）— 论文 Eq.2 / App A.2：

        c_cal(n_j) = (κ + s_c) / (κ + m_c),   κ = 3

    论文原文：“maps the raw extraction confidence c to a calibrated level under a
    Beta(1+κ, 1) prior, c_cal = (κ+s_c)/(κ+m_c), where s_c and m_c accumulate
    agreement and total observations of the extractor's confidence level, and κ=3.”

    s_c = 置信度水平 × 观测数（一致证据量），m_c = 观测数（总证据量）。
    原实现用 Beta 后验均值 (α0+kκ)/(α0+kκ+β0)，与论文公式不同。
    """
    level = {'low': 0.3, 'medium': 0.7, 'high': 1.0}.get(confidence, 0.7)
    n_obs = max(1.0, float(k))
    s_c = level * n_obs
    m_c = n_obs
    return (KAPPA + s_c) / (KAPPA + m_c)


def is_genuine_divergence(v1: Dict, v2: Dict) -> bool:
    """
    Check if two versions represent genuine divergence (both c_cal > 0.5).
    If so, they are preserved as distinct E_evol nodes.
    """
    c1 = bayesian_confidence_calibrate(v1.get('confidence', 'medium'), v1.get('weight', 1.0))
    c2 = bayesian_confidence_calibrate(v2.get('confidence', 'medium'), v2.get('weight', 1.0))
    return c1 > 0.5 and c2 > 0.5 and v1.get('object_lower') != v2.get('object_lower')


# ═══════════════════════════════════════════════════════════════════
# VKE Query Interface
# ═══════════════════════════════════════════════════════════════════

def vke_retrieve(query: str, question_type: str, efg,
                 weights: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                 now: Optional[datetime] = None
                 ) -> List[Dict]:
    """
    Complete VKE retrieval pipeline.
    
    1. Classify question type (if not provided)
    2. Run TA-HBS beam search
    3. If beam search returns nothing, fallback to broader keyword scan
    4. Prune low-confidence results (never deletes stored versions)
    5. Apply type-aware context selection
    6. Return formatted versions

    now: 查询日期，透传给 ta_hbs_search/compute_keep_score，使时间衰减以题目日期为基准。
    """
    if not question_type:
        question_type = classify_question_type_vke(query)
    
    beam_weights = weights.get(question_type, DEFAULT_WEIGHTS[question_type]) if weights else DEFAULT_WEIGHTS.get(question_type, (0.25, 0.35, 0.20, 0.20))
    
    # Run beam search
    V_sel = ta_hbs_search(query, efg, question_type, beam_weights, now=now)
    
    # Fallback: if beam search returned nothing, do broader keyword scan of all versions
    if not V_sel:
        q_lower = query.lower()
        q_words = set(re.findall(r'[a-zA-Z]{3,}', q_lower)) - _STOP
        # Scan all EFG versions for keyword matches
        for vid, v in efg.versions.items():
            obj_words = set(re.findall(r'[a-zA-Z]{3,}', v.get('object_lower', '')))
            subj_words = set(re.findall(r'[a-zA-Z]{3,}', v.get('subject', '').lower()))
            raw_content = f"{v.get('object_lower', '')} {v.get('subject', '').lower()}"
            raw_words = set(re.findall(r'[a-zA-Z]{3,}', raw_content))
            all_words = obj_words | subj_words | raw_words
            overlap = len(q_words & all_words)
            if overlap >= 1:
                V_sel.append(v)
    
    # Prune low-confidence versions（只从检索集中剔除，不删除存储的版本）
    V_filtered = [v for v in V_sel if not should_prune(v, question_type, now)]
    # 2026-09-14：暴露剪枝统计，供 run_item_v2 打印 [_wire:vke] 明细（审查 B3：
    # 剪枝后节点数可能远小于候选数，需能从日志看出来，否则“版本链已注入”是假象）
    global LAST_VKE_STATS
    LAST_VKE_STATS = {'candidates': len(V_sel), 'after_prune': len(V_filtered),
                      'rescued': bool(not V_filtered and V_sel)}
    if not V_filtered and V_sel:
        V_filtered = V_sel[:5]
    
    # Type-aware context selection
    V_final = type_aware_context_selection(V_filtered, question_type)

    # 2026-09-18 cod1（论文对齐 P4）— Eq.3 第四项 f_acc = 1/(1+λΔt_access)，
    #   Δt_access 是“自该候选上次被检索以来的时长”。原实现中 v['last_access'] 从未被写入，
    #   导致 f_acc 恒等于按版本时间戳算的值（该项退化为与 f_rec 重复）。
    #   cod1：检索发生即打戳（以查询日期为基准），使该因子真正体现“最近被访问”。
    _stamp = (now or datetime.now())
    _stamp_s = _stamp.strftime('%Y-%m-%d') if hasattr(_stamp, 'strftime') else str(_stamp)[:10]
    for _v in V_final:
        _v['last_access'] = _stamp_s
        if isinstance(_v.get('id'), int) and _v['id'] in efg.versions:
            efg.versions[_v['id']]['last_access'] = _stamp_s

    return V_final


def vke_format_context(V_sel: List[Dict], question_type: str) -> str:
    """Format VKE results into context string."""
    if not V_sel:
        return "[VKE] No versioned knowledge found."
    
    lines = ['[VERSIONED KNOWLEDGE (VKE)]']
    
    if question_type == 'KU':
        lines.append('[Time-ordered history]')
        for v in V_sel[:10]:
            conf_str = v.get('confidence', 'medium')
            c_cal = bayesian_confidence_calibrate(conf_str, v.get('weight', 1.0))
            lines.append(
                f"  [{v.get('time', '?')}] {v.get('subject', '')} {v.get('predicate', '')}: "
                f"{v.get('object', '')[:60]} (conf_cal={c_cal:.2f})"
            )
    
    elif question_type == 'TR':
        lines.append('[Temporal window]')
        for v in V_sel[:8]:
            lines.append(f"  [{v.get('time', '?')}] {v.get('object', '')[:80]}")
    
    elif question_type == 'SS':
        lines.append('[Best evidence]')
        for v in V_sel[:3]:
            lines.append(f"  [{v.get('time', '?')}] {v.get('object', '')[:80]}")
    
    elif question_type == 'PR':
        lines.append('[Ranked by frequency]')
        # Group by object, count frequency
        obj_counts = defaultdict(int)
        obj_first_time = {}
        for v in V_sel:
            obj = v.get('object_lower', v.get('object', ''))
            obj_counts[obj] += 1
            time_str = v.get('time', '')
            if obj not in obj_first_time or time_str < obj_first_time[obj]:
                obj_first_time[obj] = time_str
        for obj, count in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
            first_time = obj_first_time.get(obj, '?')
            lines.append(f"  [{first_time}] {obj}: {count}x mentioned")
    
    elif question_type == 'MS':
        lines.append('[All versions merged]')
        for v in V_sel:
            lines.append(f"  [{v.get('time', '?')}] {v.get('object', '')[:80]}")
    
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# KM Compatibility Layer
# ═══════════════════════════════════════════════════════════════════

def _load_km():
    try:
        return json.load(open(KM_PATH))
    except:
        return {'patterns': [], 'failures': []}


def _save_km(km):
    os.makedirs(os.path.dirname(KM_PATH), exist_ok=True)
    json.dump(km, open(KM_PATH, 'w'), indent=2)


def _log_km_failure(question: str, correct, predicted, qtype: str, qid: str):
    """Log failure and learn from it (v2: includes VKE weight adaptation)."""
    km = _load_km()
    km['failures'].append({
        'qid': qid, 'qtype': qtype,
        'question': str(question)[:250],
        'correct': str(correct)[:150],
        'predicted': str(predicted)[:150],
        'time': datetime.now().isoformat()[:19],
    })
    km['failures'] = km['failures'][-200:]
    _save_km(km)
    _learn_from_failure(question, correct, predicted, qtype, qid)


def _learn_from_failure(question, correct, predicted, qtype, qid):
    """Analyze failure and learn error pattern (v2: classify by VKE error types)."""
    q = str(question)[:300]
    cor = str(correct)[:200]
    pred = str(predicted)[:200]
    if not cor or len(cor) < 2:
        return
    
    err_type = _classify_error_v2(q, cor, pred)
    
    km = _load_km()
    existing_ids = {p['id'] for p in km['patterns']}
    pattern_id = f'v2_auto_{qid}'
    if pattern_id in existing_ids:
        return
    
    pattern = {
        'id': pattern_id,
        'qtype': str(qtype),
        'error_type': err_type,
        'triggers': [w for w in re.findall(r'[a-zA-Z]{3,}', q.lower()) if w not in _STOP][:4],
        'fix_type': 'vke_weight_adaptation' if err_type in ('retrieval_miss', 'temporal_error') else 'reasoning_instruction',
        'description': f'v2 auto: {str(q)[:60]}',
        'source_qid': qid,
        'confidence': 1.0,
        'total_applied': 0,
        'total_success': 0,
    }
    km['patterns'].append(pattern)
    _save_km(km)
    print(f'    [VKE learn] {pattern_id}: {err_type}', flush=True)


# Output "no info" / "I don't know" phrases:
_NOINFO_PHRASES = [
    'no information', 'not mention', 'no mention', 'does not mention',
    'not specified', 'do not have', 'cannot determine', 'not explicitly',
    'there is no mention', 'there is no information', 'not found',
    'not provided', 'does not say', 'does not contain', 'not indicated',
]

# Stop words for trigger extraction:
_COMPUTE_SPECIFICITY_STOP = {
    'the','a','an','is','was','were','are','be','been','being','have','has','had',
    'do','does','did','done','get','got','gotten','make','made','go','went','gone',
    'say','said','see','saw','seen','come','came','take','took','taken','give',
    'gave','given','tell','told','ask','asked','use','used','using','want','wanted',
    'will','would','can','could','shall','should','may','might','must','need',
    'like','just','also','very','really','quite','too','much','many','some','any',
    'all','both','each','few','more','most','other','such','only','even','still',
    'already','yet','now','then','here','there','this','that','these','those',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your',
    'his','its','our','their','mine','yours','hers','its','ours','theirs',
    'not','no','nor','and','or','but','if','because','so','than','as','for',
    'with','about','into','over','after','before','between','through','during',
    'of','to','in','on','at','by','from','up','out','off','down','under','again',
    'further','once','well','back','very','dear','hi','hey','hello','bye','oh',
    'yeah','yes','ok','okay','sure','great','nice','good','bad'
}


def _classify_error_v2(question: str, correct: str, predicted: str) -> str:
    """Classify error type for VKE: retrieval_miss, temporal_error, precision_error, hallucination."""
    q = (question or '').lower()
    p = (predicted or '').lower()
    c = (correct or '').lower()
    
    # Retrieval miss: model says no info / doesn't know
    if any(ph in p for ph in _NOINFO_PHRASES):
        return 'retrieval_miss'
    
    # Temporal error: question asks about time but answer has no temporal reference
    temporal_keywords = ['when', 'how long', 'how many days', 'what date', 'what year', 'date', 'duration']
    if any(kw in q for kw in temporal_keywords):
        return 'temporal_error'
    
    # Hallucination: answer contains information not in question context
    # Simple heuristic: predicted has many words not in correct
    if c and p:
        c_words = set(re.findall(r'[a-zA-Z]{3,}', c))
        p_words = set(re.findall(r'[a-zA-Z]{3,}', p))
        if c_words and len(p_words - c_words) >= 3 * len(c_words):
            return 'hallucination'
    
    # Precision error: partial match
    if c and p:
        c_words = set(re.findall(r'[a-zA-Z]{3,}', c))
        p_words = set(re.findall(r'[a-zA-Z]{3,}', p))
        overlap = len(c_words & p_words)
        if overlap > 0 and overlap < len(c_words):
            return 'precision_error'
    
    return 'precision_error'


def _get_km_patterns(query: str, qtype: str) -> list:
    """Get learned patterns relevant to query (compatibility)."""
    km = _load_km()
    q_lower = query.lower()
    results = []
    for p in km.get('patterns', []):
        if p.get('deprecated'):
            continue
        for trigger in p.get('triggers', []):
            if trigger.lower() in q_lower:
                results.append(p)
                break
    return results


def _apply_km_patterns(text: str, date_str: str, session_id: str, km_patterns: list) -> list:
    """Apply learned extraction patterns (compatibility)."""
    facts = []
    for p in km_patterns:
        for ptn in p.get('extra_patterns', []):
            pattern = ptn['regex']
            pred = ptn.get('predicate', 'learned')
            conf = ptn.get('confidence', 'medium')
            try:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    obj = m.group(1).strip().rstrip('.,!?;:').strip() if m.groups() else m.group(0).strip()[:80]
                    if len(obj) >= 1:
                        facts.append({
                            'subject': 'user',
                            'predicate': pred,
                            'object': obj,
                            'object_lower': obj.lower(),
                            'time': date_str,
                            'session_id': session_id,
                            'confidence': conf,
                            'raw': f'[KM: {p.get("id", "?")}] {m.group(0)[:80]}',
                        })
            except Exception:
                pass
    return facts
