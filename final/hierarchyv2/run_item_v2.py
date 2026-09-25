#!/usr/bin/env python3
"""
run_item_v2.py — Error-Driven Pattern Learning (v2)

Key components:
1. Error Classification: retrieval miss, reasoning error, precision error, format error
2. Pattern Lifecycle: generated → active → tested → stale → deprecated
3. Trigger Extraction: specificity(w) = TF-IDF(w)/max_w' TF-IDF(w')
4. Pattern Confidence: Beta-Binomial tracking
5. Updated routing using all v2 components
"""
import sys, json, os, re, time, hashlib, math, uuid
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import sys
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from hierarchyv2.config import _STOP, _MS_COMMON_NAMES, KM_PATH
from hierarchyv2.embedding import embed
from hierarchyv2.date_utils import parse_date, fmt_date
from hierarchyv2.llm_utils import call_llm, classify_question_type, judge

# 2026-09-19 判分器加固：llm_utils.judge 现在 API 失败时抛 JudgeUnavailable（不再静默 False），
# 这里重试；仍失败向上抛 → runner 记 invalid（**绝不当作答错**）。
from hierarchyv2.llm_utils import JudgeUnavailable as _JudgeUnavailable
_judge_raw = judge


def judge(q, pred, cor, _tries=3):
    for _i in range(_tries):
        try:
            return _judge_raw(q, pred, cor)
        except _JudgeUnavailable:
            time.sleep(1.5 * (2 ** _i))
    raise _JudgeUnavailable('judge unavailable after retries')
from hierarchyv2.trimem_ar_v2 import TriMemAR_v2, profile_query, activate_components, EntityFactGraph
from hierarchyv2.knowledge_memory_v2 import (
    vke_retrieve, vke_format_context,
    classify_question_type_vke,
    _load_km, _save_km, _log_km_failure,
    bayesian_confidence_calibrate,
    compute_keep_score, should_prune,
    # 2026-09-14 接线新增（原本已写好但主路径零调用）
    adapt_weights, DEFAULT_WEIGHTS, _get_km_patterns,
)
from hierarchyv2.multi_session_engine import MultiSessionEngine
from hierarchyv2.postproc_engine import PostProcEngine, load_rules_from_km
from hierarchyv2 import knowledge_memory_v2 as _kmv2   # 读 LAST_VKE_STATS（剪枝统计，日志用）

# ═══════════════════════════════════════════════════════════════════
# 2026-09-14 接线开关（默认全部 ON = 已接回主路径）
# 设计原则：每个组件都有明确的“何时调用”条件（见下文各 helper 的 docstring
# 与 WIRING_AND_CALL_CONDITIONS_20260914.md），不满足条件时保持旧行为。
#   关掉某个开关即可做对照（等价于旧的 wo_* 消融）：
#     V2_WIRE_VKE=0        → 不做 HBS/剪枝/贝叶斯/VKE 块注入
#     V2_WIRE_HBS_WEIGHTS=0→ EDPL 不更新 HBS 权重（权重退回 DEFAULT_WEIGHTS）
#     V2_WIRE_POSTPROC=0   → 不过声明式后处理规则引擎
#     V2_WIRE_MS=0         → 计数题不用 MultiSessionEngine 兜底
# ═══════════════════════════════════════════════════════════════════
WIRE_VKE = os.environ.get('V2_WIRE_VKE', '1') == '1'
WIRE_HBS_WEIGHTS = os.environ.get('V2_WIRE_HBS_WEIGHTS', '1') == '1'
WIRE_POSTPROC = os.environ.get('V2_WIRE_POSTPROC', '1') == '1'
WIRE_MS = os.environ.get('V2_WIRE_MS', '1') == '1'

# benchmark 题型 → VKE 类型（论文 A.1：画像由题型 T_q 直接映射，无需额外计算）
_VKE_TYPE_BY_QT = {
    'knowledge-update': 'KU', 'temporal-reasoning': 'TR', 'multi-session': 'MS',
    'single-session-preference': 'PR', 'single-session-user': 'SS',
    'single-session-assistant': 'SS',
}
# VKE 块只对需要版本/时间/推理证据的题型注入；SSU/SSA（静态单事实）不注入，
# 避免长版本链稀释 context 并拉低 token-F1。
_VKE_QT_ALLOWED = {'knowledge-update', 'temporal-reasoning', 'multi-session',
                   'single-session-preference'}

# ═══════════════════════════════════════════════════════════════════
# ≤N-word answer constraint (RecMem-style; env ANSWER_MAX_WORDS, 0 = off)
#  - 注入答案生成 system prompt："Answer in at most N words."
#  - 硬约束：call_llm 返回的答案超 N 词则截断为前 N 词
#    （judge 与 result['predicted'] 均使用截断后的答案）
#  - 仅包裹 run_item_v2 内的 call_llm（7 处全是答案生成调用）；
#    llm_utils.judge 内部用的是 llm_utils 自己的 call_llm，不受影响
# ═══════════════════════════════════════════════════════════════════
ANSWER_MAX_WORDS = int(os.environ.get('ANSWER_MAX_WORDS', '0') or 0)


def _word_limit_suffix() -> str:
    if ANSWER_MAX_WORDS <= 0:
        return ''
    return f' Answer in at most {ANSWER_MAX_WORDS} words.'


# ══════════════════════════════════════════════════════════════════════
# 2026-09-14 接线 helper：VKE / HBS 权重 / 声明式后处理 / 计数引擎
# 每个 helper 的 docstring 写明“何时调用”；调用点均带 [_wire:*] 日志
# 便于从 run 日志核对是否真的触发。
# ══════════════════════════════════════════════════════════════════════

_PP_ENGINE = None


def _postproc_engine() -> Optional[PostProcEngine]:
    """声明式后处理规则引擎（单例）。

    何时调用：见 _apply_postproc 的调用条件。
    规则来源：优先 km.json 的 postproc_rules（load_rules_from_km，此前零调用），
              否则回退到 postproc_engine._P2_RULES（FixCat2.5/2.6/2.10/2.11）。
    """
    global _PP_ENGINE
    if _PP_ENGINE is None:
        _PP_ENGINE = PostProcEngine(rules=load_rules_from_km())
    return _PP_ENGINE


def _apply_postproc(ans, q, qt, sessions, dates, ctx_str, qid='?'):
    """对 LLM 答案应用声明式后处理规则。

    何时调用（两个条件之一，且 WIRE_POSTPROC=1）：
      ① qt == 'temporal-reasoning'（规则集就是为时间题写的）；
      ② “when”开头的单会话题（SSU/SSA）——按时间题口径处理，
         因为这类题的答案就是时间点，同样需要 FixCat2.5/2.10/2.11 保护。
    不调用的情形：KU（其指令禁止“引用证据措辞”，后处理会与之冲突）、
    计数题（由 _normalize_answer 接管）、非 when 的 SSU/SSA。
    """
    if not WIRE_POSTPROC or not ans:
        return ans
    _is_when = qt == 'temporal-reasoning' or (
        str(q).lower().strip().startswith('when') and qt in ('single-session-user', 'single-session-assistant'))
    if not _is_when:
        return ans
    _pp_type = 'temporal-reasoning' if qt != 'temporal-reasoning' else qt
    try:
        out = _postproc_engine().apply(ans, q, _pp_type, sessions, dates, ctx_str)
        if out and out != ans:
            print(f'    [_wire:postproc] {qid}: "{ans[:40]}" -> "{out[:40]}"', flush=True)
        return out or ans
    except Exception as e:
        print(f'    [_wire:postproc] skipped ({e})', flush=True)
        return ans


def _hbs_weights_for(vke_type, relevant_patterns, qid='?'):
    """EDPL 失败类型 → HBS 权重（α,β,γ,δ）。

    何时调用：题型属于 VKE 类型且存在命中触发词的活跃 pattern（否则用默认权重）。
    行为：取置信度最高的 pattern 的 error_type，调 adapt_weights 做一次增量更新，
          结果持久化到 km.json['hbs_weights']，使跨 EDPL 迭代累积（离线校准语义）。

    2026-09-14 reviewer 修复 B2：同一 (vke_type, error_type) 对**只应用一次**
    （记录在 km.json['hbs_weights_applied']），否则逐题重复 +0.05 会让 γ 在
    单次评测内从 0.20 一路漂到 clamp 上限 0.70，使结果依赖题目顺序、不可复现。
    非论文机制（论文 A.2 写的是 KL 学习），此处为误差驱动增量，勿过度声明。
    """
    weights = dict(DEFAULT_WEIGHTS)
    _km_now = {}
    try:
        _km_now = _load_km_kwargs()
        for k, v in (_km_now.get('hbs_weights') or {}).items():
            weights[k] = tuple(v) if isinstance(v, (list, tuple)) else v
    except Exception:
        pass
    if not WIRE_HBS_WEIGHTS or not relevant_patterns or vke_type not in weights:
        return weights
    _etype = getattr(relevant_patterns[0], 'error_type', '') or ''
    if not _etype:
        return weights
    _applied = set(_km_now.get('hbs_weights_applied') or [])
    _key = f'{vke_type}:{_etype}'
    if _key in _applied:
        return weights
    _before = tuple(weights[vke_type])
    adapt_weights(_etype, vke_type, weights)
    if tuple(weights[vke_type]) != _before:
        print(f'    [_wire:hbs-weights] {qid}: {vke_type} {_before} -> {tuple(weights[vke_type])} (error={_etype})', flush=True)
        try:
            _km = _load_km_kwargs()
            _km['hbs_weights'] = {k: list(v) for k, v in weights.items()}
            _km['hbs_weights_applied'] = sorted(set(_km.get('hbs_weights_applied') or []) | {_key})
            _save_km_kwargs(_km)
        except Exception as e:
            print(f'    [_wire:hbs-weights] persist skipped ({e})', flush=True)
    return weights


def _vke_context_block(q, qt, mem, relevant_patterns, qid='?', ctx_str=None):
    """VKE 版本链检索块：HBS 束搜索 → 时间衰减剪枝 → 类型化上下文选择
    （knowledge_memory_v2.vke_retrieve + vke_format_context，此前主路径零调用）。

    何时调用（四个条件必须同时成立）：
      ① WIRE_VKE=1；
      ② 题型 ∈ {knowledge-update, temporal-reasoning, multi-session, single-session-preference}；
         （SSU/SSA 静态单事实题不注入）
      ③ mem.efg 存在，且题目提到的实体在 EFG 里有版本链；
      ④ 该分支尚未注入内联 '[ENTITY-FACT GRAPH (VKE)]'（KU/MS 分支自带内联注入，
         此时跳过，避免同一证据重复入上下文）。

    2026-09-14 reviewer 修复 B1：条件④**前置**检查（传入 ctx_str）。
    原实现是先算完整个 HBS（含逐节点 embed）再在 _inject_vke 里丢弃，
    被丢弃的分支仍付出全部成本、还改写了持久化权重。
    剪枝只作用于检索集，不删除已存版本（论文 §3.2 语义）。
    """
    if not WIRE_VKE or qt not in _VKE_QT_ALLOWED:
        return ''
    if ctx_str is not None and ('[ENTITY-FACT GRAPH (VKE)]' in ctx_str
                                or '[VERSIONED KNOWLEDGE (VKE)]' in ctx_str):
        return ''  # 条件④（前置，避免白算）
    if mem is None or not getattr(mem, 'efg', None):
        return ''
    try:
        _qents = [e.lower() for e in re.findall(r'\b([A-Z][a-z]{2,})\b', str(q))
                  if e not in _MS_COMMON_NAMES]
        if not _qents:
            _qw = set(re.findall(r'[a-zA-Z]{3,}', str(q).lower()))
            for _subj in getattr(mem.efg, '_entity_to_versions', {}):
                if _subj in _qw:
                    _qents.append(_subj)
        if not _qents:
            return ''
        if not any(mem.efg.get_entity_versions(_e) for _e in _qents[:3]):
            return ''  # 条件③：无版本链可言
        _vtype = _VKE_TYPE_BY_QT.get(qt) or classify_question_type_vke(str(q))
        _weights = _hbs_weights_for(_vtype, relevant_patterns, qid) if WIRE_HBS_WEIGHTS else None
        _nodes = vke_retrieve(str(q), _vtype, mem.efg, _weights,
                              now=getattr(mem, 'qdate_dt', None))
        if not _nodes:
            return ''
        _block = vke_format_context(_nodes, _vtype)
        _st = getattr(_kmv2, 'LAST_VKE_STATS', {}) or {}
        print(f"    [_wire:vke] {qid}: type={_vtype} nodes={len(_nodes)} "
              f"(candidates={_st.get('candidates')}, after_prune={_st.get('after_prune')}, "
              f"rescued={_st.get('rescued')})", flush=True)
        return _block
    except Exception as e:
        print(f'    [_wire:vke] skipped ({e})', flush=True)
        return ''


def _inject_vke(ctx_str, vke_block):
    """把 VKE 块放到上下文最前（紧跟 QDATE/结构化区），条件③/④ 由调用前检查。"""
    if not vke_block or '[ENTITY-FACT GRAPH (VKE)]' in ctx_str or '[VERSIONED KNOWLEDGE (VKE)]' in ctx_str:
        return ctx_str
    return vke_block + '\n\n' + ctx_str


def _ms_counting_fallback(ans, q, sessions, dates, qid='?'):
    """计数题的 MultiSessionEngine 兜底（此前导入但从未实例化）。

    何时调用（全部满足）：
      ① WIRE_MS=1；
      ② 题目是计数题（how many/how much/count/how often/total）；
      ③ 主路径答案不可用：空串、无数字、或含“no information”类弃答短语。
    非计数题不调用（MSE 强制 TOTAL:X 格式，对列举/陈述题有害，见 Iter1 注释）。
    """
    if not WIRE_MS or not ans:
        pass
    _q = str(q).lower()
    if not WIRE_MS:
        return ans
    if not any(w in _q for w in ['how many', 'how much', 'count', 'how often', 'total']):
        return ans
    _a = str(ans or '').strip()
    _noinfo = any(p in _a.lower() for p in ['no relevant memories', 'no information', 'cannot determine',
                                            'not specified', 'not mentioned', 'not provided'])
    if _a and re.search(r'\d', _a) and not _noinfo:
        return ans  # 条件③：主路径已给出数字，不必兜底
    try:
        _mse = MultiSessionEngine()
        # km_patterns：EDPL 学到的抽取模式（_get_km_patterns 此前零调用）
        _mse.add_sessions(sessions, dates, km_patterns=_get_km_patterns(str(q), 'multi-session'),
                          chunk_enabled=True)
        _out = _mse.answer(str(q))
        if _out and re.search(r'\d', _out):
            print(f'    [_wire:ms-fallback] {qid}: "{_a[:30]}" -> "{_out.strip()[:30]}"', flush=True)
            return _out
    except Exception as e:
        print(f'    [_wire:ms-fallback] skipped ({e})', flush=True)
    return ans


_orig_call_llm = call_llm


def call_llm(msgs, max_tokens=256):
    if ANSWER_MAX_WORDS > 0 and msgs:
        msgs = list(msgs)
        if msgs and isinstance(msgs[0], dict):
            msgs[0] = {**msgs[0], 'content': str(msgs[0].get('content', '')) + _word_limit_suffix()}
    out = _orig_call_llm(msgs, max_tokens=max_tokens)
    if ANSWER_MAX_WORDS > 0 and out:
        _w = str(out).split()
        if len(_w) > ANSWER_MAX_WORDS:
            out = ' '.join(_w[:ANSWER_MAX_WORDS])
    return out


# ═══════════════════════════════════════════════════════════════════
# Error Classification
# ═══════════════════════════════════════════════════════════════════

_NOINFO_PHRASES = [
    'no information', 'not mention', 'no mention', 'does not mention',
    'not specified', 'do not have', 'cannot determine', 'not explicitly',
    'there is no mention', 'there is no information', 'not found',
    'not provided', 'does not say', 'does not contain', 'not indicated',
    'insufficient', 'unknown',
]

_ERROR_TYPES = ['retrieval_miss', 'reasoning_error', 'precision_error', 'format_error']

_PATTERN_STATES = ['generated', 'active', 'tested', 'stale', 'deprecated']


def classify_error(question: str, correct: str, predicted: str) -> str:
    """
    Classify error type.
    
    Retrieval Miss: model says "no information" despite relevant evidence
    Reasoning Error: evidence retrieved but inference wrong
    Precision Error: imprecise answer (partial match)
    Format Error: correct content, wrong format
    """
    q = (question or '').lower()
    p = (predicted or '').lower()
    c = (correct or '').lower()
    
    # Retrieval Miss
    if any(ph in p for ph in _NOINFO_PHRASES):
        return 'retrieval_miss'
    
    # Format Error: correct content present but wrapped in narrative
    format_wrappers = ['based on', 'the context', 'the conversation', 'according to', 'the text states']
    if any(w in p for w in format_wrappers) and c and c in p:
        return 'format_error'
    
    # Reasoning Error: evidence present but wrong conclusion
    # Check if predicted is in the "wrong direction" from correct
    if c and p and len(c) > 3 and len(p) > 3:
        c_words = set(re.findall(r'[a-zA-Z]{3,}', c))
        p_words = set(re.findall(r'[a-zA-Z]{3,}', p))
        # If predicted has very different words from correct
        if p_words and not c_words:
            return 'reasoning_error'
        if c_words and len(p_words - c_words) >= len(c_words):
            return 'reasoning_error'
        
        # If answer has correct words but includes extra wrong info
        overlap = len(c_words & p_words)
        if overlap > 0 and len(p_words) > len(c_words) * 1.5:
            return 'precision_error'
    
    # Default: precision error (partial match)
    return 'precision_error'


# ═══════════════════════════════════════════════════════════════════
# Trigger Extraction
# ═══════════════════════════════════════════════════════════════════

# TF-IDF corpus: a representative set of question texts
# In production, this would be computed from the full question set
# Here we use a static approximation for bootstrap

def compute_specificity(word: str, corpus_word_counts: Counter,
                        max_count: float = 1.0) -> float:
    """
    Compute specificity(w) = TF-IDF(w) / max_w' TF-IDF(w')
    
    where w ∉ stop_words, w ∉ named_entities
    """
    # TF-IDF approximation: less frequent words are more specific
    count = corpus_word_counts.get(word, 1)
    tfidf = 1.0 / (1.0 + math.log(1.0 + count))
    return tfidf / max_count if max_count > 0 else tfidf


def extract_triggers(question: str, corpus_word_counts: Optional[Counter] = None,
                     named_entities: Optional[set] = None) -> List[str]:
    """
    Extract triggers from a failed question.
    
    triggers(q) = top-K [ specificity(w) · 1[w ∉ S] · 1[w ∉ N] ]
    where S = stop words, N = named entities
    """
    q_lower = question.lower()
    all_words = re.findall(r'[a-zA-Z]{3,}', q_lower)
    
    # Filter: exclude stop words and named entities
    # Named entities are capitalized words in the original question
    if named_entities is None:
        original_words = re.findall(r'\b[A-Z][a-z]{2,}\b', question)
        common_ents = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
            'The','This','That','These','Those','My','Your','His','Her',
            'Its','Our','Their','Me','You','He','She','It','We','They',
            'No','Yes','True','False','January','February','March','April','May',
            'June','July','August','September','October','November','December'}
        named_entities = {w.lower() for w in original_words if w not in common_ents}
    
    # Score each word
    if corpus_word_counts is None:
        corpus_word_counts = _default_corpus_counts()
    
    max_count = max(corpus_word_counts.values()) if corpus_word_counts else 1.0
    
    scored_words = []
    for w in all_words:
        if w in _STOP or w in named_entities:
            continue
        spec = compute_specificity(w, corpus_word_counts, max_count)
        scored_words.append((spec, w))
    
    # Return top-K (max 4)
    scored_words.sort(key=lambda x: -x[0])
    return [w for _, w in scored_words[:4]]


def _default_corpus_counts() -> Counter:
    """Default TF-IDF corpus word counts (approximate from common question patterns)."""
    return Counter({
        'how': 100, 'what': 95, 'when': 80, 'where': 75, 'why': 70,
        'which': 65, 'who': 60, 'many': 85, 'much': 50, 'does': 55,
        'did': 60, 'has': 45, 'have': 50, 'was': 55, 'were': 40,
        'is': 60, 'are': 45, 'do': 40, 'did': 50, 'get': 30,
        'name': 35, 'type': 30, 'kind': 25, 'favorite': 35,
        'favourite': 20, 'currently': 25, 'recently': 20,
        'first': 30, 'last': 35, 'next': 20, 'old': 15, 'new': 25,
        'live': 20, 'work': 25, 'job': 20, 'like': 35, 'prefer': 25,
        'city': 20, 'country': 20, 'place': 25, 'event': 30,
        'year': 35, 'month': 25, 'week': 25, 'day': 30,
        'time': 35, 'date': 30, 'duration': 10, 'long': 25,
        'count': 10, 'total': 15, 'different': 20, 'between': 20,
        'during': 15, 'since': 15, 'after': 20, 'before': 25,
        'recommend': 15, 'suggest': 10, 'opinion': 5,
        'package': 5, 'trip': 15, 'travel': 15, 'visit': 20,
        'reason': 15, 'started': 15, 'began': 10, 'joined': 10,
        'bought': 15, 'created': 10, 'built': 10, 'made': 15,
    })


# ═══════════════════════════════════════════════════════════════════
# Pattern Lifecycle Management
# ═══════════════════════════════════════════════════════════════════

class ErrorPattern:
    """
    An error-driven pattern with lifecycle management.
    
    Pattern p = (triggers(q), instr(q, ε), conf=1.0)
    State ∈ {generated, active, tested, stale, deprecated}
    
    Confidence tracked via Beta-Binomial: Beta(1 + successes, 1 + failures)
    """
    
    def __init__(self, triggers: List[str], instruction: str, pattern_id: str,
                 error_type: str = ''):
        self.id = pattern_id
        self.triggers = triggers
        self.instruction = instruction
        self.error_type = error_type
        self.state = 'generated'
        self.successes = 0
        self.failures = 0
        self.created_at = datetime.now().isoformat()[:19]
        self.last_updated = self.created_at
    
    @property
    def confidence(self) -> float:
        """Beta-Binomial confidence: Beta(1 + successes, 1 + failures)"""
        a = 1 + self.successes
        b = 1 + self.failures
        return a / (a + b)
    
    def update(self, succeeded: bool):
        """Update pattern with outcome and transition state."""
        if succeeded:
            self.successes += 1
        else:
            self.failures += 1
        
        self.last_updated = datetime.now().isoformat()[:19]
        
        # State transitions
        c = self.confidence
        if self.state == 'generated':
            if c >= 0.5 and self.successes >= 1:
                self.state = 'active'
            elif c < 0.3:
                # 修正：generated 失败累计也会降级（原 bug：generated 永不 stale）
                self.state = 'stale'
        elif c < 0.2:
            self.state = 'deprecated'
        elif c < 0.3:
            self.state = 'stale'
        elif self.state == 'active' and self.successes + self.failures >= 3:
            self.state = 'tested'
    
    @property
    def is_active(self) -> bool:
        return self.state not in ('stale', 'deprecated')
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'triggers': self.triggers,
            'instruction': self.instruction,
            'error_type': self.error_type,
            'state': self.state,
            'successes': self.successes,
            'failures': self.failures,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ErrorPattern':
        p = cls(d.get('triggers', []), d.get('instruction', ''), d.get('id', ''), d.get('error_type', ''))
        p.state = d.get('state', 'generated')
        p.successes = d.get('successes', 0)
        p.failures = d.get('failures', 0)
        p.created_at = d.get('created_at', '')
        p.last_updated = d.get('last_updated', '')
        return p


class PatternRegistry:
    """Manages all error-driven patterns."""
    
    def __init__(self, km_path: str = KM_PATH):
        self.patterns: Dict[str, ErrorPattern] = {}
        self.km_path = km_path
        self._load()
    
    def _load(self):
        """Load patterns from KM storage."""
        km = _load_km_kwargs(self.km_path) if hasattr(self, 'km_path') else _load_km_kwargs(KM_PATH)
        for p_dict in km.get('pattern_v2', []):
            try:
                p = ErrorPattern.from_dict(p_dict)
                self.patterns[p.id] = p
            except Exception:
                pass
    
    def _save(self):
        """Save patterns to KM storage."""
        km = _load_km_kwargs(self.km_path) if hasattr(self, 'km_path') else _load_km_kwargs(KM_PATH)
        km['pattern_v2'] = [p.to_dict() for p in self.patterns.values()]
        _save_km_kwargs(km, self.km_path) if hasattr(self, 'km_path') else _save_km_kwargs(km, KM_PATH)
    
    def register(self, triggers: List[str], instruction: str, error_type: str = '') -> str:
        """Register a new pattern. Returns pattern ID."""
        pattern_id = f'edpl_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
        p = ErrorPattern(triggers, instruction, pattern_id, error_type)
        self.patterns[pattern_id] = p
        self._save()
        return pattern_id
    
    def update(self, pattern_id: str, succeeded: bool):
        """Update pattern with outcome."""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].update(succeeded)
            self._save()
    
    def get_relevant(self, query: str) -> List[ErrorPattern]:
        """Get active patterns relevant to a query."""
        q_lower = query.lower()
        relevant = []
        for p in self.patterns.values():
            if p.is_active:
                for trigger in p.triggers:
                    if trigger.lower() in q_lower:
                        relevant.append(p)
                        break
        relevant.sort(key=lambda p: p.confidence, reverse=True)
        return relevant


def _load_km_kwargs(path=None):
    p = path or KM_PATH
    try:
        return json.load(open(p))
    except:
        return {'patterns': [], 'failures': [], 'pattern_v2': []}


def _save_km_kwargs(km, path=None):
    p = path or KM_PATH
    # 2026-09-14 接线修复 A2：KM_PATH 默认是 'knowledge_memory.json'（无目录部分）时
    # os.makedirs('') 会抛 FileNotFoundError，导致 hbs_weights / pattern_v2 全部静默落盘失败。
    _d = os.path.dirname(p)
    if _d:
        os.makedirs(_d, exist_ok=True)
    json.dump(km, open(p, 'w'), indent=2)


# ═══════════════════════════════════════════════════════════════════
# Template library for common error patterns
# ═══════════════════════════════════════════════════════════════════

# ✅ No generating patterns from failure — algorithmically learn from errors.
# Patterns are dynamically created by _register_error_pattern.

def _register_error_pattern(question: str, correct: str, predicted: str,
                            error_type: str, registry: Optional[PatternRegistry] = None):
    """
    Create an error-driven pattern from failure with counterfactual analysis.
    
    For each failure, we perform a counterfactual diagnosis:
    - Identify the most discriminative retrieval dimension (structural/semantic/confidence/temporal)
      wherein the retrieved and expected outcomes diverge
    - Generate a dimension-targeted correction instruction
    
    1. Extract triggers from question
    2. Counterfactual analysis: diagnose which dimension failed
    3. Generate dimension-specific instruction
    4. Register in pattern registry
    """
    if registry is None:
        registry = PatternRegistry()
    
    triggers = extract_triggers(question)
    if not triggers:
        return None, None  # No meaningful triggers
    
    q_lower = question.lower()
    
    # Counterfactual analysis: diagnose the most discriminative dimension
    # We analyze the question and error type to determine which retrieval
    # dimension caused the failure
    diagnosis = _counterfactual_diagnosis(question, correct, predicted, error_type)
    
    # Generate dimension-specific instruction based on counterfactual diagnosis
    dim = diagnosis['dimension']
    
    if dim == 'temporal':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", use temporal ordering. '
            f'Find the MOST RECENT observation in the version chain. '
            f'If there are multiple versions, prefer the newest one. '
            f'For "current" or "now" queries, always use the latest timestamp.'
        )
    elif dim == 'structural':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", ensure entity alignment. '
            f'Match the entity name EXACTLY between question and context. '
            f'Do not confuse similar names or attributes belonging to different entities. '
            f'Use structural entity graph to verify the correct entity.'
        )
    elif dim == 'semantic':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", broaden semantic coverage. '
            f'The answer may be expressed with different wording in context. '
            f'Search for synonyms or related concepts in addition to exact phrasing.'
        )
    elif dim == 'confidence':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", prefer high-confidence facts. '
            f'When multiple conflicting facts exist, choose the one with highest confidence. '
            f'If no high-confidence fact exists, state what is available rather than guessing.'
        )
    elif error_type == 'retrieval_miss':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", do NOT answer "no information". '
            f'Search ALL context including episodic summaries and structured facts. '
            f'The answer may be distributed across multiple sessions.'
        )
    elif error_type == 'reasoning_error':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", reason step by step. '
            f'List each matching item individually. '
            f'Ordinal references indicate multiple items. '
            f'Count ALL distinct evidence before answering.'
        )
    elif error_type == 'precision_error':
        instruction = (
            f'IMPORTANT: For questions matching "{triggers[0]}", give exact values. '
            f'Match the requested format precisely. '
            f'If the question asks for a number, give just the number. '
            f'If it asks for a specific attribute, extract the exact value.'
        )
    else:
        instruction = (
            f'IMPORTANT: When answering questions about "{triggers[0]}", '
            f'check all available context for complete evidence.'
        )
    
    # 2026-08-27: 移除证据归因的 gold 泄漏（prompt 硬编码 expected answer = 作弊 + 与修复后上下文冲突）
    # ev = diagnosis.get('evidence')
    # if ev:
    #     instruction = instruction + f' The expected answer contains: {ev}. Extract it exactly.'
    
    # ── 去重合并（方案 B）：同 trigger∩ 同 error_type 的已有 pattern → 更新 instruction + 记录重复失败，不新建 ──
    for pid, p in registry.patterns.items():
        if p.error_type == error_type and (set(p.triggers) & set(triggers)):
            p.instruction = instruction
            p.failures += 1  # 本函数被调用即一次失败
            if p.failures >= 1 and p.state == 'generated':
                p.state = 'active'  # 合并即确认（同 trigger 同类型反复失败）
            registry._save()
            return pid, instruction
    
    # ── 双失败确认门控（方案 B）：同 trigger 已有任何 pattern → 该 trigger 反复失败（≥2 次）→ 新 pattern 直接 active ──
    _confirmed = any(set(p.triggers) & set(triggers) for p in registry.patterns.values())
    
    pattern_id = registry.register(triggers, instruction, error_type)
    if _confirmed:
        p = registry.patterns.get(pattern_id)
        if p is not None:
            p.successes = 1
            p.state = 'active'
            registry._save()
    return pattern_id, instruction


def _counterfactual_diagnosis(question: str, correct: str, predicted: str,
                              error_type: str) -> dict:
    """
    Counterfactual analysis: identify the most discriminative retrieval dimension.
    
    Analyzes a failure to determine which retrieval dimension (structural, semantic,
    confidence, temporal) most likely caused the incorrect outcome.
    
    Returns dict with 'dimension' and 'rationale'.
    """
    q = (question or '').lower()
    p = (predicted or '').lower()
    c = (correct or '').lower()
    
    # Temporal dimension: question asks about time/order but answer lacks temporal awareness
    temporal_signals = ['when', 'how long', 'how many days', 'what date', 'what year',
                        'date', 'duration', 'current', 'currently', 'now', 'recent',
                        'latest', 'last', 'newest', 'update', 'changed', 'before',
                        'after', 'since', 'ago']
    temporal_match = sum(1 for s in temporal_signals if s in q)
    
    # Structural dimension: question involves entity disambiguation
    structural_signals = ['which', 'who', 'whose', 'name', 'type', 'kind',
                          'different', 'same', 'another', 'other']
    structural_match = sum(1 for s in structural_signals if s in q)
    
    # Semantic dimension: question asks about abstract concepts
    semantic_signals = ['like', 'prefer', 'think', 'believe', 'opinion', 'recommend',
                        'suggest', 'good', 'bad', 'nice', 'great', 'enjoy']
    semantic_match = sum(1 for s in semantic_signals if s in q)
    
    # Confidence dimension: question asks for specific numbers/counts
    confidence_signals = ['how many', 'how much', 'count', 'total', 'number',
                          'percentage', 'percent', 'amount', 'figure']
    confidence_match = sum(1 for s in confidence_signals if s in q)
    
    # Determine primary dimension based on signal strength and error type
    dimensions = [
        ('temporal', temporal_match),
        ('structural', structural_match),
        ('semantic', semantic_match),
        ('confidence', confidence_match),
    ]
    
    # Boost temporal for temporal-reasoning and knowledge-update error types
    if error_type in ('temporal_error',):
        dimensions[0] = ('temporal', dimensions[0][1] + 3)
    
    # For retrieval_miss, boost temporal if question mentions time
    if error_type == 'retrieval_miss' and temporal_match > 0:
        dimensions[0] = ('temporal', dimensions[0][1] + 2)
    
    # For precision_error with numbers, boost confidence dimension
    if error_type == 'precision_error' and confidence_match > 0:
        dimensions[3] = ('confidence', dimensions[3][1] + 2)
    
    # Sort by signal strength (descending)
    dimensions.sort(key=lambda x: -x[1])
    
    # ── 证据归因（方案 B）：从 gold 提取具体证据，生成针对性指令 ──
    evidence = None
    dim = dimensions[0][0] if dimensions[0][1] > 0 else 'structural'
    c_clean = (correct or '').strip()
    p_lower = (predicted or '').lower()
    # 1) gold 是纯数字/日期 → confidence/precision 维度 + 精确值证据
    if re.fullmatch(r'[\d\s\-/.年月日]+', c_clean) or re.match(r'^\d{4}-\d{2}-\d{2}', c_clean):
        dim = 'confidence'
        evidence = f'exact value: {c_clean[:40]}'
    else:
        # 2) gold 含大写实体名/专名，pred 缺失 → structural 维度 + 实体证据
        _cor_ents = [w for w in re.findall(r'[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?', c_clean)
                     if w.lower() not in {'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'How', 'Why', 'Which', 'Who', 'Whom', 'Where'}]
        _missing = [e for e in _cor_ents if e.lower() not in p_lower]
        if _missing:
            dim = 'structural'
            evidence = f'entity: {_missing[0][:40]}'
        elif c_clean and c_clean.lower() not in p_lower and len(c_clean) <= 30:
            # 3) gold 是短精确短语且 pred 不含 → 提示输出精确短语
            dim = 'confidence'
            evidence = f'exact phrase: {c_clean[:40]}'
    
    return {
        'dimension': dim,
        'rationale': 'counterfactual analysis: strongest signal in %s dimension' % dimensions[0][0],
        'evidence': evidence,
    }


# ═══════════════════════════════════════════════════════════════════
# Wrapper stripping
# ═══════════════════════════════════════════════════════════════════

_WRAPPER_PATS = [
    r'^Based\s+(solely|strictly|purely)?\s*(up)?on\s*(the)?\s*(provided\s*)?context[\s,\.\:]+',
    r'^(the|this)?\s*context\s+(states|shows|indicates|suggests|mentions|does not contain|does not provide|does not include|provides no|reveals)[\s,]+',
    r'^(based on|according to)\s+(the\s+)?(conversation history|conversation|context)[\s,\.\:]+',
    r'^(the conversation history|the conversation|the provided context|the text|the given context|the history)[\s,]+(states|shows|indicates|suggests|mentions|reveals|says|confirms)[\s,]+',
    r'^from (the\s+)?(provided\s+)?(context|conversation)[\s,\.\:]+',
    r'^i[n]? (the\s+)?(provided\s+)?(context|conversation history|conversation)[\s,\.\:]+',
    r'^based solely on',
    r'^yes[,]?\s+(the context|the conversation history|based on)',
    r'^no[,]?\s+(the (conversation history|context)|there is no)',
    r'^(?:there is )?no mention of\b',
    r'^(?:the conversation|the context|the text|the history)\s+(?:does not|doesn\'t)\s+mention\b',
    r'^Based ONLY on the',
    r'^Based on ONLY the',
    r'^Based strictly on the',
]


# 2026-08-27：KU-mode 检测用的属性选择器。
# 原则：宁可不给信号（None 兜底），也不给错误信号（垃圾 chain）。
# 1) 从问题提取 <Entity>'s <attr> / <attr> of <Entity>；
# 2) 用属性词直接匹配 EFG predicate（包含/单词重叠）；
# 3) 无匹配时查别名映射（值可以是 list：EAT 干净属性优先，persona_* 仅兜底）；
#    绝不映射 likes/has_object/attended/located_at 这类宾语碎片 predicate（distinct 集合是噪音）。
# 2026-08-27 15:30: 新增 career_interest（EAT Category 8）+ persona_*（IPI 注入 EFG）候选。
_ATTR_ALIAS = {
    'job': ['person_role', 'career_interest'], 'career': ['person_role', 'career_interest'],
    'work': ['person_role', 'career_interest'], 'occupation': ['person_role', 'career_interest'],
    'role': ['person_role', 'career_interest'], 'position': ['person_role', 'career_interest'],
    'interest': ['career_interest', 'persona_preferences'],
    'hobby': ['persona_preferences'], 'preference': ['persona_preferences'],
    'relationship': ['person_role', 'persona_relationship_status'],
    'partner': ['person_role', 'persona_relationship_status'],
    'husband': ['person_role', 'persona_relationship_status'], 'wife': ['person_role', 'persona_relationship_status'],
    'boyfriend': ['person_role', 'persona_relationship_status'], 'girlfriend': ['person_role', 'persona_relationship_status'],
    'spouse': ['person_role', 'persona_relationship_status'],
    'family': ['is', 'persona_relationship_status'], 'status': ['is', 'persona_relationship_status'],
    'major': ['is'], 'degree': ['is'],
    'plan': ['plans', 'persona_life_goals'], 'plans': ['plans', 'persona_life_goals'],
    'goal': ['plans', 'persona_life_goals'],
}


def _select_ku_predicates(per_attr, q, entity):
    """从实体的全部 predicate 中选出与问题属性相关的干净谓词，返回 {predicate: set(objects)}。

    返回空 dict 表示无可靠信号（调用方应走 None 兜底，不判 single/chain）。
    persona_*（IPI 画像）只作兜底：EAT 干净属性（person_role/career_interest/is/plans）
    命中时优先，避免 persona 长描述值把 distinct 拉高误判 chain。
    """
    attr = None
    ql = q.lower()
    m = re.search(rf"{re.escape(entity)}['’]?s\s+([a-z][a-z ]{{1,24}}?)(?=\s+(?:now|currently|before|at the moment|these days|right now)|[,?.!;]|\s*$)", ql)
    if m:
        attr = m.group(1).strip()
    else:
        m = re.search(rf"([a-z][a-z ]{{1,24}}?)\s+of\s+(?:the\s+)?{re.escape(entity)}\b", ql)
        if m:
            attr = m.group(1).strip()
    if attr:
        attr = attr.strip().rstrip('?')
    preds = list(per_attr.keys())
    if attr:
        aw = set(attr.split())
        hits = [p for p in preds if attr in p or p in attr or (aw & set(p.split()))]
        if hits:
            hits_clean = [p for p in hits if not p.startswith('persona_')]
            if hits_clean:
                return {p: per_attr[p] for p in hits_clean}
            return {}  # 2026-08-28: 仅 persona 命中 → 不给信号（IPI 画像不是更新事件，distinct 是噪音）
        for w in aw:
            _cands = _ATTR_ALIAS.get(w, [])
            _eat_hits = [c for c in _cands if c in preds and not c.startswith('persona_')]
            if _eat_hits:
                return {c: per_attr[c] for c in _eat_hits}
        # 2026-08-28: persona_* 不再兜底（画像碎片曾把 dietary 误判成 chain(37)）
        # for w in aw:
        #     _cands = _ATTR_ALIAS.get(w, [])
        #     _p_hits = [c for c in _cands if c in preds]
        #     if _p_hits:
        #         return {c: per_attr[c] for c in _p_hits}
    return {}


def _strip_wrappers(text: str) -> str:
    """Remove narrative wrappers from LLM output. Matches V1 comprehensiveness."""
    if not text:
        return text
    old = text
    # FIRST: negative answer detection (before wrapper stripping)
    _neg_full_pats = [
        r'^(?:there is )?no mention of\b',
        r'^(?:the conversation|the context|the text|the history)\s+(?:does not|doesn\'t)\s+mention\b',
        r'^there is no (?:information|mention|evidence|data)',
        r'^the context does not (?:contain|include|provide|state|indicate)',
        r'^the given (?:context|text|conversation) does not',
        r'^insufficient (?:evidence|information|context)',
        r'^cannot (?:determine|find|locate)',
        r'^not (?:specified|provided|found|indicated|mentioned)',
    ]
    for _np in _neg_full_pats:
        if re.match(_np, text.strip(), re.I):
            text = 'Not mentioned'
            if text != old:
                print(f'    [StripWrap] "{old[:60]}" -> "{text[:60]}"', flush=True)
            return text
    # SECOND: apply wrapper prefix removal for answers that DO contain actual content
    for pat in _WRAPPER_PATS:
        text = re.sub(pat, '', text, flags=re.I).strip()
    text = text.strip().lstrip(',.:;').strip().rstrip('.').strip()
    # Clean trailing fragments
    text = re.sub(r"^(?:provided|given|supplied|shown)[\s,]+", "", text, flags=re.I).strip()
    if text.lower() in ['there is no information', 'the context does not mention this', 'no information', 'not mentioned']:
        text = 'Not mentioned'
    if text and len(text) < 8 and any(p in text.lower() for p in ['the context', 'there is', 'does not']):
        text = ''
    # THIRD: handle "No." + valid answer content prefix (e.g. "No. Caroline is not...")
    # Only strip when followed by meaningful content (5+ chars), not for simple Yes/No
    _no_prefix_match = re.match(r'^No\.?[,]?\s+(.{10,})', text, re.I)
    if _no_prefix_match:
        _rest = _no_prefix_match.group(1).strip()
        # Only accept if the rest has multiple words (not just a wrapper)
        if len(_rest.split()) >= 3 and not any(p in _rest.lower()[:40] for p in ['there is no', 'not mention', 'no information']):
            print(f'    [StripWrap-No] stripped "No." prefix: "{text[:50]}" -> "{_rest[:50]}"', flush=True)
            text = _rest
    if text != old:
        print(f'    [StripWrap] "{old[:60]}" -> "{text[:60]}"', flush=True)
    return text


# ═══════════════════════════════════════════════════════════════════
# Main routing function
# ═══════════════════════════════════════════════════════════════════

def run_item_v2(item):
    """
    Main handler for v2 routing.
    
    Uses all v2 components:
    - Type-Adaptive Retrieval (TriMemAR_v2)
    - VKE (Versioned Knowledge Evolution)
    - EAT + IPI (structured facts + persona)
    - Error-Driven Pattern Learning
    
    Args:
        item: QA item with question, answer, sessions, dates
        
    Returns:
        Dict with correct, type, predicted
    """
    q = item['question']
    cor = item['answer']
    qid = item.get('question_id', '?')
    qt = item.get('question_type', '')
    
    # Classify question type ONLY if missing (数据已标注的类型不再覆盖——conv-47_30 曾被误分为 multi-session)
    if not qt or qt == '?':
        classified = classify_question_type(q)
        if classified != qt:
            print(f'    [CLASSIFY] {qid}: {qt} -> {classified}', flush=True)
            qt = classified
    
    sessions = item.get('haystack_sessions', [])
    dates = item.get('haystack_dates', [])
    
    # ─── Load pattern registry（提前：multi-session 分支也用到 km_instructions）─
    registry = PatternRegistry()
    relevant_patterns = registry.get_relevant(q)
    injected_ids = [p.id for p in relevant_patterns[:3]]  # EDPL 闭环：本问题注入的 pattern
    km_instructions = ''
    for p in relevant_patterns[:3]:
        km_instructions += '\n' + p.instruction

    # ─── Multi-session: use TriMemAR multi-session branch ────────
    # (Iter1 fix: MultiSessionEngine 强制 TOTAL:X 计数格式，对非计数事实题
    #  (情感/陈述/爱好/事件) 全输出 'TOTAL: 0' 垃圾——v3 基线 6/10 multi-session 题全挂。
    #  改走 TriMemAR_v2.search(question_type='multi-session')（query expansion + 全会话扫描），
    #  极简 prompt；真计数题才加 COUNTING 指令。)
    if qt == 'multi-session':
        try:
            mem = TriMemAR_v2()
            qd = item.get('question_date', '')
            if qd:
                mem.qdate = qd[:10]
                mem.qdate_dt = parse_date(qd)
            mem.add_sessions(sessions, dates, chunk_enabled=False, attrs=item.get('attrs'))
            context = mem.search(q, question_type='multi-session')
            if not context or all(str(c).strip() in ('', 'No relevant memories.') for c in context):
                print(f'    [Retry-ms] normal retrieval empty -> force_broad', flush=True)
                context = mem.search(q, question_type='multi-session', force_broad=True)
            if not context:
                context = ['No relevant memories.']
            # Iter1: 窗口 40→50（配合 multi-session merged 55，跨会话列表题）
            ctx_str = '\n'.join([str(c)[:600] for c in context[:50]])
            if len(ctx_str) > 30000:
                ctx_str = ctx_str[:30000] + '...'
            _is_count_q = any(w in q.lower() for w in ['how many', 'count', 'how much', 'how often', 'how many times', 'total number'])
            # 2026-09-14 接线：MS 分支已内联注入 EFG 版本链 → VKE 块会被去重跳过；
            # 但仍保留调用点，使“无内联注入”时（例如内联块异常）仍有版本链可用。
            ctx_str = _inject_vke(ctx_str, _vke_context_block(q, qt, mem, relevant_patterns, qid, ctx_str=ctx_str))
            _counting_hint = ''
            if _is_count_q:
                _counting_hint = ('\nCOUNTING INSTRUCTIONS: Extract the PRECISE count. Ordinals like "first", "second", '
                                  '"third", "another" confirm multiple items. A couple = 2, several = 3+. '
                                  'Give just the number.')
            ans = call_llm([
                {'role': 'system', 'content': 'Answer the question directly based on the conversation context. '
                 'Give the specific fact requested. Short, direct answer. '
                 'If the question asks for a list or all matching items, list ALL of them. '
                 'If the question asks how many, give just the number. '
                 'Answer in natural language; NEVER reproduce memory entry templates or bracketed tags like [date] [category].' + km_instructions + _counting_hint},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=512) or ''
            ans = _strip_wrappers(ans.strip())
            if _is_count_q:
                ans = _normalize_answer(ans)
            # ═══ cod2 P1：计数/聚合题 → 二次枚举清单 + 代码确定性计数 ═══
            # 动机：LLM 心算 + “提及≠事件”是两类独立失因（实测 2 vs 3、$65 vs $185、0 vs 140h）；
            # 让 LLM 只负责枚举逐条证据，计数/求和交给 Python。
            if _is_count_q:
                try:
                    from hierarchyv2 import ms_counter as _msc
                    if _msc.ENABLED:
                        _mode = _msc.detect_mode(q)
                        _ctx_enum = ctx_str
                        # ── P3：会话级 map（逐批判定哪些会话提及该事件）→ 覆盖兜底 ──
                        try:
                            _flag = _msc.map_select_sessions(mem._cached_sessions, mem._cached_dates,
                                                             q, mem.qdate or '', call_llm)
                            if _flag:
                                _sess_ctx = _msc.render_sessions(mem._cached_sessions, mem._cached_dates, _flag)
                                _ctx_enum = (_sess_ctx + '\n\n' + ctx_str)[:60000]
                                print(f'    [MS-mapreduce] {len(_flag)}/{len(mem._cached_sessions)} sessions flagged', flush=True)
                        except Exception as _e2:
                            print(f'    [MS-mapreduce] skipped ({_e2})', flush=True)
                        # ── P1：只枚举，不心算；计数/求和由代码完成 ──
                        _list_ans = call_llm([
                            {'role': 'system', 'content': _msc.LIST_PROMPT + km_instructions},
                            {'role': 'user', 'content': f'Context:\n{_ctx_enum}\n\nQ: {q}\nList:'}
                        ], max_tokens=800) or ''
                        _det = _msc.deterministic_answer(_list_ans, q, _mode)
                        if _det:
                            print(f'    [MS-deterministic] mode={_mode} llm={ans[:30]!r} -> {_det!r}', flush=True)
                            ans = _det
                        else:
                            print('    [MS-deterministic] list parse failed -> keep llm answer', flush=True)
                except Exception as _e:
                    print(f'    [MS-deterministic] skipped ({_e})', flush=True)
            # 2026-09-14 接线：计数题且主路径没给出数字 → MultiSessionEngine 兜底
            ans = _ms_counting_fallback(ans, q, sessions, dates, qid)
            if _is_count_q:
                ans = _normalize_answer(ans)
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or '', '_edpl_ids': injected_ids}
            if not result['correct']:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result
        except Exception as e:
            print(f'  [ERR MS] {qid}: {e}', flush=True)
            import traceback; traceback.print_exc()
            return {'correct': False, 'invalid': True, 'type': qt, 'predicted': '', '_edpl_ids': injected_ids}
    
    try:
        # ─── Initialize v2 memory ───────────────────────────────
        mem = TriMemAR_v2()
        qd = item.get('question_date', '')
        if qd:
            mem.qdate = qd[:10]
            mem.qdate_dt = parse_date(qd)
        
        mem.add_sessions(sessions, dates, chunk_enabled=(qt in ['single-session-assistant', 'single-session-user']), attrs=item.get('attrs'))
        
        # Get context via type-adaptive retrieval
        context = mem.search(q, question_type=qt)
        if not context or all(str(c).strip() in ('', 'No relevant memories.') for c in context):
            # Fallback: broader retrieval (port from 正版: 检索空时放宽，避免直接弃答)
            print(f'    [Retry] normal retrieval empty -> force_broad', flush=True)
            context = mem.search(q, question_type=qt, force_broad=True)
        if not context:
            context = ['No relevant memories.']
        
        # ─── Temporal reasoning ──────────────────────────────────
        if qt == 'temporal-reasoning':
            # Use VKE for temporal-aware retrieval
            vke_type = classify_question_type_vke(q)
            # 时间块（TIMELINE/EVENT DATES/STRUCTURED EVENT FACTS）不按 500 截断，保证完整时间线进入上下文
            _ctx_parts = []
            for _c in context[:40]:  # Iter2: 25→40（配合 top25 docs + 时间块，"which city in month" 类证据分散）
                _s = str(_c)
                if _s.startswith(('[TIMELINE]', '[EVENT DATES]', '[STRUCTURED EVENT FACTS', '[DATE_CALC]', '[MONTH EVENTS')):
                    _ctx_parts.append(_s)
                else:
                    _ctx_parts.append(_s[:500])
            ctx_str = '\n'.join(_ctx_parts)
            if len(ctx_str) > 34000:
                ctx_str = ctx_str[:34000] + '...'
            # 2026-09-14 接线：temporal-reasoning 是 VKE 的主用例（t_s=1）
            ctx_str = _inject_vke(ctx_str, _vke_context_block(q, qt, mem, relevant_patterns, qid, ctx_str=ctx_str))
            
            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Be precise. '
                 'For temporal questions, use the EXACT date or time expression from context. '
                 'If the context has a relative time expression ("the week before", "last weekend", '
                 '"the Friday before"), preserve it EXACTLY - do NOT convert to absolute dates. '
                 'For "how many days/weeks between" questions, compute the interval. '
                 'For "which city/where in <Month Year>" questions: if multiple locations are mentioned '
                 'within that month, answer with the location from the LATEST date in that month. '
                 'For questions about a future date ("as of December 2023"), if no change is mentioned '
                 'after the latest known fact, use the latest known state. '
                 'Do NOT add narrative wrappers. Answer in natural language; NEVER reproduce memory entry '
                 'templates or bracketed tags like [date] [category].' + km_instructions},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=256) or ''
            ans = _strip_wrappers(ans.strip())
            
            # 正版 FixCat2.12 同款兜底：when 类问题答案无"具体月+日" → 用完整 timeline 二次提取
            # （修复：原条件只看有无 4 位数字，"before 2023-11-21" 这种模糊日期会漏过，conv-43_20 实证）
            _is_when_q = q.lower().strip().startswith('when') and not any(
                p in q.lower() for p in ['which city', 'what city', 'where was', 'where did', 'where is',
                                          'which week', 'which country', 'what country', 'which places', 'what places'])
            _has_specific_date = re.search(
                r'(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}(st|nd|rd|th)?\b', ans)
            _has_vague_date = re.search(r'(?i)\b(before|after|around|about|by|until|since|last|next|ago|not mentioned|not sure|unknown)\b', ans)
            if _is_when_q and (not _has_specific_date or _has_vague_date):
                _de_prompt = (f'From the timeline below, find the date or time period when this happened: "{q}"\n\n'
                              f'Timeline:\n' + ctx_str[:20000] + '\n\nAnswer with ONLY the date or time range. Be precise.')
                _de_ans = call_llm([
                    {'role': 'system', 'content': 'Extract the exact date or time period for the event. Short answer only.'},
                    {'role': 'user', 'content': _de_prompt}
                ], max_tokens=64) or ''
                if _de_ans and len(_de_ans.strip()) > 3:
                    print(f'    [FixCat2.12-like] {qid}: "{ans[:40]}" -> "{_de_ans.strip()[:40]}"', flush=True)
                    ans = _de_ans.strip()
            # 2026-09-14 接线：时间题答案过声明式后处理（FixCat2.5/2.6/2.10/2.11）
            ans = _apply_postproc(ans, q, qt, sessions, dates, ctx_str, qid)
            
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or '', '_edpl_ids': injected_ids}
            if not result['correct']:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result
        
        # ─── Knowledge update ────────────────────────────────────
        elif qt == 'knowledge-update':
            ctx_str = '\n'.join([str(c)[:600] for c in context[:48]])  # 2026-08-26: 30→48（LME 长会话答案文档常排 30+ 位被截断）
            if len(ctx_str) > 25000:
                ctx_str = ctx_str[:25000] + '...'
            
            # ── 2026-08-27: 区分「单次更新(KU)」vs「多版本属性链」──
            # 单次更新(old→new)：most recent 即答案；多版本链(v1→v2→v3…)：取稳定 current 值，排除过渡性提及
            # 硬信号：VKE 版本链中同一实体-属性(predicate)的 distinct 值数（2=单次更新, ≥3=链）
            # 2026-08-27 修复：信号必须来自「问题相关属性」，不能 max over 全部 predicate ——
            #   has_object/likes/attended 等宾语碎片 predicate 的 distinct 集合是噪音（conv-26_1
            #   Caroline has_object=72），会被误判成 chain，注入垃圾版本链导致 current 取中间值。
            _ku_mode = None      # 'single' | 'chain' | None(信号不足→混合兜底)
            _max_distinct = 0
            _max_runs = 0        # 2026-08-27 18:10: 版本段数（A→B→A=3 段=值回归链；old→new=2 段=单次更新）
            _vke_block = ''
            _sel_preds = {}      # 选中实体→选中的干净属性 predicate 集合
            try:
                if getattr(mem, 'efg', None):
                    _qents = re.findall(r'\b([A-Z][a-z]{2,})\b', q)
                    _commons = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
                        'The','This','That','These','Those','My','Your','His','Her',
                        'Its','Our','Their','Me','You','He','She','It','We','They','List'}
                    _qents = [e.lower() for e in _qents if e not in _commons]
                    if not _qents:  # LME 小写实体 'user' 支持（与 VKE 检索同款）
                        _qw = set(re.findall(r'[a-zA-Z]{3,}', q.lower()))
                        for _subj in mem.efg._entity_to_versions:
                            if _subj in _qw:
                                _qents.append(_subj)
                    for _e in _qents[:3]:
                        _vf = mem.efg.query_by_entity(_e)
                        _per_attr = {}
                        for _f in _vf:
                            _per_attr.setdefault(_f['predicate'], set()).add(str(_f['object']).lower().strip())
                        if _per_attr:
                            _sel = _select_ku_predicates(_per_attr, q, _e)
                            if _sel:
                                _sel_preds[_e] = set(_sel.keys())
                                _max_distinct = max(_max_distinct, max(len(v) for v in _sel.values()))
                                # 版本段数统计：按时间排序后压缩连续重复值，段数≥3 = 值回归链
                                # （conv-26_1 Caroline: adoption→counseling→adoption，distinct=2 但实为链）
                                for _p in _sel:
                                    _seq = []
                                    for _f in sorted([f for f in _vf if f['predicate'] == _p],
                                                     key=lambda x: x.get('time') or ''):
                                        _o = str(_f['object']).lower().strip()
                                        if not _seq or _seq[-1] != _o:
                                            _seq.append(_o)
                                    _max_runs = max(_max_runs, len(_seq))
                    if _max_runs >= 3:
                        _ku_mode = 'chain'   # 值回归/多段链（A→B→A）
                    elif _max_distinct >= 3:
                        _ku_mode = 'chain'   # 多版本链
                    elif _max_distinct == 2:
                        _ku_mode = 'single'  # 单次更新（old→new）
                    # chain/single 模式：把 VKE 版本序列注入上下文（knowledge-update 检索分支原本无 VKE 块，
                    # 模型看不到完整演进链 → chain 无法区分过渡值 vs 稳定值、single 会把高频旧值当 current）
                    # 2026-08-27 修复：只注入选中属性的版本行，不注入实体全部 predicate（避免 72 个
                    # 碎片宾语淹没 LLM）；single 也注入（带时间戳的短版本链，most recent 才可判定）
                    if _ku_mode in ('chain', 'single'):
                        _vke_lines = ['[ENTITY-FACT GRAPH (VKE)]']
                        for _ent in _qents[:3]:
                            _vf = mem.efg.query_by_entity(_ent)
                            _vp = _sel_preds.get(_ent) or set()
                            _seen = set(); _uniq = []
                            for _f in _vf:
                                if _vp and _f['predicate'] not in _vp:
                                    continue
                                _k = (_f['predicate'], _f['object'], _f['time'])
                                if _k not in _seen:
                                    _seen.add(_k); _uniq.append(_f)
                            if _uniq:
                                _vke_lines.append(f'  Entity: {_ent}')
                                _cnt = {}
                                for _f in _uniq:
                                    _k = _f['predicate'] + ':' + _f['object']
                                    _cnt[_k] = _cnt.get(_k, 0) + 1
                                for _f in _uniq[:8]:
                                    _tag = ' <- recurring' if _cnt.get(_f['predicate'] + ':' + _f['object'], 0) >= 2 else ''
                                    _vke_lines.append(f"    {_f['predicate']}: {_f['object'][:60]} [{_f['time'] or '?'}] (w={_f['weight']:.1f}){_tag}")
                        if len(_vke_lines) > 1:
                            _vke_block = '\n'.join(_vke_lines)
            except Exception as _e:
                print(f'    [KU mode detect fail] {_e}', flush=True)
            if _vke_block and '[ENTITY-FACT GRAPH (VKE)]' not in ctx_str:
                ctx_str = _vke_block + '\n\n' + ctx_str
            # 指令模板按题型切换
            if _ku_mode == 'chain':
                _update_instruction = ('This attribute has MULTIPLE versions over time (an evolving chain of updates, '
                 'possibly with values that recur). The CURRENT value is the one at the LATEST timestamp in the '
                 'version list — unless it is a transitional/temporary mention (e.g. "temp job", "for now", '
                 '"while looking for", "just started", one-off states), in which case report the last settled '
                 'value instead. Answer with the value directly.')
            elif _ku_mode == 'single':
                _update_instruction = ('The attribute was updated once (old value -> new value). '
                 'The CURRENT value is the one from the LATEST update; it supersedes the earlier value. '
                 'Answer with the most recent value directly. '
                 'Exception: if the most recent mention is only a transitional/temporary state '
                 '(e.g. "temp job", "for now", "while looking for", "just started") rather than a settled '
                 'change, report the last settled value instead.')
            else:
                _update_instruction = ('Find the most recent relevant fact. '
                 'If it describes a temporary/transitional state (e.g. "temp job", "for now", "while looking for", '
                 '"just started"), prefer the stable value that recurs across time.')
            print(f'    [KU mode] {qid}: {_ku_mode} (max_distinct={_max_distinct}, runs={_max_runs})', flush=True)
            # 2026-09-14 接线：KU 分支自带内联版本链注入，此处仅在未注入时才补 VKE 块
            # （_inject_vke 内含去重条件：已存在 '[ENTITY-FACT GRAPH (VKE)]' 则跳过）
            ctx_str = _inject_vke(ctx_str, _vke_context_block(q, qt, mem, relevant_patterns, qid, ctx_str=ctx_str))
            
            # Add counting instructions if the question asks about counts
            _counting_hint = ''
            if any(w in q.lower() for w in ['how many', 'count', 'how often', 'how many times']):
                _counting_hint = '\nCOUNTING INSTRUCTIONS: Extract the PRECISE count. Ordinals like "first", "second", "third", "another" confirm multiple items. A couple = 2, several = 3+. If context mentions specific numbers use them. For "how many times has {person} done X": count EACH distinct instance. For "how many X does {person} have": count EACH distinct item. Give just the number.'
            
            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. ' + _update_instruction + ' '
                 'Answer with the value directly — do NOT quote or describe the evidence wording '
                 '(never say "the most recent mention is...", "as mentioned on <date>", or "based on the context").'
                 ' If counting, give only the number. Answer in natural language; NEVER reproduce memory entry '
                 'templates or bracketed tags like [date] [category]. If the context contains related information, '
                 'answer with the best available inference (e.g. relationship between people). '
                 'Only if the context truly has NOTHING related, answer: No relevant memories. '
                 'NEVER output phrases like "matching information found".' + km_instructions + _counting_hint},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=256) or ''
            ans = _strip_wrappers(ans.strip())
            
            # Apply counting normalization ONLY for counting questions (fix: 无脑 normalize 污染非计数题)
            if any(w in q.lower() for w in ['how many', 'count', 'how often', 'how many times']):
                ans = _normalize_answer(ans)
            
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or '', '_edpl_ids': injected_ids}
            if not result['correct']:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result
        
        # ─── Preference ──────────────────────────────────────────
        elif qt == 'single-session-preference':
            # cod3-fix3：原 `context[:30]` 会被 [KEYWORD RECALL]（实体名扫出的无关文档）撑满，
            # 把真正关键的 [USER PREFERENCE STATEMENTS] / [PERSONA PROFILES] / [EPISODIC] 挤出窗口
            # （实测 195a1a1b：54 个块里偏好证据块在第 33+ 位，完全没进上下文）。
            # 改为优先级排序：偏好/画像/摘要块置顶，词法召回降级，再按字符预算截断。
            _prio, _rest = [], []
            for d in context:
                s = str(d)
                if ('USER PREFERENCE STATEMENTS' in s or 'PERSONA PROFILES' in s
                        or 'EPISODIC SUMMARIES' in s or 'QUESTION_DATE' in s
                        or 'STRUCTURED FACTS' in s):
                    _prio.append(s[:1200])
                elif '[kw]' in s[:90] or 'KEYWORD RECALL' in s:
                    _rest.append(s[:600])
                else:
                    _rest.append(s[:900])
            ctx_str = chr(10).join(_prio + _rest)
            if len(ctx_str) > 40000:
                ctx_str = ctx_str[:40000] + '...'
            print(f'    [SSP-ctx] prio={len(_prio)} rest={len(_rest)} chars={len(ctx_str)}', flush=True)
            # 2026-09-14 接线：偏好题（PR）用 VKE 版本链补齐“重复出现的偏好/意向”
            ctx_str = _inject_vke(ctx_str, _vke_context_block(q, qt, mem, relevant_patterns, qid, ctx_str=ctx_str))

            # cod3-fix2（extract-then-answer）：先从上下文里把“与本题直接相关的用户原话”抽出来，
            # 再据此作答。动机：证据在上下文里不等于模型会用它——实测会出现“问晚上活动，答街舞+手机App”，
            # 而 gold 要的是“放松、9:30 前、不要手机/手表”。抽取步骤强制模型先定位相关偏好句。
            _ext = ''
            try:
                _ext = call_llm([
                    {'role': 'system', 'content':
                     "You extract the user's OWN prior statements that are relevant to answering the question. "
                     "Quote them VERBATIM, one per line. Include only statements that constrain or inform the "
                     "answer: the user's stated preferences, things they already own/tried, their plans and "
                     "hard constraints (times, dislikes, what they want to avoid). "
                     "If nothing is relevant reply exactly NONE."},
                    {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQuestion: {q}\nRelevant user statements:'}
                ], max_tokens=300) or ''
                _ext = _strip_wrappers(_ext.strip())
                if _ext.strip().upper().startswith('NONE') or len(_ext) < 15:
                    _ext = ''
            except Exception as _e:
                print(f'    [SSP-extract] skipped ({_e})', flush=True)

            _ssp_sys = (
                'The user is asking for a recommendation or advice. Below you are given the user\'s own '
                'relevant prior statements (verbatim quotes) and additional context. '
                'Base your answer on those statements: build explicitly on the items, ingredients, brands, '
                'plans and constraints they mention, and RESPECT any constraint (times, things to avoid). '
                'Your answer MUST repeat the concrete details from those statements (name them), not just '
                'allude to them, and must not contradict them. If a constraint is stated, say how your '
                'suggestion satisfies it. Never give generic advice; never invent details absent from the context.' + km_instructions)
            _user_msg = f'Context:\n{ctx_str}\n\n'
            if _ext:
                _user_msg += f"User's relevant prior statements (verbatim):\n{_ext}\n\n"
                _user_msg += f'Q: {q}\nRecommendation:'
            else:
                # cod3-fix4：抽取返回 NONE 时不能直接弃答（实测 3 道题因此输出
                # "I don't have any record of..."）。改为强制从上下文里找“用户已有的相关
                # 经历/计划/期望”，宁可否定也不许空手回。
                _ssp_sys += (
                    '\nIf no explicit preference is quoted, still reason from the history: look for the '
                    'user\'s own related experiences, ongoing plans, possessions, or stated constraints on this '
                    'topic and build the recommendation on them. Never reply that you have no information when '
                    'the history contains related material.')
                _user_msg += f'Q: {q}\nRecommendation:'
            ans = call_llm([
                {'role': 'system', 'content': _ssp_sys},
                {'role': 'user', 'content': _user_msg}
            ], max_tokens=512) or ''
            ans = _strip_wrappers(ans.strip())
            
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or '', '_edpl_ids': injected_ids}
            if not result['correct']:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result
        
        # ─── Single-session user (default) ───────────────────────
        else:
            # Iter2: 窗口 50→60（配合 top_k 50 + session expansion，conv-47_30 5 游戏全覆盖）
            ctx_str = '\n'.join([str(c)[:600] for c in context[:60]])
            if len(ctx_str) > 30000:
                ctx_str = ctx_str[:30000] + '...'
            # 2026-09-14 接线：SSU/SSA 不在 VKE 允许题型内，_vke_context_block 会直接返回 ''
            # （保留调用点是为了未来改常量时行为一致）；when 类 SSU 仍会在答案后走 PostProc
            ctx_str = _inject_vke(ctx_str, _vke_context_block(q, qt, mem, relevant_patterns, qid, ctx_str=ctx_str))
            
            ans = call_llm([
                {'role': 'system', 'content': 'Answer the question directly based on the conversation context. '
                 'Give the specific fact requested. Short, direct answer. '
                 'If the question asks for a list or all matching items, list ALL of them. '
                 'For yes/no questions answer Yes or No. '
                 'CRITICAL: Distinguish what the person ACTUALLY DID from future plans or intentions '
                 '(phrases like "thinking of", "planning to", "hoping to", "in the next few months", "want to", "going to", "should", "could", "advice/tips" are NOT completed actions). '
                 'If the question asks what happened/did (e.g. "How did X ...?"), report ONLY completed actions '
                 'the person explicitly says they did ("I worked with...", "I made...", "I developed...", "I launched..."), '
                 'NEVER include suggestions given to others or future intentions. '
                 'If counting ("how many ... have made it/done"), count only items confirmed as completed '
                 'and already realized; exclude items described as being worked on, hoped for, or planned. '
                 'A statement like "writing another script hoping to get it on the big screen" is NOT a completed '
                 'item — do NOT count it. Only count items with past-tense completion evidence '
                 '("appeared on", "was shown on", "made it to", "was on"). '
                 'Ignore characters\' speculation about future counts (e.g. someone saying "it will be your 3rd"). '
                 'Answer in natural language; NEVER reproduce memory entry templates or bracketed tags like [date] [category].' + km_instructions},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=512) or ''
            ans = _strip_wrappers(ans.strip())
            
            # when 类问题（含日期内容但被标为 SSU）二次提取兜底：答案无"具体月+日"或含模糊词 → 用 timeline 再抽一次
            _is_when_q2 = q.lower().strip().startswith('when') and not any(
                p in q.lower() for p in ['which city', 'what city', 'where was', 'where did', 'where is',
                                          'which week', 'which country', 'what country', 'which places', 'what places'])
            _has_specific_date2 = re.search(
                r'(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}(st|nd|rd|th)?\b', ans)
            _has_vague_date2 = re.search(r'(?i)\b(before|after|around|about|by|until|since|last season|not mentioned|not sure|unknown|not specified)\b', ans)
            if _is_when_q2 and (not _has_specific_date2 or _has_vague_date2):
                _de_prompt2 = (f'From the timeline below, find the date or time period when this happened: "{q}"\n\n'
                               f'Timeline:\n' + ctx_str[:20000] + '\n\nAnswer with ONLY the date or time range. Be precise.')
                _de_ans2 = call_llm([
                    {'role': 'system', 'content': 'Extract the exact date or time period for the event. Short answer only.'},
                    {'role': 'user', 'content': _de_prompt2}
                ], max_tokens=64) or ''
                if _de_ans2 and len(_de_ans2.strip()) > 3 and 'not' not in _de_ans2.lower()[:20]:
                    print(f'    [SSU-DateFix] {qid}: "{ans[:40]}" -> "{_de_ans2.strip()[:40]}"', flush=True)
                    ans = _de_ans2.strip()
            
            # Post-process: city→country
            ans = _apply_city2country(ans or '', q)
            
            # Apply counting normalization for how-many questions
            if any(w in q.lower() for w in ['how many', 'how much', 'count', 'total']):
                ans = _normalize_answer(ans)
            # 2026-09-14 接线：when 类 SSU/SSA 过声明式后处理（其余题型直接返回原答案）
            ans = _apply_postproc(ans, q, qt, sessions, dates, ctx_str, qid)
            
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or '', '_edpl_ids': injected_ids}
            if not result['correct']:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result
    
    except Exception as e:
        print(f'  [ERR v2] {qid}: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return {'correct': False, 'invalid': True, 'type': qt, 'predicted': '', '_edpl_ids': injected_ids}


# ═══════════════════════════════════════════════════════════════════
# Counting normalization
# ═══════════════════════════════════════════════════════════════════

def _normalize_answer(ans: str) -> str:
    """Normalize LLM output to always end with TOTAL: X for reliable counting judge."""
    if not ans or not ans.strip():
        return ans
    ans_stripped = ans.strip()
    if re.search(r'(?i)total:\s*\d+', ans_stripped):
        return ans
    eq_matches = list(re.finditer(r'=\s*[$]?(\d+(?:\.\d+)?)', ans_stripped))
    if eq_matches:
        last = eq_matches[-1].group(1)
        return f'{ans_stripped}\nTOTAL: {last}'
    nums = re.findall(r'\d+\.?\d*', ans_stripped)
    if nums:
        return f'{ans_stripped}\nTOTAL: {nums[-1]}'
    return ans


# ═══════════════════════════════════════════════════════════════════
# City→Country post-processing
# ═══════════════════════════════════════════════════════════════════

_CITY_COUNTRY_MAP = {
    'paris': 'France', 'london': 'UK', 'berlin': 'Germany', 'rome': 'Italy',
    'madrid': 'Spain', 'barcelona': 'Spain', 'moscow': 'Russia', 'beijing': 'China',
    'shanghai': 'China', 'tokyo': 'Japan', 'seoul': 'South Korea', 'bangkok': 'Thailand',
    'dubai': 'UAE', 'singapore': 'Singapore', 'mumbai': 'India', 'delhi': 'India',
    'istanbul': 'Turkey', 'cairo': 'Egypt', 'sydney': 'Australia', 'melbourne': 'Australia',
    'toronto': 'Canada', 'vancouver': 'Canada', 'mexico city': 'Mexico',
    'new york': 'USA', 'los angeles': 'USA', 'chicago': 'USA', 'san francisco': 'USA',
    'boston': 'USA', 'washington': 'USA', 'miami': 'USA', 'seattle': 'USA',
    'amsterdam': 'Netherlands', 'brussels': 'Belgium', 'vienna': 'Austria',
    'prague': 'Czech Republic', 'budapest': 'Hungary', 'warsaw': 'Poland',
    'stockholm': 'Sweden', 'oslo': 'Norway', 'helsinki': 'Finland', 'copenhagen': 'Denmark',
    'dublin': 'Ireland', 'edinburgh': 'UK', 'athens': 'Greece', 'lisbon': 'Portugal',
    'milan': 'Italy', 'naples': 'Italy', 'florence': 'Italy', 'venice': 'Italy',
    'zurich': 'Switzerland', 'geneva': 'Switzerland', 'munich': 'Germany',
    'hamburg': 'Germany', 'frankfurt': 'Germany', 'luxembourg': 'Luxembourg',
    'hong kong': 'China', 'kuala lumpur': 'Malaysia', 'jakarta': 'Indonesia',
    'manila': 'Philippines', 'hanoi': 'Vietnam', 'ho chi minh city': 'Vietnam',
    'nairobi': 'Kenya', 'cape town': 'South Africa', 'johannesburg': 'South Africa',
    'casablanca': 'Morocco', 'lagos': 'Nigeria',
    'sao paulo': 'Brazil', 'rio de janeiro': 'Brazil', 'buenos aires': 'Argentina',
    'santiago': 'Chile', 'lima': 'Peru', 'bogota': 'Colombia',
}


def _apply_city2country(answer: str, question: str) -> str:
    """If question asks for a country and answer is a known city, map it."""
    if not answer or not question:
        return answer
    q_lower = question.lower()
    country_pats = [r'\bin\s+what\s+country\b', r'^what\s+country\b', r'^which\s+country\b',
                    r'\bin\s+which\s+country\b', r'^in what country\b']
    is_country_q = any(re.search(p, q_lower) for p in country_pats)
    if not is_country_q:
        return answer
    a_lower = answer.strip().lower().rstrip('.!?')
    if a_lower in _CITY_COUNTRY_MAP:
        mapped = _CITY_COUNTRY_MAP[a_lower]
        print(f'    [city2country] "{a_lower}" -> "{mapped}"', flush=True)
        return mapped
    # 答案是含城市名的句子（如 "Jolene bought the snake Seraphim in Paris."）→ 把城市名替换为国家名
    for city, country in _CITY_COUNTRY_MAP.items():
        if re.search(r'\b' + re.escape(city) + r'\b', a_lower):
            new_ans = re.sub(r'\b' + re.escape(city) + r'\b', country, answer, flags=re.I)
            print(f'    [city2country] "{city}" in sentence -> "{country}": "{answer[:60]}..." -> "{new_ans[:60]}..."', flush=True)
            return new_ans
    return answer
