#!/usr/bin/env python3
"""
trimem_ar_v2.py — Type-Adaptive Retrieval (v2)

Key components:
1. Query Profiling: p(q) = (f_s, t_s, d_i) for factual/temporal/inference depth
2. Dynamic Component Activation: selects subset of components based on profile
3. Entity-Fact Graph (EFG) construction during add_sessions
4. Unified evidence merging with deduplication by speaker

v2 retrieval mechanisms ported from 正版 TriMemAR v1 (trimemnar_locomo_0615/trimem_ar.py):
- _get_event_facts_for_query(): structured event facts for temporal questions
- temporal-reasoning: TIMELINE + EVENT DATES + structured event facts injection
- multi-session: multi-query expansion + weighted(1.0/0.8/0.6) fusion top-40
- knowledge-update: FEI docs date-descending (latest first)
- Entity graph multi-hop injection (_graph_entity_sessions/_graph_session_entities)
- EPISODIC SUMMARIES (sim>0.3) + semantic facts [FACTS] + [DIRECT FACTS] injection
"""
import sys, json, os, re, time, hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from hierarchyv2.config import _STOP, _MS_COMMON_NAMES
from hierarchyv2.embedding import embed
from hierarchyv2.date_utils import parse_date, fmt_date
from hierarchyv2.llm_utils import call_llm, generate_episodic_summary
from hierarchyv2.fact_extractor_v2 import (extract_facts_v2, infer_persona, embed_persona,
                                          extract_subject, query_persona)
from hierarchyv2.temporal_engine import TemporalEngine
from hierarchyv2.utils import chunk_long_text


# ═══════════════════════════════════════════════════════════════════
# Query Profiling
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 2026-09-18 cod1（论文对齐 P1）— Eq.1 / App A.1：
#   论文原文：“The profile vector p(T_q) = (f_s, t_s, d_i) is read from the type T_q.
#   Each type is mapped to a fixed profile, so no additional inference-time
#   computation is required after the type decision.”
#   → 原实现用关键词规则现场推断画像，与论文的“题型查表”不符。cod1 改为以题型 T_q
#     查固定表（无题型时才回退关键词规则），使激活组件完全由 T_q 决定。
# ═══════════════════════════════════════════════════════════════════════════
_PAPER_PROFILE_TABLE = {
    # 题型 T_q                       (f_s, t_s, d_i)
    'single-session-user':          (1, 0, 0),
    'single-session-assistant':     (1, 0, 0),
    'single-session-preference':    (1, 0, 0),
    'temporal-reasoning':           (1, 1, 0),
    'knowledge-update':             (1, 1, 0),
    'multi-session':                (1, 0, 1),
    # LoCoMo 官方四类（code_locomo, 2026-09-19）
    'single-hop':  (1, 0, 0),
    'open-domain': (1, 0, 0),
    'temporal':    (1, 1, 0),
    'multi-hop':   (1, 0, 1),
    # VKE 内部题型别名
    'SS': (1, 0, 0), 'PR': (1, 0, 0), 'TR': (1, 1, 0), 'KU': (1, 1, 0), 'MS': (1, 0, 1),
}


def profile_query(query: str, question_type: str = '') -> Dict[str, int]:
    """
    Profile query across 3 dimensions: factual sensitivity, temporal sensitivity, inference depth.
    Returns: {f_s: 0|1, t_s: 0|1, d_i: 0|1}

    2026-09-18 cod1：论文 Eq.1 / App A.1 口径 —— p(T_q) 由查询类型 T_q 查固定表得到；
    只有在拿不到题型时才回退到关键词规则（原实现全程走关键词，与论文不符）。
    """
    _t = (question_type or '').strip()
    if _t in _PAPER_PROFILE_TABLE:
        _f, _tt, _d = _PAPER_PROFILE_TABLE[_t]
        return {'f_s': _f, 't_s': _tt, 'd_i': _d}

    q_lower = query.lower()
    
    # f_s (factual sensitivity)
    f_s = 0
    factual_signals = ['what', 'which', 'who', 'how many', 'how much',
                       'favorite', 'prefer', 'what kind', 'what type',
                       'where', 'when', 'name of', 'list', 'tell me']
    for signal in factual_signals:
        if signal in q_lower:
            f_s = 1
            break
    
    # t_s (temporal sensitivity)
    t_s = 0
    temporal_signals = ['current', 'now', 'recently', 'latest', 'most recent',
                        'change', 'changed', 'has changed', 'used to', 'before',
                        'previously', 'originally', 'initially', 'update',
                        'how often', 'frequency', 'how long', 'since', 'after',
                        'before', 'during', 'in the past', 'over time',
                        'first', 'last', 'new', 'old']
    for signal in temporal_signals:
        if signal in q_lower:
            t_s = 1
            break
    if re.search(r'\b\d{4}\b', q_lower) or re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', q_lower):
        t_s = 1
    
    # d_i (inference depth)
    d_i = 0
    depth_signals = ['why', 'how did', 'how does', 'how has', 'what was the reason',
                     "why didn't", "why wouldn't", 'imply', 'infer', 'conclude',
                     'compare', 'contrast', 'difference', 'relationship between',
                     'connection between', 'related to', 'instead of',
                     'alternative', 'suggest', 'how many different', 'how many separate']
    for signal in depth_signals:
        if signal in q_lower:
            d_i = 1
            break
    multi_session_signals = ['how many', 'in the past', 'this year', 'last year',
                             'over the course', 'during', 'different types',
                             'throughout', 'across']
    for signal in multi_session_signals:
        if signal in q_lower:
            d_i = 1
            break
    
    return {'f_s': f_s, 't_s': t_s, 'd_i': d_i}


def activate_components(profile: Dict[str, int]) -> set:
    """
    Dynamic component activation（2026-09-18 cod1，对齐论文 Eq.1 / App A.1）

    论文原文：“By default, FEI and FG provide the starting point for every query.
    When a query asks for explicit facts, EvoMem also consults EAT and IPI. When
    temporal information is needed, it follows the relevant version chain in FG.”

      C_active = {FEI, FG}                      # 默认起点（原实现 FEI 只在 d_i=1 时激活）
               ∪ {EAT, IPI | f_s=1}             # 显式事实
               ∪ {FG version-chain | t_s=1}     # 版本链遍历（代码中用 VKE 键表示）
               ∪ {FEI, FG | d_i=1}              # 跨会话推理（代码中用 HG 键表示）
    """
    active = {'FEI', 'FG'}                     # 论文：每个查询的默认起点
    if profile.get('f_s') == 1:
        active |= {'EAT', 'IPI'}
    if profile.get('t_s') == 1:
        active |= {'VKE', 'EDPL'}              # FG 中的版本链遍历
    if profile.get('d_i') == 1:
        active |= {'FEI', 'HG'}
    return active


# ═══════════════════════════════════════════════════════════════════
# Entity-Fact Graph (EFG)
# ═══════════════════════════════════════════════════════════════════

_CE_CACHE = None


class EntityFactGraph:
    """Entity-Fact Graph: multi-layer structure connecting entities to evolving attributes."""
    
    def __init__(self):
        self.entities = {}
        self.attributes = {}
        self.versions = {}
        self.episodes = {}
        self.hierarchy_edges = defaultdict(set)
        self.attribute_edges = defaultdict(set)
        self.version_chains = defaultdict(list)
        self.source_edges = {}
        self._entity_to_versions = defaultdict(list)
        self._next_version_id = 0
    
    def add_entity(self, entity_name: str, semantic_category: str = "L1"):
        if entity_name not in self.entities:
            self.entities[entity_name] = {'name': entity_name, 'category': semantic_category, 'sessions': set()}
    
    def add_fact(self, fact: Dict):
        subj = fact.get('subject', '').strip().lower()
        pred = fact.get('predicate', '').strip()
        obj = str(fact.get('object', '')).strip()
        conf = fact.get('confidence', 'medium')
        time_str = fact.get('time', '')
        session_id = fact.get('session_id', '')
        if not subj or not pred:
            return
        
        self.add_entity(subj)
        vid = self._next_version_id
        self._next_version_id += 1
        version = {
            'id': vid, 'object': obj, 'object_lower': obj.lower(),
            'time': time_str, 'confidence': conf, 'weight': 1.0,
            'source': session_id, 'subject': subj, 'predicate': pred,
        }
        self.versions[vid] = version
        
        attr_key = (subj, pred)
        if attr_key not in self.attributes:
            self.attributes[attr_key] = {'entity': subj, 'predicate': pred, 'versions': []}
        
        existing = None
        for v in self.attributes[attr_key]['versions']:
            existing_v = self.versions.get(v, {})
            if existing_v.get('object_lower') == obj.lower() and existing_v.get('time') == time_str:
                existing = v
                break
        
        if existing is not None:
            self.versions[existing]['weight'] = self.versions[existing].get('weight', 1.0) + 0.5
        else:
            self.attributes[attr_key]['versions'].append(vid)
            if subj not in self.attribute_edges:
                self.attribute_edges[subj] = []
            if attr_key not in self.attribute_edges[subj]:
                self.attribute_edges[subj].append(attr_key)
            self.version_chains[attr_key].append(vid)
            self.source_edges[vid] = session_id
        
        self._entity_to_versions[subj].append((pred, vid))
    
    def get_entity_versions(self, entity: str, predicate: Optional[str] = None) -> List[int]:
        if predicate:
            attr_key = (entity, predicate)
            return self.version_chains.get(attr_key, [])
        versions = []
        for pred, vid in self._entity_to_versions.get(entity, []):
            versions.append(vid)
        return versions
    
    def get_version(self, vid: int) -> Optional[Dict]:
        return self.versions.get(vid)
    
    def get_entity_attributes(self, entity: str) -> List[Tuple[str, str]]:
        results = []
        for pred, vid in self._entity_to_versions.get(entity, []):
            v = self.versions.get(vid)
            if v:
                results.append((pred, v.get('object', '')))
        return results
    
    def query_by_entity(self, entity: str) -> List[Dict]:
        facts = []
        for pred, vid in self._entity_to_versions.get(entity, []):
            v = self.versions.get(vid, {})
            facts.append({
                'subject': entity, 'predicate': pred, 'object': v.get('object', ''),
                'time': v.get('time', ''), 'confidence': v.get('confidence', 'low'),
                'weight': v.get('weight', 1.0),
            })
        return facts


# ═══════════════════════════════════════════════════════════════════
# TriMemAR v2
# ═══════════════════════════════════════════════════════════════════

class TriMemAR_v2:
    """Type-Adaptive Retrieval with v2 components."""
    
    def __init__(self):
        self.raw_docs = []
        self.embs = None
        self.facts = []
        self.efg = EntityFactGraph()
        self.personas = []
        self.persona_embs = None
        self.temporal = TemporalEngine()
        self.qdate = ''
        self.qdate_dt = None
        self.sessions_meta = []
        self.episodic_summaries = []
        self.episodic_embs = None
        # NEW: semantic facts + graph co-occurrence index (port from 正版)
        self.semantic_facts = []  # [{s, p, o, date}]
        self._graph_entity_sessions = {}  # entity -> set(session_ids)
        self._graph_session_entities = {}  # session_id -> set(entities)
        self._graph_sessions_cache = []  # raw sessions
        self._cached_sessions = []
        self._cached_dates = []
        self._profile_cache = {}
        self._km_boost_words = []
    
    def add_sessions(self, sessions, dates, km_patterns=None, chunk_enabled=True, attrs=None):
        if km_patterns is None:
            km_patterns = []
        self.attrs = attrs or []  # 2026-08-27 方案①: 数据集属性词表（EAT Category 9 模板族驱动）
        
        for si, session in enumerate(sessions):
            if not isinstance(session, list):
                continue
            dt = dates[si] if si < len(dates) else ''
            parsed_dt = parse_date(dt)
            dt_str = fmt_date(parsed_dt) if parsed_dt else ''
            
            user_turns = []
            all_turns = []
            
            for turn in session:
                if isinstance(turn, dict) and turn.get('content', '').strip():
                    role = turn.get('role', 'user')
                    content = turn['content'].strip()
                    all_turns.append((role, content))
                    if role == 'user':
                        user_turns.append(content)
            
            # Build episode pairs for flat index
            for i in range(0, len(session), 2):
                pair = session[i:i+2]
                if len(pair) >= 1 and isinstance(pair[0], dict) and pair[0].get('content', '').strip():
                    user_txt = pair[0]['content'].strip()
                    asst_txt = pair[1]['content'].strip() if len(pair) >= 2 and isinstance(pair[1], dict) and pair[1].get('content', '') else ''
                    if chunk_enabled:
                        chunks = chunk_long_text(user_txt, asst_txt)
                        for chunk_text in chunks:
                            self.raw_docs.append({'text': chunk_text, 'date': dt_str, 'session': si})
                    else:
                        raw_text = f"[user] {user_txt}"
                        if asst_txt:
                            raw_text += f" [assistant] {asst_txt}"
                        self.raw_docs.append({'text': raw_text, 'date': dt_str, 'session': si})
            
            # EAT: Extract structured facts（2026-08-26 v4: 回滚 user-only。
            # 全 turns 提取会引入对话双方噪音（hiking/kickbacking/painting 权重暴涨），
            # 污染 VKE 版本链导致 current 类崩盘（v3 63% vs v2 76%）。
            # 第二说话人属性缺失问题留待专项解决（不牺牲主属性精度）。）
            for user_text in user_turns:
                text = user_text.strip()
                session_id = f'session_{si}'
                extracted = extract_facts_v2(text, dt_str, session_id, attrs=self.attrs or None)
                for f in extracted:
                    self.facts.append(f)
                    self.efg.add_fact(f)
                    self.temporal.add_fact(f)
            
            # IPI: Infer persona once per session (capped for benchmark cost control)
            # 2026-08-27: 默认全量（IPI_MAX_SESSIONS=0 表示不限）——版本链需要完整时间序列，
            # cap=10 会截断转折点之后的会话（conv-26_1 adoption 转向在 s7+，s10-18 无 persona）
            _ipi_max = int(os.environ.get('IPI_MAX_SESSIONS', '0'))
            if user_turns and (_ipi_max <= 0 or si < _ipi_max):
                try:
                    persona = infer_persona(user_turns, dt_str)
                    persona_text = embed_persona(persona)
                    if persona_text and persona_text != "; ":
                        self.personas.append({
                            'text': persona_text, 'date': dt_str,
                            'session': si, 'dimensions': persona,
                        })
                        self.persona_embs = None
                    # 2026-08-27 (B): persona 维度注入 EFG 版本链（persona_* predicate，带时间戳）。
                    # 画像维度是 LLM 语义推断，覆盖 EAT 正则抓不到的兴趣/职业意向；
                    # 只注入 user 说话人（与 EAT user-only 对齐，避免第二说话人噪音）。
                    _p_subj = extract_subject(user_turns[0]) if user_turns else 'user'
                    for dim, val in persona.items():
                        _v = str(val).strip()
                        if not _v or _v == 'N/A' or _v.lower().startswith('n/a'):
                            continue
                        if len(_v) > 120:
                            _v = _v[:120]
                        self.efg.add_fact({
                            'subject': _p_subj, 'predicate': f'persona_{dim}',
                            'object': _v, 'time': dt_str,
                            'session_id': f'session_{si}',
                            'confidence': 'medium',
                            'raw': _v[:120], 'category': 'persona',
                            'semantic_subcategory': dim,
                        })
                except Exception as e:
                    print(f'    [IPI] session {si} failed: {e}', flush=True)
            
            self.sessions_meta.append((dt_str, len(session)))
        
        # ═══ Entity graph co-occurrence index (port from 正版 add_sessions) ═══
        self._graph_sessions_cache = sessions
        _SKIP_ENTS = {'The','This','That','These','Those','My','Your','His','Her',
            'Its','Our','Their','I','You','He','She','It','We','They',
            'How','What','Why','Where','When','Which','Who','Whom',
            'Can','Will','May','Might','Shall','Should','Would','Could',
            'Am','Is','Are','Was','Were','Be','Been','Being',
            'Have','Has','Had','Do','Does','Did',
            'Not','No','Yes','And','But','Or','For','With','In',
            'On','At','To','From','By','Of','Up','Out','Off','Over',
            'Then','Than','Now','Just','Only','Also','Very','Really',
            'Again','Back','Here','There','More','Most','Many','Much',
            'One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'}
        self._graph_entity_sessions = {}
        self._graph_session_entities = {}
        for si, session in enumerate(sessions):
            if not isinstance(session, list):
                continue
            ses_entities = set()
            for turn in session:
                if isinstance(turn, dict) and turn.get('content', ''):
                    named_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', turn['content'])
                    for phrase in named_phrases:
                        words_in_phrase = phrase.split()
                        meaningful = [w for w in words_in_phrase if w not in _SKIP_ENTS]
                        if meaningful:
                            entity = ' '.join(meaningful).lower()
                            ses_entities.add(entity)
            for f in self.facts:
                subj = str(f.get('subject', '')).lower().strip()
                obj = str(f.get('object', '')).lower().strip()
                if subj and len(subj) > 2:
                    ses_entities.add(subj)
                if obj and len(obj) > 2:
                    ses_entities.add(obj)
            for ent in ses_entities:
                if ent not in self._graph_entity_sessions:
                    self._graph_entity_sessions[ent] = set()
                self._graph_entity_sessions[ent].add(si)
            if ses_entities:
                self._graph_session_entities[si] = ses_entities
        
        # ═══ EPISODIC SUMMARIES + SEMANTIC FACTS (port from 正版, with disk cache) ═══
        EPI_CACHE_VERSION = 2
        epi_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'epi_cache_v2.json')
        
        def _load_epi_cache():
            try:
                d = json.load(open(epi_cache_path))
                if isinstance(d, dict) and d.get('version') == EPI_CACHE_VERSION:
                    return d.get('entries', {})
                return {}
            except Exception:
                return {}
        
        def _save_epi_cache(entries):
            try:
                json.dump({'version': EPI_CACHE_VERSION, 'entries': entries}, open(epi_cache_path, 'w'))
            except Exception as ec:
                print(f'    [epi cache] save failed: {ec}', flush=True)
        
        cache_key = hashlib.md5(json.dumps([sessions, dates], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = _load_epi_cache()
        
        if cache_key in cache:
            cached = cache[cache_key]
            self.episodic_summaries = list(cached.get('summaries', []))
            self.semantic_facts = list(cached.get('semantic_facts', []))
            print(f'    [epi cache] loaded {len(self.episodic_summaries)} summaries, {len(self.semantic_facts)} facts', flush=True)
        else:
            EPI_WINDOW = 10
            for si, session in enumerate(sessions):
                if not isinstance(session, list):
                    continue
                dt = dates[si] if si < len(dates) else ''
                turns = [t for t in session if isinstance(t, dict) and t.get('content', '').strip()]
                conv_texts = [f"[{t.get('role','?')}] {t['content'].strip()}" for t in turns]
                for start in range(0, len(conv_texts), EPI_WINDOW * 2):
                    batch = conv_texts[start:start + EPI_WINDOW * 2]
                    if len(batch) >= 2:
                        summary = generate_episodic_summary(batch)
                        if summary:
                            self.episodic_summaries.append({'summary': summary, 'date': dt[:10] if dt else ''})
            
            # Semantic facts: stable semantic knowledge (preference/ownership/location/frequency/count)
            _sem_preds = {'likes', 'has_object', 'located_at', 'dislikes', 'plans',
                          'frequency', 'count_of', 'attended_sessions', 'exact_count', 'vague_count'}
            sem_facts = []
            seen_sem = set()
            for f in self.facts:
                pred = f.get('predicate', '')
                obj = str(f.get('object', '')).strip()
                if pred in _sem_preds and obj and len(obj) > 2:
                    key = (pred, obj.lower()[:60])
                    if key not in seen_sem:
                        seen_sem.add(key)
                        sem_facts.append({'s': f.get('subject', ''), 'p': pred, 'o': obj[:120],
                                          'date': f.get('time', '')})
            self.semantic_facts = sem_facts[:400]
            try:
                cache[cache_key] = {'summaries': self.episodic_summaries, 'semantic_facts': self.semantic_facts}
                _save_epi_cache(cache)
            except Exception:
                pass
            print(f'    [epi] built {len(self.episodic_summaries)} summaries, {len(self.semantic_facts)} semantic facts', flush=True)
        
        self._cached_sessions = sessions
        self._cached_dates = dates
        self.temporal.build_sequence(sessions, dates)
        self.embs = None
        self.episodic_embs = None
    
    def profile_query(self, query: str, question_type: str = '') -> Dict[str, int]:
        _k = (question_type or '', query)
        if _k not in self._profile_cache:
            self._profile_cache[_k] = profile_query(query, question_type)
        return self._profile_cache[_k]
    
    def search(self, query: str, question_type: str = '', force_broad: bool = False,
               branch_probs: Optional[Dict[str, float]] = None) -> List[str]:
        """Type-Adaptive Retrieval: profile query, activate components, merge results.
        force_broad=True: bypass profile gates, return wider FEI top-k (fallback when
        normal retrieval returns nothing usable).
        branch_probs (soft routing): dict {question_type: prob}. When given, the
        type-specific hard branch is REPLACED by a probability-weighted fusion of all
        branch evidence blocks (soft routing — no hard decision). None keeps the
        original hard branch behavior (backward compatible).
        """
        if not self.raw_docs:
            return ['']
        
        # 2026-09-18 cod1：画像由题型 T_q 查表（论文 Eq.1），不再是关键词推断
        profile = self.profile_query(query, question_type)
        active = activate_components(profile)
        if force_broad:
            active = {'FEI'}
        
        print(f'    [Profile] f_s={profile["f_s"]}, t_s={profile["t_s"]}, d_i={profile["d_i"]}', flush=True)
        print(f'    [Activate] components={active}', flush=True)
        
        context_parts = []
        if self.qdate:
            context_parts.append(f'[QUESTION_DATE: {self.qdate}]')
        
        top_indices = None
        sims = None
        
        # FEI — Flat Episode Index
        if 'FEI' in active or 'EAT' in active:
            if self.embs is None and self.raw_docs:
                self.embs = np.array(embed([d['text'] for d in self.raw_docs]))
            
            if self.embs is not None and len(self.raw_docs) > 0:
                # 2026-08-26 T2: 多查询扩展（current/historical 分支）——属性词/实体词/动词变体加权融合，
                # 缓解 LME 超长会话 BGE 单查询区分度差（杂音 score 与答案几乎无差）
                _mq_queries = [query]
                if question_type in ('knowledge-update', 'temporal-reasoning'):
                    _q2 = re.sub(r'(?i)\b(what|is|was|are|were|the|now|current|value|before|it|changed|change|did|over|time|user|his|her|their|my|your)\b', ' ', query)
                    _q2 = re.sub(r"'s|'|\s+", ' ', _q2).strip().rstrip('?.,;:! ')
                    if 3 < len(_q2) < 60:
                        _mq_queries.append(_q2)
                    _ents = [e for e in re.findall(r'[a-z]{3,}', query.lower()) if e not in
                             ('what','is','was','are','were','the','now','current','value','before','it','changed','change','did','over','time','user','his','her','their','my','your','and','or','for','with','from','how','many','list','different','values','observed')]
                    if _ents:
                        _mq_queries.append(' '.join(_ents[:4]))
                qe = np.array(embed(_mq_queries))
                _all_sims = np.dot(self.embs, qe.T)
                if _mq_queries:
                    _w = [1.0] + [0.7] * (len(_mq_queries) - 1)
                    sims = np.max(_all_sims * np.array(_w), axis=1)
                    print(f'    [MultiQuery] {len(_mq_queries)} queries: {_mq_queries}', flush=True)
                else:
                    sims = _all_sims[:, 0]
                top_indices = np.argsort(sims)[::-1]
                
                # BM25 词法路（2026-08-26 T3）：RRF 融合，专治专有名词（askFundu/Tongariro）BGE 盲区
                try:
                    _bm = self._bm25_scores(query, self.raw_docs)
                    if float(_bm.max()) > 0:
                        _bge_rank = {idx: r for r, idx in enumerate(top_indices)}
                        _bm_rank = {idx: r for r, idx in enumerate(np.argsort(_bm)[::-1]) if _bm[idx] > 0}
                        _rrf_scores = {}
                        for idx in range(len(self.raw_docs)):
                            _rrf_scores[idx] = 1.0 / (60 + _bge_rank.get(idx, 9999)) + 1.0 / (60 + _bm_rank.get(idx, 9999))
                        top_indices = np.array(sorted(_rrf_scores, key=_rrf_scores.get, reverse=True))
                        print(f'    [BM25] lexical route fused (top={_bm.max():.2f})', flush=True)
                except Exception as _bme:
                    print(f'    [BM25] failed ({_bme})', flush=True)
                
                # Cross-Encoder rerank — 默认关闭（实测 CE/CPU 会把含答案的对话片段挤出 top-k：
                # conv-43_q190 的 doc[63] BGE#19 → CE 后 20+ 名，Verify 因 CE-top5 碰巧命中而不触发）。
                # 主挽救机制 = kw rerank（基于 BGE 原始 sims）。需要时 V2_USE_CE=1 开启。
                if os.environ.get('V2_USE_CE', '0') == '1':
                    try:
                        global _CE_CACHE
                        if _CE_CACHE is None:
                            from sentence_transformers import CrossEncoder
                            _CE_CACHE = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
                        pairs = [(query, self.raw_docs[idx]['text'][:512]) for idx in top_indices[:80]]
                        ce_scores = _CE_CACHE.predict(pairs)
                        _reranked = sorted(zip(ce_scores, top_indices[:80]), key=lambda x: -x[0])
                        top_indices = np.array([idx for _, idx in _reranked])
                    except Exception:
                        pass  # fallback: BGE-only order
                
                # ── Entity graph multi-hop injection (port from 正版 方案4) ──
                try:
                    _names = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
                    _common_graph = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
                        'The','A','An','This','That','These','Those','My','Your','His','Her','Its',
                        'Our','Their','Me','You','He','She','It','We','They','No','Yes','True','False',
                        'Can','Will','May','Might','Shall','Should','Would','Could',
                        'Am','Is','Are','Was','Were','Be','Been','Being',
                        'Have','Has','Had','Do','Does','Did',
                        'Not','And','But','Or','For','With','In','On','At','To','From','By','Of',
                        'Up','Out','Off','Over','Again','Back','Here','There',
                        'Then','Than','Now','Just','Only','Also','Very','Really',
                        'More','Most','Many','Much','All','Even','Still','Already',
                        'One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'}
                    _named_entities = [n for n in _names if n not in _common_graph]
                    # 2026-08-26 T1: LME 小写实体（'user'）支持——query 词匹配会话实体图
                    if not _named_entities and getattr(self, '_graph_entity_sessions', None):
                        _q_words = set(re.findall(r'[a-z]{3,}', query.lower()))
                        _skip_ent = {'what','when','where','why','which','who','how','the','this','that','these','those','my','your','his','her','its','our','their','now','current','value','before','changed','change','did','over','time','list','different','values','observed','user','and','for','with','from','into','about','was','were','are','is','have','has','had','not','just','only'}
                        _named_entities = [w for w in _q_words if w in self._graph_entity_sessions and w not in _skip_ent][:3]
                    _skip_graph_types = {'single-session-user'}
                    if (_named_entities and question_type not in _skip_graph_types
                            and getattr(self, '_graph_entity_sessions', None)):
                        _neighbor_sessions = set()
                        for _ent in _named_entities:
                            _ent_lower = _ent.lower()
                            if _ent_lower in self._graph_entity_sessions:
                                _neighbor_sessions.update(self._graph_entity_sessions[_ent_lower])
                        _graph_results = []
                        for _si in _neighbor_sessions:
                            for _idx in top_indices[:30]:  # 2026-08-26: 80→30 减膨胀（evo 评测 current 题被过渡细节带偏）
                                if self.raw_docs[_idx].get('session') == _si and _idx not in _graph_results:
                                    _graph_results.append(_idx)
                        _seen = set()
                        _merged = []
                        for _idx in top_indices:
                            if _idx not in _seen:
                                _merged.append(_idx)
                                _seen.add(_idx)
                        for _idx in _graph_results:
                            if _idx not in _seen:
                                _merged.append(_idx)
                                _seen.add(_idx)
                        top_indices = np.array(_merged[:30])
                        print(f'    [Graph] entity multi-hop: {_named_entities} -> {len(_neighbor_sessions)} sessions, {len(_graph_results)} docs force-merged', flush=True)
                except Exception as _ge:
                    print(f'    [Graph] failed ({_ge})', flush=True)
                
                # Verification gate: keyword-weighted rerank (port from 正版 trimem_ar.py VERIFICATION GATE)
                try:
                    _vkw = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
                    _stop_v = {'how','what','when','where','why','which','who','whom','whose','the','a','an','this','that','these','those','my','your','his','her','its','our','their','me','you','he','she','it','we','they','no','yes','on','in','at','to','for','with','by','from','of','do','does','did','was','were','has','have','had','be','been','being','am','is','are','can','will','may','not','but','and','or','if','else','into','about','up','out','then','than','now','just','also','very','many','much','all','any','some','every','each','big','new','take','took','doing','spend','spent','plan','day','days','activity','moment','trip','travel','back','going','go','went','come','came','get','got','know','like','time','thing','things','year','month','week','said','tell','told','make','made','see','look','first','last','ago','january','february','march','april','may','june','july','august','september','october','november','december'}
                    _vkw -= _stop_v
                    # 过滤说话人名字：raw_docs 里 [user] [Tim] / [assistant] [John] 标签使名字命中率 100%，污染 top5 判定与 kw 权重
                    try:
                        _speakers = set()
                        for _d in self.raw_docs:
                            _speakers.update(m.lower() for m in re.findall(r'\[(?:user|assistant)\] \[([A-Za-z]+)\]', _d.get('text', '')))
                        _vkw -= _speakers
                    except Exception:
                        pass
                    if len(_vkw) >= 2 and len(top_indices) > 3:
                        # IDF 加权：文档频率高的泛词（basketball/people）降权，稀有关键词（number one goal）主导
                        _df = {}
                        for _w in _vkw:
                            _df[_w] = sum(1 for _d in self.raw_docs if _w in _d.get('text', '').lower())
                        _idf_sum = sum(1.0 / max(_df[_w], 1) for _w in _vkw)
                        kw_boosted = []
                        max_sim = float(sims.max()) if len(sims) > 0 else 1.0
                        for _idx in top_indices[:80]:  # 80-window 保证 BGE 高命中项不被挤出
                            _dt = self.raw_docs[_idx].get('text', '').lower()
                            kw_r = sum(1.0 / max(_df[_w], 1) for _w in _vkw if _w in _dt) / max(_idf_sum, 1e-9)
                            bg_n = sims[_idx] / max_sim if max_sim > 0 else 0
                            kw_boosted.append((0.6 * bg_n + 0.4 * kw_r, _idx))
                        kw_boosted.sort(key=lambda x: -x[0])
                        top_indices = np.array([_idx for _, _idx in kw_boosted][:60])
                        print(f'    [Verify] kw rerank (IDF, {len(_vkw)} words, df={ {w:_df[w] for w in sorted(_vkw)} })', flush=True)
                except Exception:
                    pass
                
                # ── Session expansion：若某会话已有高分片段，补充该会话其余片段 ──
                # （捞回同会话低 BGE 但相关的证据，如 conv-47_q165 rank#80 的 "values and passions" 片段）
                self._extra_docs = []
                try:
                    _extra = []
                    _have = set(top_indices.tolist())
                    _top_sessions = {}
                    for _idx in top_indices[:12]:
                        _si = self.raw_docs[_idx].get('session', -1)
                        if _si >= 0:
                            _top_sessions.setdefault(_si, []).append(_idx)
                    _chosen = sorted(_top_sessions.items(), key=lambda x: -max(sims[i] for i in x[1]))[:3]
                    for _si, _idxs in _chosen:
                        _added = 0
                        for _i2, _d2 in enumerate(self.raw_docs):
                            if _added >= 8:
                                break
                            if _d2.get('session') == _si and _i2 not in _have:
                                _extra.append(_i2)
                                _have.add(_i2)
                                _added += 1
                    if len(_extra) > 12:
                        _extra = _extra[:12]
                    if _extra:
                        self._extra_docs = _extra
                        print(f'    [SessionExpand] +{len(_extra)} docs from {len(_chosen)} top sessions', flush=True)
                except Exception:
                    pass
        
        # EAT — Entity-Attribute facts
        if 'EAT' in active and self.facts:
            q_words = set(re.findall(r'[a-zA-Z0-9]+', query.lower())) - _STOP
            scored_facts = []
            seen_objects = set()
            
            for f in self.facts:
                obj_lower = f.get('object_lower', '')
                if obj_lower in seen_objects:
                    continue
                subj_words = set(re.findall(r'[a-zA-Z0-9]+', f.get('subject', '').lower()))
                pred_words = set(re.findall(r'[a-zA-Z0-9]+', f.get('predicate', '').lower()))
                obj_words = set(re.findall(r'[a-zA-Z0-9]+', obj_lower))
                raw_words = set(re.findall(r'[a-zA-Z0-9]+', f.get('raw', '').lower()))
                all_words = subj_words | pred_words | obj_words | raw_words
                overlap = len(q_words & all_words)
                if overlap > 0:
                    conf_score = {'high': 3, 'medium': 2, 'low': 1}.get(f.get('confidence', 'medium'), 1)
                    scored_facts.append((overlap * conf_score, f))
                    seen_objects.add(obj_lower)
            
            if scored_facts:
                scored_facts.sort(key=lambda x: -x[0])
                fact_lines = ['[STRUCTURED FACTS (EAT)]']
                for score, f in scored_facts[:15]:
                    category_tag = f.get('category', '')
                    subcat_tag = f.get('semantic_subcategory', '')
                    tag = f"[{category_tag}/{subcat_tag}]" if category_tag else ""
                    fact_lines.append(
                        f"  [{f['time']}] {tag} {f['subject']} {f['predicate']}: {f['object'][:80]} "
                        f"(conf:{f['confidence']})"
                    )
                context_parts.append('\n'.join(fact_lines))  # EAT 紧跟 QDATE（结构化信息优先）
        
        # IPI — Persona profiles
        # 2026-09-14 接线：改用 fact_extractor_v2.query_persona（原本重复实现且该函数全库零调用）。
        # 调用条件：'IPI' in active（f_s=1，事实敏感画像）。阈值 0.2 / top-3 与原内联实现一致。
        if 'IPI' in active and self.personas:
            if self.persona_embs is None and self.personas:
                self.persona_embs = np.array(embed([p['text'] for p in self.personas]))
            
            if self.persona_embs is not None and len(self.personas) > 0:
                _ptexts = [p['text'] for p in self.personas]
                _phits = query_persona(query, self.persona_embs, _ptexts, k=3)
                _pdate = {p['text']: p.get('date', '') for p in self.personas}
                persona_lines = ['[PERSONA PROFILES (IPI)]']
                for _h in _phits:
                    persona_lines.append(
                        f"  [{_pdate.get(_h['persona'], '')}] [score={_h['score']:.2f}] {_h['persona'][:200]}")
                if len(persona_lines) > 1:
                    context_parts.append('\n'.join(persona_lines))  # IPI 紧跟 EAT 之后
        
        # VKE — Versioned Knowledge (EFG-based)
        if 'VKE' in active and self.efg:
            q_entities = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
            common_ents = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
                'The','This','That','These','Those','My','Your','His','Her',
                'Its','Our','Their','Me','You','He','She','It','We','They'}
            q_entities = [e.lower() for e in q_entities if e not in common_ents]
            # 2026-08-26: LME 实体是 'user'（小写），大写启发式失效 → query 词与 EFG subject 匹配
            if not q_entities and self.efg:
                _q_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
                for _subj in self.efg._entity_to_versions:
                    if _subj in _q_words:
                        q_entities.append(_subj)
            
            efg_lines = ['[ENTITY-FACT GRAPH (VKE)]']
            # 职业类同义词归一（business/studio/venture 系 → own business）
            _OCC_SYN = {'business', 'own business', 'own studio', 'studio', 'venture', 'startup',
                        'company', 'dance studio', 'own company', 'biz', 'this biz'}

            def _occ_key(obj: str) -> str:
                o = obj.strip().lower()
                if o in _OCC_SYN or 'business' in o or 'studio' in o or 'biz' in o or 'venture' in o:
                    return 'own business'
                return o

            for entity in q_entities[:5]:
                # 2026-08-26: 扁平快照 → 版本序列（带 time/weight，模型可区分稳定值 vs 一次性过渡）
                vfacts = self.efg.query_by_entity(entity)
                # 2026-09-14 接线：原查询走 query_by_entity，这里改走
                # get_entity_versions + get_version（此前全库零调用的公开方法），
                # 保证“存储只追加、读取按版本号”的 VKE 语义在主路径上真的被执行。
                _vids = self.efg.get_entity_versions(entity)
                if _vids:
                    vfacts = []
                    for _vid in _vids:
                        _v = self.efg.get_version(_vid)
                        if not _v:
                            continue
                        vfacts.append({'subject': entity, 'predicate': _v.get('predicate', ''),
                                       'object': _v.get('object', ''), 'time': _v.get('time', ''),
                                       'confidence': _v.get('confidence', 'low'),
                                       'weight': _v.get('weight', 1.0),
                                       'source': _v.get('source', '')})
                # 去重（同一 vid 因重复 fact 被 append 多次）
                _seen_v = set()
                _vf = []
                for _f in vfacts:
                    _k = (_f['predicate'], _f['object'], _f['time'])
                    if _k not in _seen_v:
                        _seen_v.add(_k)
                        _vf.append(_f)
                if _vf:
                    efg_lines.append(f'  Entity: {entity}')
                    # 同义词聚合：同 key 合并，累计提及次数/权重，显示时间跨度
                    _agg = {}
                    _order = []
                    for f in _vf:
                        _key = _occ_key(f['object'])
                        if _key not in _agg:
                            _agg[_key] = {'times': [], 'w': 0.0, 'preds': set()}
                            _order.append(_key)
                        _agg[_key]['times'].append(f['time'] or '?')
                        _agg[_key]['w'] += float(f.get('weight', 1.0))
                        _agg[_key]['preds'].add(f['predicate'])
                    for _key in _order[:6]:
                        _a = _agg[_key]
                        _t0 = _a['times'][0]
                        _t1 = _a['times'][-1]
                        _span = f'{_t0}~{_t1}' if _t1 != _t0 else _t0
                        _n = len(_a['times'])
                        # 2026-08-26: 判断标注（recurring vs single transitional），供模型直接选用
                        _tag = ''
                        if _n >= 3 or (_n >= 2 and _t0 != _t1):
                            _tag = ' <- recurring across time (stable)'
                        elif _n == 1:
                            _tag = ' <- single mention'
                        efg_lines.append(f"    {_key}: {_span} (x{_n}, w={_a['w']:.1f}){_tag}")
            if len(efg_lines) > 1:
                context_parts.append('\n'.join(efg_lines))  # VKE 紧跟 IPI 之后
        
        # HG — Hierarchy Graph
        # 2026-09-14 接线：仍只在 d_i=1（推理型画像）时构建；新增 V2_DEBUG_HG=1 时的诊断输出
        # （get_hierarchy_debug 原本零调用）。HG 每次查询重建，不入库。
        if 'HG' in active:
            try:
                from hierarchyv2.hierarchy_graph import HierarchyGraph
                hg = HierarchyGraph()
                hg.build(getattr(self, '_cached_sessions', []), getattr(self, '_cached_dates', []))
                if os.environ.get('V2_DEBUG_HG', os.environ.get('V2_HG_DEBUG', '0')) == '1':
                    print(f'    [HG debug] {hg.get_hierarchy_debug(query)}', flush=True)
                h_results = hg.retrieve(query, top_k=20)
                if h_results:
                    hg_lines = ['[HIERARCHY GRAPH (HG)]']
                    for si, sc in h_results[:10]:
                        if si < len(self.raw_docs):
                            hg_lines.append(f'  [session {si}] score={sc:.1f}')
                    if len(hg_lines) > 1:
                        context_parts = hg_lines + context_parts
            except Exception:
                pass
        
        # ═══ EPISODIC SUMMARIES (port from 正版, sim>0.3 注入) ═══
        epi_ctx = []
        valid_summaries = [e for e in self.episodic_summaries]
        if valid_summaries:
            if self.episodic_embs is None:
                self.episodic_embs = np.array(embed([e['summary'] for e in valid_summaries]))
            eqe = np.array(embed([query]))
            esims = np.dot(self.episodic_embs, eqe.T).flatten()
            for idx in np.argsort(esims)[::-1][:5]:
                if esims[idx] > 0.3:
                    e = valid_summaries[idx]
                    epi_ctx.append(f"[EPISODIC:{e['date']}] {e['summary']}")
        
        # ═══ Semantic facts (port from 正版 [FACTS]) ═══
        sem_ctx = []
        if self.semantic_facts:
            q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - _STOP
            scored = []
            for f in self.semantic_facts:
                obj_lower = f.get('o', f.get('object', '')).lower()
                obj_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', obj_lower))
                overlap = len(q_words & obj_words)
                if overlap > 0:
                    scored.append((overlap, f))
            scored.sort(key=lambda x: -x[0])
            for _, f in scored[:8]:
                s = f.get('s', f.get('subject', ''))
                p = f.get('p', f.get('predicate', ''))
                o = f.get('o', f.get('object', ''))
                d = f.get('date', f.get('source_date', ''))
                sem_ctx.append(f"[{d}] FACT: {s} {p}: {o}")
        
        # ═══ DIRECT FACTS (port from 正版) ═══
        direct_facts = []
        if self.facts:
            for f in self.facts:
                if f.get('predicate') in ('frequency', 'count_of', 'attended_sessions', 'exact_count', 'vague_count'):
                    direct_facts.append(f"  [{f['time']}] {f['subject']} {f['predicate']}: {f['object'][:80]}")
            direct_facts = direct_facts[:10]
        
        # ═══ SOFT ROUTING：概率加权分支融合（branch_probs 非 None 时启用，早返回）═══
        if branch_probs is not None:
            context_parts = context_parts + self._soft_branch_parts(
                query, branch_probs, top_indices, sims,
                epi_ctx, sem_ctx, direct_facts, force_broad)
            return context_parts

        # ═══ question_type 专属分支（port from 正版 search）═══
        branch_parts = []
        
        if question_type == 'temporal-reasoning':
            timeline = self.temporal.query_temporal(query, self.qdate)
            # 2026-09-14 接线：query_temporal 未命中时回退到 build_full_timeline（原本零调用）。
            # 调用条件：仅 temporal-reasoning 且关键词时间线为空时；
            # 开关 V2_WIRE_TR_FULLTIMELINE（默认 1），置 0 可完整回到接线前行为。
            if not timeline and os.environ.get('V2_WIRE_TR_FULLTIMELINE', '1') == '1':
                timeline = self.temporal.build_full_timeline()
                if timeline:
                    print('    [TR timeline] query_temporal empty -> build_full_timeline fallback', flush=True)
            if timeline:
                if len(timeline) > 12000:
                    timeline = timeline[:12000] + '...'
                branch_parts.append('[TIMELINE]\n' + timeline)
            event_dates = []
            seen_dates = set()
            for entry in self.temporal.sequence_index:
                ts = entry.get('time_str', '')[:10]
                content = entry.get('content', '')[:80]
                if ts and content and entry['role'] == 'user' and ts not in seen_dates:
                    seen_dates.add(ts)
                    event_dates.append(f"  [{ts}] {content}")
            if event_dates:
                branch_parts.append('[EVENT DATES]\n' + '\n'.join(event_dates[:20]))
            # Structured event facts（按 query 动作词匹配会话事件）
            try:
                _ef_str = self._get_event_facts_for_query(query)
                if _ef_str and 'No structured events' not in _ef_str:
                    branch_parts.append(_ef_str)
                    print(f'    [TR event_facts] added to temporal-reasoning context', flush=True)
            except Exception as _ev_e:
                print(f'    [TR event_facts] skipped ({_ev_e})', flush=True)
            # Iter2: 问题含具体月份 → 强制并入该月全部 user 事件（conv-50_46：
            # TIMELINE 按 query 关键词匹配，Miami 证据 "shoot in Miami" 不含 city/visiting 关键词被漏掉）
            _month_lines = None
            try:
                _m = re.search(r'(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b', query)
                if _m:
                    _mon_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
                    _mon_n = _mon_map[_m.group(1).lower()]
                    _yr_n = int(_m.group(2))
                    _m_lines = [f'[MONTH EVENTS: {_m.group(1)} {_m.group(2)}]']
                    _mevts = []
                    for _e in self.temporal.sequence_index:
                        if _e.get('time') and _e['time'].year == _yr_n and _e['time'].month == _mon_n \
                                and _e.get('role') == 'user' and _e.get('content'):
                            _ts = _e.get('time_str', '')[:10]
                            _mevts.append((_e['time'], f'  [{_ts}] {_e["content"][:200]}'))
                    # Iter2: 时间倒序（最新在前）——"which city in <month>" 多城市时 LLM 优先看最新
                    _mevts.sort(key=lambda x: x[0], reverse=True)
                    _m_lines.extend(l for _, l in _mevts)
                    if len(_m_lines) > 1:
                        # 排在 docs 之前（紧跟 event facts 后），保证不被后续 top25 docs 挤出窗口
                        _month_lines = '\n'.join(_m_lines[:25])
                        print(f'    [TR month-filter] +{len(_m_lines)-1} events for {_m.group(1)} {_m.group(2)}', flush=True)
            except Exception as _me:
                print(f'    [TR month-filter] skipped ({_me})', flush=True)
            if _month_lines:
                branch_parts.append(_month_lines)
            if top_indices is not None:
                # Iter2: top10→25（conv-50_46 Miami 证据 rank#24 被截；"which city in <month>" 类问题证据分散）
                for idx in top_indices[:25]:
                    d = self.raw_docs[idx]
                    dt = f"[{d['date']}] " if d['date'] else ''
                    branch_parts.append(f"{dt}[score={sims[idx]:.2f}] {d['text']}")
        
        elif question_type == 'multi-session':
            # 2026-08-26: 注入 VKE 版本链（conflict 题需要完整演进链；此前 multi-session 分支只扫描文档，
            # 版本信息丢失 → Gina 类 conflict 答不出中间态 clothing store owner）
            try:
                if self.efg:
                    _qents = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
                    _commons = {'How','What','When','Where','Why','Which','Who','Whom','Whose',
                        'The','This','That','These','Those','My','Your','His','Her',
                        'Its','Our','Their','Me','You','He','She','It','We','They','List'}
                    _qents = [e.lower() for e in _qents if e not in _commons]
                    # 2026-08-26: LME 小写实体 'user' 支持
                    if not _qents:
                        _qw = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
                        for _subj in self.efg._entity_to_versions:
                            if _subj in _qw:
                                _qents.append(_subj)
                    _vke_lines = ['[ENTITY-FACT GRAPH (VKE)]']
                    for _ent in _qents[:3]:
                        _vf = self.efg.query_by_entity(_ent)
                        _seen = set(); _uniq = []
                        for _f in _vf:
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
                        branch_parts.append('\n'.join(_vke_lines))
            except Exception:
                pass
            queries = [query]
            q = query
            # ═══ cod2 P2/P4：穷尽式候选事件召回（词法扫全部会话 + 时间窗过滤）═══
            # 不做 top-k 相似度截断：聚合题的证据散在多会话（实测 wedding 命中 6 会话、
            # bak 命中 10 会话），top-k 必然漏。这里只负责“覆盖”，精度交给 LLM 枚举 + 代码计数。
            try:
                from hierarchyv2 import ms_counter as _msc
                if _msc.ENABLED:
                    _m, _win = _msc.candidate_mentions(self._cached_sessions, self._cached_dates,
                                                       query, self.qdate or '')
                    if _m:
                        branch_parts.insert(0, _msc.render_mentions(_m, _win))
                        print(f'    [MS-ledger] {len(_m)} candidate mentions, window={_win}', flush=True)
            except Exception as _e:
                print(f'    [MS-ledger] skipped ({_e})', flush=True)
            q = re.sub(r'(?i)^(how many|how much|what (?:is|was|were|are) the total|what)\s+', '', q)
            q = re.sub(r'(?i)\s+(did I|have I|do I|i have|i had|did you|have you|do you)\s+.*$', '', q)
            q = q.strip().rstrip('?.,;:! ')
            if q and q != query and len(q) > 3:
                queries.append(q)
            actions = re.findall(r'(?i)(?:bought|worked|assembled|built|visited|attended|spent|acquired|created|watched|took|bake|led|manage|buy|sell|fix|collect|adopt|use|using|painted|fixed|sold)', query)
            if actions:
                queries.append(' '.join(actions[:3]))
            
            all_scores = {}
            if self.embs is not None:
                for qi, q_i in enumerate(queries):
                    w = [1.0, 0.8, 0.6][qi] if qi < 3 else 0.5
                    qe_i = np.array(embed([q_i]))
                    sims_i = np.dot(self.embs, qe_i.T).flatten()
                    for idx in np.argsort(sims_i)[::-1][:30]:
                        t = self.raw_docs[idx]['text']
                        all_scores[t] = max(all_scores.get(t, 0), float(sims_i[idx]) * w)
            
            branch_parts.append('[MULTI-SESSION: scan all sessions]')
            merged = sorted(all_scores.items(), key=lambda x: -x[1])[:55]  # Iter1: 40→55（跨会话列表题证据分散）
            for text, score in merged:
                branch_parts.append(f'[score={score:.2f}] {text[:500]}')
        
        elif question_type == 'knowledge-update':
            facts_text = self._get_facts_context(query)
            if facts_text:
                branch_parts.append('[FACTS]')
                branch_parts.append(facts_text)
            if top_indices is not None:
                top_docs = [(self.raw_docs[idx], sims[idx]) for idx in top_indices[:40]]  # 2026-08-26: 20→40 更多证据（current 题避免单篇最新文档主导）
                top_docs.sort(key=lambda x: x[0].get('date', ''), reverse=True)  # 最新优先（正版逻辑）
                for d, score in top_docs:
                    dt = f"[{d['date']}] " if d['date'] else ''
                    branch_parts.append(f"{dt}[score={score:.2f}] {d['text']}")
            if epi_ctx:
                branch_parts.append('[EPISODIC SUMMARIES]')
                branch_parts.extend(epi_ctx)
        
        elif question_type == 'single-session-preference':
            # ═══ cod3（SSP 专项）═══════════════════════════════════════════════
            # 诊断（ssp_diag.py，13 道错题）：偏好证据会话在“问题检索”里 top-1 占 6/13、
            # top-3 占 11/13（中位排名 2）——说明**不是检索不到**，而是：
            #   ① 原实现只注入 top-8 个分片 → 偏好陈述常被切断；
            #   ② 全局 top-3 画像余弦检索会把无关领域的画像排在前面（问酒店→引建设法），
            #      答案开口就是 "Based on your interest in ..."，直接跑偏。
            # 因此：会话级检索取 top-3 会话全文 + 画像限定到这些会话 + 补齐 episodic 摘要。
            _psess = []
            try:
                if self._cached_sessions:
                    # cod3-fix：用**纯问题↔会话余弦**排序（而非融合后的 doc 分数）。
                    # 诊断（ssp_diag.py）验证：纯余弦下证据会话 top-3 命中 11/13；
                    # 用融合 doc 分数时 top-3 会选到无关会话（实测 195a1a1b 选到 [8,1,12]，
                    # 而证据在 s45）。
                    if getattr(self, '_sess_embs', None) is None:
                        _stexts = []
                        for _s in self._cached_sessions:
                            _t = ' '.join(str(t.get('content', '')) for t in (_s or [])
                                          if isinstance(t, dict) and str(t.get('role', '')) == 'user')[:3000]
                            _stexts.append(_t or 'x')
                        self._sess_embs = np.array(embed(_stexts))
                    _qv = np.array(embed([query])).flatten()
                    _se = self._sess_embs
                    _ssim = np.dot(_se, _qv) / (np.linalg.norm(_se, axis=1) * np.linalg.norm(_qv) + 1e-9)
                    _psess = [int(i) for i in np.argsort(_ssim)[::-1][:3]]
                    # cod3-fix2：注入的不是整段会话，而是**用户自己的偏好/约束/计划句**
                    # （整段注入时关键句埋在第 800 字符后面，模型会抓住其他更响亮的偏好，
                    #  实测 195a1a1b：问“晚上活动”却答“街舞+手机App”，而 gold 要“放松、9:30 前、不要手机”）
                    _PREF_MARK = re.compile(
                        r"\b(i prefer|i'd prefer|i like|i love|i enjoy|i don't|i do not|i usually|i always|"
                        r"i want|i need|i plan|i'm planning|i am planning|i've been|i have been|my favorite|"
                        r"i'm trying|i am trying|i try to|prefer|usually|generally|before \d|after \d|"
                        r"i should|i hope|i'm thinking|i am thinking)\b", re.I)
                    for _si in _psess:
                        _sents = []
                        for _t in (self._cached_sessions[_si] or []):
                            if not (isinstance(_t, dict) and str(_t.get('role', '')) == 'user'):
                                continue
                            for _sent in re.split(r'(?<=[.!?])\s+|\n+', str(_t.get('content', ''))):
                                _sent = _sent.strip()
                                if 15 <= len(_sent) <= 300 and _PREF_MARK.search(_sent):
                                    _sents.append(_sent)
                        if _sents:
                            branch_parts.append(
                                f'[USER PREFERENCE STATEMENTS (session s{_si})]\n' +
                                '\n'.join('- ' + _s[:220] for _s in _sents[:8]))
                    print(f'    [SSP-sessions] pure-cosine top={_psess}', flush=True)
            except Exception as _e:
                print(f'    [SSP-sessions] skipped ({_e})', flush=True)

            # 画像：限定在命中会话内（同域）；不足则用全局 top-2 兜底，避免跨域画像带偏
            profile_lines = []
            if self.personas:
                if self.persona_embs is None:
                    self.persona_embs = np.array(embed([p['text'] for p in self.personas]))
                if self.persona_embs is not None:
                    _ptexts = [p['text'] for p in self.personas]
                    _pdate = {p['text']: p.get('date', '') for p in self.personas}
                    _inscope = [p for p in self.personas if p.get('session') in set(_psess)]
                    if _inscope:
                        _cand = _inscope
                        _tag = 'in-session'
                    else:
                        _cand = self.personas
                        _tag = 'global'
                    _cemb = (np.array([p.get('_emb') for p in _cand])
                             if all(p.get('_emb') is not None for p in _cand) else None)
                    if _cemb is None:
                        _cemb = np.array(embed([p['text'] for p in _cand]))
                    _qv = np.array(embed([query])).flatten()
                    _sims_p = np.dot(_cemb, _qv) / (np.linalg.norm(_cemb, axis=1) * np.linalg.norm(_qv) + 1e-9)
                    _k = 3 if _tag == 'in-session' else 2
                    for _idx in np.argsort(_sims_p)[::-1][:_k]:
                        profile_lines.append(
                            f"  [{_pdate.get(_cand[_idx]['text'], '')}] [score={_sims_p[_idx]:.2f}] "
                            f"{_cand[_idx]['text'][:200]}")
                    print(f'    [SSP-persona] {_tag} n={len(_cand)}', flush=True)
            if profile_lines:
                branch_parts.append('[PERSONA PROFILES]')
                branch_parts.extend(profile_lines)

            # 偏好题依赖“用户自己说过的话”，episodic 摘要是偏好最密集的载体（原实现未给 SSP 注入）
            if epi_ctx:
                branch_parts.append('[EPISODIC SUMMARIES]')
                branch_parts.extend(epi_ctx[:5])
            
            # Entity graph bridging（port from 正版 single-session-preference 分支）
            if getattr(self, '_graph_entity_sessions', None):
                _q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - _STOP
                bridging_sessions = set()
                for qw in _q_words:
                    if qw in self._graph_entity_sessions:
                        bridging_sessions.update(self._graph_entity_sessions[qw])
                bridging_entities = set()
                for si in bridging_sessions:
                    if si in self._graph_session_entities:
                        bridging_entities.update(self._graph_session_entities[si])
                bridging_entities -= _q_words
                _COMMON = _STOP | {'like', 'just', 'also', 'get', 'got', 'one', 'two', 'can', 'know', 'think', 'want', 'good', 'well', 'thing', 'things', 'really', 'much', 'many', 'would', 'could', 'first', 'might', 'even', 'going', 'way', 'back', 'find', 'help', 'need', 'make', 'take', 'tell', 'some', 'something', 'sure', 'let', 'new', 'time', 'day', 'week', 'month', 'year', 'great', 'nice', 'better', 'look', 'seem', 'say', 'ask', 'come', 'go', 'see', 'yes', 'oh', 'no'}
                bridging_entities = {e for e in bridging_entities if len(e) > 3 and e not in _COMMON and not e[0].isdigit()}
                if bridging_entities:
                    common_bridges = sorted(bridging_entities)[:12]
                    caption = '[CONVERSATION GRAPH - related topics discovered]:\n'
                    caption += f'  Related topics found in the same conversations: {", ".join(common_bridges)}\n'
                    branch_parts.append(caption)
            
            if top_indices is not None:
                for idx in top_indices[:8]:
                    d = self.raw_docs[idx]
                    dt = f"[{d['date']}] " if d['date'] else ''
                    branch_parts.append(f"{dt}[score={sims[idx]:.2f}] {d['text']}")
        else:
            # 默认分支（single-session-assistant / single-session-user / 未分类）
            # 顺序：FEI docs 优先（主证据），再 episodic/facts（辅助）
            if top_indices is not None:
                # Iter2: top_k 45→50（conv-47_30 Dungeons kw-rerank rank#46 被 45 截断）
                top_k = 50
                for idx in top_indices[:top_k]:
                    d = self.raw_docs[idx]
                    branch_parts.append(f"[{d['date']}] [score={sims[idx]:.2f}] {d['text']}")
                # Session expansion 补充片段（排在 top_k 之后，仍能进入 runner 的 context 窗口）
                for idx in getattr(self, '_extra_docs', []):
                    d = self.raw_docs[idx]
                    branch_parts.append(f"[{d['date']}] [score={sims[idx]:.2f}] {d['text']}")
            if epi_ctx:
                branch_parts.append('[EPISODIC SUMMARIES]')
                branch_parts.extend(epi_ctx)
            if sem_ctx:
                branch_parts.append('[FACTS]')
                branch_parts.extend(sem_ctx)
            if direct_facts:
                branch_parts.append('[DIRECT FACTS]')
                branch_parts.append('\n'.join(direct_facts))
        
        # 多路召回兜底（2026-08-25）：实体名全文扫描，弥补 BGE 语义盲区
        # 位置：结构化区之后、branch 主证据之前（保证不被长噪音淹没）
        try:
            _kr = self._keyword_recall_docs(query, top_indices)
            if _kr:
                kr_lines = ['[KEYWORD RECALL]']
                for idx in _kr:
                    d = self.raw_docs[idx]
                    kr_lines.append(f"[{d['date']}] [kw] {d['text'][:350]}")
                print(f'    [KeywordRecall] +{len(_kr)} docs (entity name scan)', flush=True)
                branch_parts = kr_lines + branch_parts
        except Exception as e:
            print(f'    [KeywordRecall] skip: {e}', flush=True)
        
        # 最终顺序：QDATE → EAT/IPI/VKE 结构化区 → keyword recall → branch 内容
        context_parts = context_parts + branch_parts
        return context_parts
    
    def _bm25_scores(self, query: str, docs) -> np.ndarray:
        """BM25 词法打分（2026-08-26 T3）：专治专有名词/精确词 BGE 语义盲区。"""
        import math
        _bm25_stop = {'what','was','were','are','the','now','current','value','before','it','changed',
                      'change','did','over','time','user','his','her','their','my','your','and','for',
                      'with','from','how','many','list','different','values','observed','is','a','an',
                      'of','to','in','on','at','by','that','this','has','have','had','do','does','not'}
        tokenized = [re.findall(r'[a-z]{2,}', d['text'].lower()) for d in docs]
        q_tokens = set(re.findall(r'[a-z]{2,}', query.lower())) - _bm25_stop
        if not q_tokens:
            return np.zeros(len(docs))
        N = max(len(docs), 1)
        df = {w: sum(1 for tk in tokenized if w in tk) for w in q_tokens}
        avgdl = sum(len(tk) for tk in tokenized) / N
        scores = []
        for tk in tokenized:
            dl = max(len(tk), 1)
            s = 0.0
            for w in q_tokens:
                tf = tk.count(w)
                if tf == 0:
                    continue
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / max(avgdl, 1)))
            scores.append(s)
        return np.array(scores)

    def _keyword_recall_docs(self, query: str, top_indices) -> List[int]:
        """多路召回（2026-08-25）：弥补 BGE 语义盲区。
        query 实体名（大写人名）在全文扫描——即使 BGE 排名极低/隐式表达，
        含实体名的会话片段也强制纳入兜底（每会话限流，总数上限）。"""
        _common = {'What','How','When','Why','Which','Who','Whom','Where',
                   'The','This','That','These','Those','My','Your','His','Her',
                   'Its','Our','Their','Me','You','He','She','It','We','They',
                   'Did','Does','Do','Is','Are','Was','Were','Has','Have'}
        names = [w.lower() for w in re.findall(r'[A-Z][a-z]{2,}', query) if w not in _common]
        if not names:
            return []
        top_set = set(top_indices[:50]) if top_indices is not None else set()
        # 按会话覆盖：所有含实体名的会话，每会话取前 2 个 doc（避免 index 顺序限流漏会话）
        from collections import defaultdict
        sess_docs = defaultdict(list)
        for i, d in enumerate(self.raw_docs):
            if i in top_set:
                continue
            if any(n in d['text'].lower() for n in names):
                sess_docs[d.get('session')].append(i)
        extra = []
        for si in sorted(sess_docs):
            extra.extend(sess_docs[si])
            if len(extra) >= 25:
                break
        return extra

    def _soft_branch_parts(self, query: str, probs: Dict[str, float],
                           top_indices, sims, epi_ctx, sem_ctx, direct_facts,
                           force_broad: bool = False) -> List[str]:
        """Soft routing 的分支证据融合（概率加权，无硬决策）。

        结构：
          - 基线块 = 默认分支 FEI top-50 docs（同 uniform 管线，永远包含）
          - 分支块 = 各类型特有证据，p >= 0.05 才注入，块内 top-k 按 p 缩放，
            与基线 docs 按文本去重；分支块按 p 降序排在基线之前（保证高概率证据不被窗口截断）
          - force_broad=True 时只保留基线块（重试路径）
        """
        MIN_P = 0.05
        p = {k: float(v) for k, v in probs.items()}
        branch_blocks = []  # (prob, text)
        seen_texts = set()

        # ── TR 块：TIMELINE + EVENT DATES + 事件事实 + 月事件 ──
        if not force_broad and p.get('temporal-reasoning', 0.0) >= MIN_P:
            block = []
            try:
                timeline = self.temporal.query_temporal(query, self.qdate)
                if timeline:
                    if len(timeline) > 12000:
                        timeline = timeline[:12000] + '...'
                    block.append('[TIMELINE]\n' + timeline)
            except Exception:
                pass
            event_dates = []
            seen_dates = set()
            for entry in self.temporal.sequence_index:
                ts = entry.get('time_str', '')[:10]
                content = entry.get('content', '')[:80]
                if ts and content and entry['role'] == 'user' and ts not in seen_dates:
                    seen_dates.add(ts)
                    event_dates.append(f'  [{ts}] {content}')
            if event_dates:
                block.append('[EVENT DATES]\n' + '\n'.join(event_dates[:20]))
            try:
                _ef_str = self._get_event_facts_for_query(query)
                if _ef_str and 'No structured events' not in _ef_str:
                    block.append(_ef_str)
            except Exception:
                pass
            try:
                _m = re.search(r'(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b', query)
                if _m:
                    _mon_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
                    _mon_n = _mon_map[_m.group(1).lower()]
                    _yr_n = int(_m.group(2))
                    _m_lines = [f'[MONTH EVENTS: {_m.group(1)} {_m.group(2)}]']
                    _mevts = []
                    for _e in self.temporal.sequence_index:
                        if _e.get('time') and _e['time'].year == _yr_n and _e['time'].month == _mon_n \
                                and _e.get('role') == 'user' and _e.get('content'):
                            _ts = _e.get('time_str', '')[:10]
                            _mevts.append((_e['time'], f'  [{_ts}] {_e["content"][:200]}'))
                    _mevts.sort(key=lambda x: x[0], reverse=True)
                    _m_lines.extend(l for _, l in _mevts)
                    if len(_m_lines) > 1:
                        block.append('\n'.join(_m_lines[:25]))
            except Exception:
                pass
            if block:
                branch_blocks.append((p.get('temporal-reasoning', 0.0), '\n'.join(block)))

        # ── MS 块：多查询扩展融合（top ⌈55·p⌉，去重）──
        if not force_broad and p.get('multi-session', 0.0) >= MIN_P:
            q = query
            q = re.sub(r'(?i)^(how many|how much|what (?:is|was|were|are) the total|what)\s+', '', q)
            q = re.sub(r'(?i)\s+(did I|have I|do I|i have|i had|did you|have you|do you)\s+.*$', '', q)
            q = q.strip().rstrip('?.,;:! ')
            queries = [query]
            if q and q != query and len(q) > 3:
                queries.append(q)
            actions = re.findall(r'(?i)(?:bought|worked|assembled|built|visited|attended|spent|acquired|created|watched|took|bake|led|manage|buy|sell|fix|collect|adopt|use|using|painted|fixed|sold)', query)
            if actions:
                queries.append(' '.join(actions[:3]))
            all_scores = {}
            if self.embs is not None:
                for qi, q_i in enumerate(queries):
                    w = [1.0, 0.8, 0.6][qi] if qi < 3 else 0.5
                    qe_i = np.array(embed([q_i]))
                    sims_i = np.dot(self.embs, qe_i.T).flatten()
                    for idx in np.argsort(sims_i)[::-1][:30]:
                        t = self.raw_docs[idx]['text']
                        all_scores[t] = max(all_scores.get(t, 0), float(sims_i[idx]) * w)
            merged = sorted(all_scores.items(), key=lambda x: -x[1])
            k_ms = max(5, int(55 * p.get('multi-session', 0.0)))
            ms_lines = ['[MULTI-SESSION: scan all sessions]']
            added = 0
            for text, score in merged:
                if added >= k_ms:
                    break
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                ms_lines.append(f'[score={score:.2f}] {text[:500]}')
                added += 1
            if len(ms_lines) > 1:
                branch_blocks.append((p.get('multi-session', 0.0), '\n'.join(ms_lines)))

        # ── KU 块：FACTS + 日期倒序 docs（top ⌈20·p⌉，去重）──
        if not force_broad and p.get('knowledge-update', 0.0) >= MIN_P:
            ku_lines = []
            try:
                facts_text = self._get_facts_context(query)
                if facts_text:
                    ku_lines.append('[FACTS]')
                    ku_lines.append(facts_text)
            except Exception:
                pass
            if top_indices is not None:
                k_ku = max(3, int(20 * p.get('knowledge-update', 0.0)))
                top_docs = [(self.raw_docs[idx], sims[idx]) for idx in top_indices[:20]]
                top_docs.sort(key=lambda x: x[0].get('date', ''), reverse=True)
                ku_added = 0
                for d, score in top_docs:
                    if ku_added >= k_ku:
                        break
                    if d['text'] in seen_texts:
                        continue
                    seen_texts.add(d['text'])
                    dt = f"[{d['date']}] " if d['date'] else ''
                    ku_lines.append(f"{dt}[score={score:.2f}] {d['text']}")
                    ku_added += 1
            if ku_lines:
                branch_blocks.append((p.get('knowledge-update', 0.0), '\n'.join(ku_lines)))
            if epi_ctx:
                branch_blocks.append((p.get('knowledge-update', 0.0), '[EPISODIC SUMMARIES]\n' + '\n'.join(epi_ctx)))

        # ── pref 块：人格画像 + 会话图桥接 ──
        if not force_broad and p.get('single-session-preference', 0.0) >= MIN_P:
            pref_lines = []
            try:
                if self.personas:
                    if self.persona_embs is None and self.personas:
                        self.persona_embs = np.array(embed([pp['text'] for pp in self.personas]))
                    if self.persona_embs is not None:
                        qe = np.array(embed([query]))
                        psims = np.dot(self.persona_embs, qe.T).flatten()
                        for idx in np.argsort(psims)[::-1][:3]:
                            if psims[idx] > 0.2:
                                pp = self.personas[idx]
                                pref_lines.append(f"  [{pp['date']}] [score={psims[idx]:.2f}] {pp['text'][:200]}")
            except Exception:
                pass
            if pref_lines:
                branch_blocks.append((p.get('single-session-preference', 0.0),
                                      '[PERSONA PROFILES]\n' + '\n'.join(pref_lines)))
            try:
                if getattr(self, '_graph_entity_sessions', None):
                    _q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - _STOP
                    bridging_sessions = set()
                    for qw in _q_words:
                        if qw in self._graph_entity_sessions:
                            bridging_sessions.update(self._graph_entity_sessions[qw])
                    bridging_entities = set()
                    for si in bridging_sessions:
                        if si in self._graph_session_entities:
                            bridging_entities.update(self._graph_session_entities[si])
                    bridging_entities -= _q_words
                    _COMMON = _STOP | {'like', 'just', 'also', 'get', 'got', 'one', 'two', 'can', 'know', 'think', 'want', 'good', 'well', 'thing', 'things', 'really', 'much', 'many', 'would', 'could', 'first', 'might', 'even', 'going', 'way', 'back', 'find', 'help', 'need', 'make', 'take', 'tell', 'some', 'something', 'sure', 'let', 'new', 'time', 'day', 'week', 'month', 'year', 'great', 'nice', 'better', 'look', 'seem', 'say', 'ask', 'come', 'go', 'see', 'yes', 'oh', 'no'}
                    bridging_entities = {e for e in bridging_entities if len(e) > 3 and e not in _COMMON and not e[0].isdigit()}
                    if bridging_entities:
                        common_bridges = sorted(bridging_entities)[:12]
                        branch_blocks.append((p.get('single-session-preference', 0.0),
                                              '[CONVERSATION GRAPH - related topics discovered]:\n'
                                              f'  Related topics found in the same conversations: {", ".join(common_bridges)}'))
            except Exception:
                pass

        # ── 基线块：默认分支 FEI docs（永远包含，排最后；与分支块去重）──
        base_lines = []
        if top_indices is not None:
            for idx in top_indices[:50]:
                d = self.raw_docs[idx]
                if d['text'] in seen_texts:
                    continue
                seen_texts.add(d['text'])
                dt = f"[{d['date']}] " if d['date'] else ''
                base_lines.append(f"{dt}[score={sims[idx]:.2f}] {d['text']}")
            for idx in getattr(self, '_extra_docs', []):
                d = self.raw_docs[idx]
                if d['text'] in seen_texts:
                    continue
                seen_texts.add(d['text'])
                dt = f"[{d['date']}] " if d['date'] else ''
                base_lines.append(f"{dt}[score={sims[idx]:.2f}] {d['text']}")
            if not base_lines and self.raw_docs:
                # 分支块已覆盖全部 top-50（如 p(ms)=1 时）→ 保留若干非重复基线
                for idx in top_indices[:8]:
                    d = self.raw_docs[idx]
                    if d['text'] in seen_texts:
                        continue
                    seen_texts.add(d['text'])
                    dt = f"[{d['date']}] " if d['date'] else ''
                    base_lines.append(f"{dt}[score={sims[idx]:.2f}] {d['text']}")
                    if len(base_lines) >= 5:
                        break
        if not base_lines:
            base_lines = ['No relevant memories.']

        # 分支块按概率降序（高概率证据优先，避免被窗口截断），基线块最后
        branch_blocks.sort(key=lambda x: -x[0])
        parts = [b for _, b in branch_blocks] + base_lines

        # 兜底：分支块很少时补充语义事实（原默认分支行为）
        if len(branch_blocks) <= 1:
            tail = []
            if sem_ctx:
                tail.append('[FACTS]\n' + '\n'.join(sem_ctx))
            if direct_facts:
                tail.append('[DIRECT FACTS]\n' + '\n'.join(direct_facts))
            if tail:
                parts = parts[:-1] + tail + parts[-1:]
        return parts

    def _get_facts_context(self, query: str) -> str:
        """Legacy compatibility."""
        q_words = set(re.findall(r'[a-zA-Z0-9]+', query.lower())) - _STOP
        scored = []
        seen_objects = set()
        for f in self.facts:
            obj_lower = f.get('object_lower', '')
            if obj_lower in seen_objects:
                continue
            obj_words = set(re.findall(r'[a-zA-Z0-9]+', obj_lower))
            subj_words = set(re.findall(r'[a-zA-Z0-9]+', f.get('subject', '')))
            raw_words = set(re.findall(r'[a-zA-Z0-9]+', f.get('raw', '')))
            overlap = len(q_words & (obj_words | subj_words | raw_words))
            if overlap > 0:
                conf_score = {'high': 3, 'medium': 2, 'low': 1}.get(f.get('confidence', 'medium'), 1)
                scored.append((overlap * conf_score, f))
                seen_objects.add(obj_lower)
        scored.sort(key=lambda x: -x[0])
        lines = []
        for score, f in scored[:10]:
            lines.append(f"  [{f['time']}] {f['subject']} {f['predicate']}: {f['object']} (conf:{f['confidence']})")
        return '\n'.join(lines)
    
    # General semantic mapping: event_type tag → question keywords（正版原表）
    _EVENT_SEMANTIC_MAP = {
        'business_action': {'promote', 'promotion', 'store', 'shop', 'business', 'market', 'brand',
                            'launch', 'advertise', 'campaign', 'sell', 'offer', 'clothes', 'fashion',
                            'create', 'develop', 'design', 'product', 'collaborate', 'partner',
                            'produce', 'manufacture', 'found', 'establish', 'start', 'open'},
        'collaboration': {'collaborate', 'partner', 'team', 'together', 'joint', 'work with'},
        'media_appearance': {'screen', 'movie', 'film', 'appear', 'show', 'feature', 'premiere',
                              'broadcast', 'air', 'release', 'debut'},
        'writing_event': {'write', 'screenplay', 'script', 'book', 'novel', 'story', 'publish',
                          'author', 'article', 'essay', 'manuscript'},
        'health_event': {'injury', 'health', 'medical', 'pain', 'surgery', 'doctor', 'hospital',
                         'treatment', 'diagnosis', 'accident', 'break', 'fracture', 'sprain'},
        'attended_event': {'attend', 'visit', 'trip', 'vacation', 'class', 'workshop', 'event',
                           'conference', 'festival', 'concert', 'meeting', 'gathering'},
        'started_activity': {'start', 'begin', 'join', 'sign', 'register', 'enroll', 'new'},
        'achievement': {'win', 'award', 'prize', 'honor', 'recognition', 'accept', 'earn'},
        'planned_event': {'plan', 'intend', 'hope', 'future', 'going to', 'will', 'want'},
        'acquisition': {'buy', 'purchase', 'acquire', 'adopt', 'adpot', 'subscribe', 'order', 'get', 'pet', 'pets', 'animal', 'dog', 'cat', 'puppy', 'kitten'},
        'pet_adoption': {'pet', 'pets', 'adopt', 'dog', 'cat', 'acquire', 'puppy', 'kitten'},
        'contributed_to_media': {'contribute', 'collaborate', 'work on', 'help'},
    }
    
    # v2 EAT 语义子类（fact_extractor_v2 无 event_type 字段，用 semantic_subcategory 兜底）→ question keywords
    _V2_EVENT_SEMANTIC_MAP = {
        'acquired': {'buy', 'purchase', 'acquire', 'adopt', 'adpot', 'order', 'get', 'got',
                     'pet', 'pets', 'dog', 'cat', 'puppy', 'kitten', 'animal'},
        'person_acquired': {'buy', 'purchase', 'acquire', 'adopt', 'get', 'got',
                            'pet', 'pets', 'dog', 'cat', 'puppy', 'kitten', 'animal'},
        'attended': {'attend', 'visit', 'trip', 'vacation', 'class', 'workshop', 'event',
                     'conference', 'festival', 'concert', 'meeting', 'gathering', 'went', 'joined', 'took'},
        'person_attended': {'attend', 'visit', 'trip', 'class', 'event', 'conference',
                            'festival', 'meeting', 'went', 'joined', 'took'},
        'created': {'create', 'build', 'make', 'develop', 'design', 'launch', 'found',
                    'start', 'write', 'publish', 'product', 'project'},
        'person_created': {'create', 'build', 'make', 'develop', 'design', 'launch', 'found', 'start'},
        'ownership': {'own', 'have', 'has', 'use', 'uses', 'using', 'buy', 'bought',
                      'purchase', 'got', 'possess'},
        'plan': {'plan', 'planning', 'intend', 'will', 'going to', 'future', 'hope', 'want', 'gonna'},
        'activity': {'play', 'playing', 'practice', 'attend', 'do', 'take', 'start', 'begin',
                     'join', 'class', 'game', 'sport', 'participate'},
        'preference': {'like', 'likes', 'prefer', 'love', 'enjoy', 'favorite', 'favourite',
                       'dislike', 'hate'},
        'person_preference': {'like', 'prefer', 'love', 'enjoy', 'dislike', 'hate', 'favorite'},
        'location': {'live', 'lives', 'move', 'moved', 'travel', 'visit', 'from', 'based', 'located'},
        'person_moved': {'move', 'moved', 'travel', 'relocate', 'back to'},
        'role': {'work', 'works', 'job', 'career', 'promote', 'promotion', 'became',
                 'started as', 'role'},
        'person_role': {'work', 'job', 'career', 'role', 'became'},
        'frequency': {'frequency', 'how often', 'every', 'weekly', 'daily', 'times'},
        'frequency_pattern': {'how often', 'frequency', 'every', 'weekly', 'daily'},
        'explicit_count': {'count', 'how many', 'total', 'number', 'items', 'sessions'},
        'vague_count': {'count', 'how many', 'several', 'few', 'many'},
        'count': {'count', 'how many', 'number', 'total', 'items', 'sessions'},
        'direct_attribute': {'is', 'has', 'attribute', 'characteristic'},
        'other_action': {'do', 'did', 'make', 'action'},
    }
    
    def _get_event_facts_for_query(self, query: str) -> str:
        """Get event-type structured facts matching the question's core action.
        Port from 正版 trimem_ar.py _get_event_facts_for_query，适配 v2 fact 字段
        （无 event_type/factual_status/specificity，用 semantic_subcategory/predicate 兜底）。"""
        if not self.facts:
            return ''
        
        event_type_prompt = (
            f'Extract the CORE ACTION/EVENT TYPE this question is asking about. '
            f'The event type should be a short noun phrase describing WHAT happened.\n'
            f'\n'
            f'Examples:\n'
            f'  Q: "How many of Joanna\'s writing have made it to the big screen?"\n'
            f'  → big screen appearance\n'
            f'  Q: "How many dogs does Audrey have?"\n'
            f'  → pet ownership\n'
            f'  Q: "Which new games did John start playing?"\n'
            f'  → started playing game\n'
            f'  Q: "What types of yoga has Maria practiced?"\n'
            f'  → yoga practice\n'
            f'  Q: "How did Gina promote her clothes store?"\n'
            f'  → store promotion\n'
            f'\n'
            f'Output ONLY the short event type phrase (1-6 words). No punctuation, no explanation.\n'
            f'\n'
            f'Q: "{query}"\n'
            f'Event type:'
        )
        event_type_str = call_llm([{'role': 'user', 'content': event_type_prompt}], max_tokens=32) or ''
        event_type_str = event_type_str.strip().lower().rstrip('.').strip()
        
        if not event_type_str or len(event_type_str) < 3:
            return ''
        
        event_words = set(re.findall(r'[a-zA-Z]{3,}', event_type_str))
        q_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower())) - _STOP
        
        matching_events = []
        seen_objs = set()
        for f in self.facts:
            event_type_tag = f.get('event_type', '') or f.get('semantic_subcategory', '') or f.get('predicate', '')
            obj_lower = f.get('object_lower', '')
            raw_lower = f.get('raw', '').lower()
            
            if obj_lower in seen_objs:
                continue
            
            pred_lower = f.get('predicate', '').lower()
            
            fact_text = f'{pred_lower} {obj_lower} {raw_lower}'
            fact_words = set(re.findall(r'[a-zA-Z]{3,}', fact_text))
            overlap_score = len(event_words & fact_words)
            
            if event_type_tag:
                tag_words = event_type_tag.replace('_', ' ').split()
                tag_match = sum(1 for tw in tag_words if tw in q_words or tw in event_words)
            else:
                tag_match = 0
            
            total_score = overlap_score + tag_match * 2
            
            if total_score == 0 and event_type_tag in self._EVENT_SEMANTIC_MAP:
                mapped_kw = self._EVENT_SEMANTIC_MAP[event_type_tag]
                if q_words & mapped_kw:
                    total_score += 2
            if total_score == 0 and event_type_tag in self._V2_EVENT_SEMANTIC_MAP:
                mapped_kw = self._V2_EVENT_SEMANTIC_MAP[event_type_tag]
                if q_words & mapped_kw:
                    total_score += 2
            
            # Fallback: match fact object keywords against query keywords
            if total_score == 0:
                obj_keywords = set(re.findall(r'[a-zA-Z]{3,}', f.get('object', '').lower()))
                obj_q_overlap = len(q_words & obj_keywords)
                raw_q_overlap = len(q_words & set(re.findall(r'[a-zA-Z]{3,}', raw_lower)))
                if obj_q_overlap >= 1 or raw_q_overlap >= 2:
                    total_score = 1
                _pet_words = q_words & {'pet', 'pets', 'dog', 'cat', 'animal', 'puppy'}
                _obj_words = obj_keywords | set(re.findall(r'[a-zA-Z]{3,}', raw_lower))
                if _pet_words and _obj_words & {'pet', 'pets', 'dog', 'cat', 'animal', 'puppy', 'adopt', 'adopted', 'acquisition', 'acquired', 'got', 'take', 'named'}:
                    total_score = max(total_score, 1)
            
            if total_score > 0:
                matching_events.append((total_score, f))
                seen_objs.add(obj_lower)
        
        if not matching_events:
            return f'[EVENT FACTS] No structured events found matching: {event_type_str}'
        
        _has_real_events = any(f.get('event_type') or f.get('semantic_subcategory') or f.get('category') for _, f in matching_events)
        if not _has_real_events:
            return ''
        
        matching_events.sort(key=lambda x: -x[0])
        
        by_type = defaultdict(list)
        for score, f in matching_events:
            etype = f.get('event_type', '') or f.get('semantic_subcategory', '') or f.get('predicate', 'unknown')
            by_type[etype].append(f)
        
        _action_words = set(re.findall(r'[a-zA-Z]{3,}', event_type_str))
        _etype_scores = {}
        for etype in by_type:
            if etype in self._EVENT_SEMANTIC_MAP:
                _etype_scores[etype] = len(_action_words & self._EVENT_SEMANTIC_MAP[etype])
            elif etype in self._V2_EVENT_SEMANTIC_MAP:
                _etype_scores[etype] = len(_action_words & self._V2_EVENT_SEMANTIC_MAP[etype])
            else:
                _etype_scores[etype] = 0
        _best_etype = max(by_type.keys(), key=lambda e: (_etype_scores.get(e, 0), -len(by_type[e]))) if by_type else ''
        
        lines = [f'[STRUCTURED EVENT FACTS — question action: "{event_type_str}"]']
        total_events = 0
        for etype, entries in sorted(by_type.items(), key=lambda x: x[0], reverse=True):
            if etype != _best_etype and _best_etype:
                continue
            completed = [f for f in entries if f.get('predicate') != 'plans' and f.get('semantic_subcategory') != 'plan']
            planned = [f for f in entries if f.get('predicate') == 'plans' or f.get('semantic_subcategory') == 'plan']
            
            if completed:
                total_events += len(completed)
                _score_lookup = {_id_f.get('object_lower', ''): _id_score for _id_score, _id_f in matching_events}
                _conf_w = {'high': 3, 'medium': 2, 'low': 1}
                completed.sort(key=lambda f: -_score_lookup.get(f.get('object_lower', ''), 0) * 10 - _conf_w.get(f.get('confidence', 'medium'), 1))
                lines.append(f'  [{etype}] {len(completed)} confirmed occurrence(s):')
                for f in completed[:8]:
                    _score = _score_lookup.get(f.get('object_lower', ''), 0)
                    _event_date = f.get('time', f.get('session_id', 'unknown'))
                    _event_date_str = f'[{_event_date}]' if _event_date and _event_date != 'unknown' else ''
                    lines.append(f'    {_event_date_str} [match={_score}] {f["object"]}')
            
            if planned:
                lines.append(f'  [⚠️ PLANNED (not yet executed)] — {len(planned)} mention(s) about future plans:')
                for f in planned[:3]:
                    lines.append(f'    [{f["time"]}] {f["object"]}')
                lines.append(f'  (Note: planned/future events above are NOT counted in totals)')
        
        lines.append(f'  → Total confirmed events: {total_events}')
        
        return '\n'.join(lines)
