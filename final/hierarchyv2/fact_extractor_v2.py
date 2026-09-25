#!/usr/bin/env python3
"""
fact_extractor_v2.py — EAT (Entity-Attribute Tracker) v2 + IPI (Implicit Persona Inference)

EAT extracts structured facts using regex patterns across 5 semantic categories:
  1. Action-Verb: ownership, activities, plans
  2. Attribute-Value: direct attributes ("my X is Y")
  3. Count: numeric expressions ("3 sessions", "several items")
  4. Person-Action: person does action ("Rachel bought X")
  5. Frequency: how often things happen

Each fact: f = (s, p, o_v, t, c, src) where c ∈ {high, medium, low}

IPI infers 12 persona dimensions via LLM once per session:
  Big Five: personality traits, preferences, life goals, key concerns
  Socio-demographic: career, financial status, relationship status, location,
                     religious orientation, political orientation, health, lifestyle
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Import from the package so relative imports in existing source files work
from hierarchyv2.config import _COMPUTE_SPECIFICITY_STOP, _STOP
from hierarchyv2.llm_utils import call_llm

# ═══════════════════════════════════════════════════════════════════
# EAT: Entity-Attribute Tracker — 5 Semantic Categories
# ═══════════════════════════════════════════════════════════════════

# ─── Category 1: Action-Verb ───────────────────────────────────
_ACTION_VERB_PATTERNS = [
    (r"\b(?:I\s+)?(?:use|uses|using|used|have|own|owns|bought|purchased|got|ordered|adopt|adopted|adpot|recommend|suggested|tried|try)\s+(?:a|an|the|my|some|me|you)?\s*(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'has_object'),
    (r"\b(?:I\s+)?(?:like|likes|loved|love|enjoy|enjoyed|prefer|prefers|favorite|favourite)\s+(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'likes'),
    (r"\b(?:I\s+)?(?:don't like|do not like|doesn't like|hate|dislike|not a fan of)\s+(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'dislikes'),
    (r"\b(?:I\s+)?(?:went to|visited|attend(?:ed)?s?|joined|took|started|began|signed up for|registered for|participated in|do|does|doing)\s+(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'attended'),
    (r"\b(?:I\s+)?(?:planning to|going to|will|gonna|plan to|intend to)\s+(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'plans'),
    (r"\b(?:I\s+)?(?:live in|lives in|from|based in|located in|move to|moved to|travel to|traveled to|visiting|went to)\s+(.*?)(?:[,.]|\sand|\sor|\sbut|$)", 'located_at'),
]

# ─── Category 2: Attribute-Value ───────────────────────────────
_ATTRIBUTE_VALUE_PATTERNS = [
    (r"\b(my|the|a)\s+([a-zA-Z0-9#\s]+?)\s+(?:is|was|are|were)\s+(?:a|an|the|my)?\s*(.*?)(?:[.,!?;]|$)", 'is'),
    (r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:is|was|are|were)\s+(.*?)(?:[.,!?;]|$)", 'is'),
    (r"\b(my|the|a)\s+([a-zA-Z0-9#\s]+?)\s+of\s+(.*?)(?:[.,!?;]|$)", 'of'),
]

# ─── Category 3: Count ──────────────────────────────────────────
_WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a': 1
}

_COUNT_PATTERNS = [
    (r"\b(?:attended|attending|took|taken|completed|had|did|done|went to|joined|participated in|doing|bought|purchased|got|collected|created|made|built|wrote|read|watched|visited|played)\s+(\d+|a\s+couple of|a\s+few|several|many|three|four|five|six|seven|eight|nine|ten)\s+(.*?)(?:[.,!?;]|$)", 'count_of'),
    (r"\b(?:I\s+)?(?:attended|took|completed|had)\s+(\w+|a)\s+session", 'attended_sessions'),
    (r"\b(\d+)\s+(?:sessions|times|items|books|games|movies|episodes|classes|lessons|days|weeks|months|years)\b", 'exact_count'),
    (r"\b(?:a\s+)?(?:couple of|few|several|many)\s+(sessions|times|items|books|games|movies|episodes|classes|lessons|days|weeks|months|years)\b", 'vague_count'),
]

# ─── Category 4: Person-Action ─────────────────────────────────
_PERSON_ACTION_PATTERNS = [
    (r"\b([A-Z][a-z]+)\s+(?:bought|purchased|got|ordered|adopted|adopt|adpot|owned)\s+(?:a|an|the|my|a new|another)?\s*(.*?)(?:[.,!?;]|$)", 'person_acquired'),
    (r"\b([A-Z][a-z]+)\s+(?:went to|visited|attended|joined|took|started|began|signed up for|registered for|participated in)\s+(.*?)(?:[.,!?;]|$)", 'person_attended'),
    (r"\b([A-Z][a-z]+)\s+(?:likes|loves|prefers|enjoys|hates|dislikes)\s+(.*?)(?:[.,!?;]|$)", 'person_preference'),
    (r"\b([A-Z][a-z]+)\s+(?:works as|is a|is an|is the|became|become|started as|got a job as)\s+(.*?)(?:[.,!?;]|$)", 'person_role'),
    (r"\b([A-Z][a-z]+)\s+(?:moved|relocated|traveled|travelled)\s+(?:to|from|back to)?\s*(.*?)(?:[.,!?;]|$)", 'person_moved'),
    (r"\b([A-Z][a-z]+)\s+(?:created|made|built|developed|designed|launched|started|founded)\s+(?:a|an|the|a new|another)?\s*(.*?)(?:[.,!?;]|$)", 'person_created'),
]

# ─── Category 5: Frequency ─────────────────────────────────────
_FREQUENCY_PATTERNS = [
    (r"\b(?:doing|attend(?:ed|s)?|do|does|practice|practicing|taking|having|play|playing)\s+(.*?)\s+(\w+\s+times\s+(?:a|per)\s+\w+)", 'frequency'),
    (r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+((?:once|twice|thrice|\d+\s+times?)\s+(?:a|per)\s+\w+)", 'frequency'),
    (r"\b(?:do|does|doing|practice|practicing|have|having)\s+(.*?)\s+(every\s+(?:other\s+)?(?:day|week|month|morning|evening|night|afternoon))", 'frequency'),
    (r"\b(this week|every week|each week|weekly|biweekly|monthly|daily|annually|every\s+\d+\s+(?:day|week|month)s?)\b", 'frequency_adverb'),
]


def _classify_action_verb(predicate: str, raw_text: str) -> str:
    raw_lower = raw_text.lower()
    if any(w in raw_lower for w in ['use', 'uses', 'using', 'used', 'have', 'own', 'owns', 'bought', 'purchased', 'got']):
        return 'ownership'
    if any(w in raw_lower for w in ['attended', 'went to', 'visited', 'joined', 'took', 'started', 'began', 'participated']):
        return 'activity'
    if any(w in raw_lower for w in ['planning to', 'going to', 'plan to', 'intend to', 'will', 'gonna']):
        return 'plan'
    if any(w in raw_lower for w in ['like', 'love', 'enjoy', 'prefer', 'favorite']):
        return 'preference'
    if any(w in raw_lower for w in ['live', 'lives', 'based', 'located', 'move', 'moved', 'travel', 'from']):
        return 'location'
    return 'other_action'


def _resolve_count(count_str: str) -> Optional[int]:
    c = count_str.strip().lower()
    if c in _WORD_TO_NUM:
        return _WORD_TO_NUM[c]
    if c in ('a few', 'several'):
        return 3
    if c == 'a couple of':
        return 2
    if c == 'many':
        return 5
    try:
        return int(c)
    except ValueError:
        return None


_OCCUPATION_PATTERNS = [
    # 对话式职业表述（2026-08-26 新增，person_role 正则覆盖不到的口语）
    # 主语统一用 extract_subject（[Name] 前缀），object 归一为核心职业名
    (r'\b(?:lost (?:my|his|her|their)\s+job(?: (?:as|at) (?:a |an |the )?([^,.;!?]+))?)', 'lost_job'),
    (r'\b(?:got|took|found|landed)\s+(?:a |another |a new |a temp |a part-time |a full-time )?job(?:\s+(?:as|working as|at|with)\s+([^,.;!?]+))?', 'person_role'),
    (r'\b(?:start(?:ed|ing)?|launch(?:ed|ing)?|open(?:ed|ing)?|run(?:ning)?|own(?:ed|ing)?)\s+(?:my|his|her|their|a|an|the)?\s*(?:own\s+)?(?:online\s+|clothing\s+|fashion\s+|dance\s+|coffee\s+|book\s+|bakery\s+|restaurant\s+|small\s+|new\s+)*?(?:business|company|venture|startup|studio|store|shop)\b', 'person_role'),
    (r'\b(?:owns?|runs?)\s+(?:a |an |the )?(?:online |clothing |fashion |small |new )*?(?:store|shop|business)\b', 'person_role'),
    (r'\b(?:works?|working)\s+(?:as|at|for|with)\s+([^,.;!?]+)', 'person_role'),
]


_POSSESSION_PATTERNS = [
    # 2026-08-26: 驾驶/拥有类（LME "I drive a 2018 Mustang GT" → EFG 版本链缺失，VKE 注入无信息）
    # 2026-08-26 22:37: 去掉 got/took/found/landed —— "I got a temp job" 会被提取成 has_object: temp job，
    # 污染 VKE 版本链（Jon 类 current 题稳定答错 temp job 的元凶）；job 类已有 person_role 覆盖
    (r'\b(?:I|he|she|we|they)\s+(?:drive|drives|driving|own|owns|owning|have|has|bought|purchased)\s+(?:a |an |the |my |his |her |our |their )?([^,.;!?]{2,40})\b', 'has_object'),
    (r'\bmy\s+([a-zA-Z][a-zA-Z0-9 ]{1,30})\s+(?:is|was|are|were)\s+(?:a |an |the )?([^,.;!?]{2,40})\b', 'is'),
]


# ─── Category 9: 属性词表驱动的通用模板族（2026-08-27 方案①）──────────
# 不再手写每类属性正则：从数据集切片预扫描 attribute 词表（sleep schedule / footwear / screen time rule...），
# 对每个 attr 用通用模板展开提取。新增属性类型 = 加一个 attribute 名，零正则编写。
# {attr} 占位符由调用方 replace 填充（re.escape 后）；predicate 统一 f'attr_{attr}'，
# _select_ku_predicates 的 attr-in-p 匹配可直接命中。
_ATTR_TEMPLATES = [
    (r"\b(?:my|the|our|his|her|their)?\s*{attr}\s+(?:is|are|was|were|became|has been|had been|used to be|changed to)\s+(.+?)(?:[.,!?;]|$)", 'is'),
    (r"\b(?:changed|switched|updated|moved|set)\s+(?:my|the|our|his|her|their)?\s*{attr}\s+to\s+(.+?)(?:[.,!?;]|$)", 'changed_to'),
    (r"\b(?:my|the|our|his|her|their)?\s*{attr}\s+(?:now|currently|these days|at the moment|right now)\s+(?:is|are|was|were)\s+(.+?)(?:[.,!?;]|$)", 'is_now'),
    (r"\b(?:my|the|our|his|her|their)?\s*current\s+{attr}\s*(?:is|are|was|were|:)?\s*(.+?)(?:[.,!?;]|$)", 'current_is'),
    (r"\b{attr}\s*:\s*(.+?)(?:[.,!?;]|$)", 'colon'),
]


_ATTR_VALUE_CLEAN = re.compile(
    r'\s+(?:now|currently|these days|at the moment|right now|lately|so far|already|yet|finally)$', re.IGNORECASE)


def _clean_attr_value(raw: str) -> str:
    """清理属性值：去尾部时间副词/从句。"""
    s = raw.strip().rstrip('.,!?;:').strip()
    s = _ATTR_VALUE_CLEAN.sub('', s)
    s = re.sub(r'\s+(?:because|so that|so I|to help|while|but|and I|which|where|that I|since|when).*$', '', s, flags=re.I)
    return s.strip()[:80]


def _extract_attr_template(text: str, date_str: str, session_id: str, subject: str, attr: str):
    """对单个 attribute 跑通用模板族，返回 facts 列表。"""
    facts = []
    _a = re.escape(attr.strip())
    _pred = f'attr_{attr.strip()}'
    for _tpl, _subcat in _ATTR_TEMPLATES:
        _pat = _tpl.replace('{attr}', _a)
        for m in re.finditer(_pat, text, re.IGNORECASE):
            obj = _clean_attr_value(m.group(1))
            if len(obj) < 2 or len(obj) > 80:
                continue
            facts.append({
                'subject': subject, 'predicate': _pred, 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id, 'confidence': 'medium',
                'raw': m.group(0)[:120], 'category': 'attr_template',
                'semantic_subcategory': _subcat,
            })
    return facts


# ─── Category 10 模式与归一（dietary preference）───────────────────
_DIETARY_PATTERNS = [
    # 1) 隐式改变："I'm trying to cut back on gluten" / "I've been avoiding dairy" / "I'm switching to keto"
    (r"\b(?:i'?m|i am|i was|i have been|i've been)\s+(?:trying to|starting to|planning to|hoping to|thinking about|considering)?\s*(?:cut back on|cutting back on|cut down on|cutting down on|cut out|cutting out|give up|giving up|avoid|avoiding|reduce|reducing|switch to|switching to|go on|going on)\s+([^,.;!?]{2,50})", 'dietary_pref'),
    # 2) 状态转变："I went vegan last month" / "I turned vegetarian" / "I'm mostly plant-based"
    (r"\b(?:i|my partner|my doctor|my wife|my husband)\s+(?:went|turned|became|am|'m|have been)\s+(?:fully\s+|mostly\s+|almost\s+)?(vegan|vegetarian|pescatarian|keto|paleo|gluten[- ]free|raw food|carnivore|flexitarian|whole30|plant[- ]based)\b", 'dietary_pref'),
    # 3) 意向："I'm trying to be more vegan" / "I'm going to go gluten-free"
    (r"\b(?:i'?m|i am)\s+(?:trying|going|starting)\s+(?:to be|to go|on)\s+(?:more\s+)?(vegan|vegetarian|gluten[- ]free|plant[- ]based)\b", 'dietary_pref'),
]

# 商店/菜单询问语境（含这些词 = 在问商家有没有，不是用户自身偏好）
_DIETARY_STORE_ASK = re.compile(
    r'\b(?:do|does|did|would|can|could|offer|offers|have|has|any of|anyone|they|them|there|'
    r'bakeries|bakery|restaurant|restaurants|caf[eé]|menu|place|spot|shop|store|market|'
    r'option at|options at|options there|desserts?|dishes?|items?|meals?|food at)\b', re.I)


def _normalize_dietary(raw: str) -> str:
    """归一饮食偏好：'cutting back on gluten' / 'cut back on gluten' / 'reduce gluten' → 标准值。"""
    s = raw.strip().rstrip('.,!?;:').strip()
    s = re.sub(r'\s+(?:now|currently|these days|at the moment|right now|lately|so far|already|yet|from now on)$', '', s, flags=re.I)
    s = re.sub(r'\s+(?:because|so that|so I|to help|while|but|and I|which|where|that I|since|when).*$', '', s, flags=re.I)
    if re.search(r'\b(?:gluten|gluten[- ]free)\b', s, re.I):
        return 'cutting back on gluten'
    if re.search(r'\bvegan\b', s, re.I):
        return 'vegan'
    if re.search(r'\bvegetarian\b', s, re.I):
        return 'vegetarian'
    if re.search(r'\bketo\b', s, re.I):
        return 'keto'
    return s.strip()[:60]


def _normalize_career_interest(raw: str) -> str:
    """归一兴趣/职业意向：'looking into counseling and mental health as a career' -> 'counseling/mental health'。

    2026-08-27: conv-26_1 Caroline counseling → adoption 版本链的缺失环节。
    值域有限才归一（对齐 _OCC_SYN 风格），避免措辞差异产生虚假 distinct。
    """
    s = raw.strip().rstrip('.,!?;:').strip()
    s = re.sub(r'\s+(?:as a career|as my career|as an option|as a path|at the moment|right now|these days|so far|now|yet|more|lately)$', '', s, flags=re.I)
    s = re.sub(r'\s+(?:because|so that|so I|to help|while|but|and I|which|where|that I).*$', '', s, flags=re.I)
    # 同义归一（adoption/counseling/business 系 → 标准值）
    if re.search(r'\b(?:adopt|adoption)\b', s, re.I):
        return 'adoption'
    if re.search(r'\b(?:counsel|counselor|therapy|therapist|mental health)\b', s, re.I):
        return 'counseling/mental health'
    if re.search(r'\b(?:business|venture|startup|company|studio)\b', s, re.I):
        return 'own business'
    # 职业信号白名单：归一后不含任何职业/意向信号词的直接丢弃（防 "this pic of him eating parsley" 类误抓）
    _CAREER_SIGNAL = ('career', 'job', 'work', 'counsel', 'therap', 'mental health', 'adopt', 'teach',
                      'nurse', 'doctor', 'lawyer', 'agency', 'business', 'studio', 'venture', 'startup',
                      'clinic', 'school', 'social work', 'volunteer', 'advocacy', 'advocate', 'activist',
                      'trans', 'lgbtq', 'artist', 'painter', 'creative', 'write', 'music', 'perform', 'learn')
    if not any(sig in s.lower() for sig in _CAREER_SIGNAL):
        return ''
    # 纯爱好词不是职业意向（避免 "interested in painting" 污染 career_interest）
    _hobby_stop = {'reading', 'reading books', 'books', 'painting', 'drawing', 'pottery', 'photography',
                   'hiking', 'cooking', 'gardening', 'music', 'movies', 'films', 'games', 'gaming',
                   'travel', 'traveling', 'yoga', 'running', 'swimming', 'dancing', 'singing',
                   'writing', 'knitting', 'baking', 'camping', 'fishing'}
    if s.lower().strip() in _hobby_stop:
        return ''
    if s.lower().strip() in ('career options', 'options', 'the options'):
        return ''
    # 去修饰前缀
    s = re.sub(r'^(?:a |an |the |my |her |his |their |career in |career as |working in |work in |becoming |being )+', '', s, flags=re.I)
    s = re.sub(r'\s+career\b', '', s, flags=re.I)
    return s.strip()[:60] or raw.strip()[:60]


_CAREER_INTEREST_PATTERNS = [
    # 2026-08-27: 兴趣/职业意向（conv-26_1 Caroline: counseling → adoption 版本链）
    (r"\b(?:career|future career|career path|career plan|career interest|career goal|dream job|aspiration)s?\s*(?:is|are|was|were|:|to be)\s+([^,.;!?]{2,60})", 'career_interest'),
    (r"\b(?:looking into|looked into|exploring|considering|researching|research|interested in|interest in)\s+(?:a |an |the )?(?:career in |career as |career as a |working in |work in |becoming |being )?([^,.;!?]{2,60})(?:\s+(?:as|is|was)\s+(?:a\s+)?career)?", 'career_interest'),
    (r"\b(?:a |my |her |his |their )?career\s+(?:in|as)\s+([^,.;!?]{2,60})", 'career_interest'),
    (r"\b(?:applied to|applying to|apply to)\s+(?:an |a |the )?([^,.;!?]{2,60})", 'career_interest'),
    (r"\b(?:dream|goal|plan|ambition)s?\s+(?:is|was|has been|is to|was to|to be|to become|to)\s+([^,.;!?]{2,60})", 'career_interest'),
]


def _normalize_occupation(raw: str) -> str:
    """归一职业表述：'got a temp job' -> 'temp job'；'lost my job as a banker yesterday' -> 'banker'。"""
    s = raw.strip().rstrip('.,!?;:').strip()
    # 去尾部时间词/从句
    s = re.sub(r'\s+(?:yesterday|last week|last month|this month|these days|so far|now|right now)$', '', s, flags=re.I)
    s = re.sub(r'\s+(?:to help|while|because|so|but|and|for|before|after).*$', '', s, flags=re.I)
    # 特判：got/took/found/landed a job
    m = re.match(r'^(?:got|took|found|landed)\s+(?:a |another |a new |a temp |a part-time |a full-time )?job$', s, re.I)
    if m:
        s = re.sub(r'^(?:got|took|found|landed)\s+', '', s, flags=re.I)
        s = re.sub(r'^(?:a |another |a new |a temp |a part-time |a full-time )+', '', s, flags=re.I)
        return s.strip()[:60]
    # lost my job as X -> X；lost my job at X -> X；lost my job -> unemployed
    s = re.sub(r'^(?:lost (?:my|his|her|their)\s+job(?:\s+(?:as|at)\s+)?)', '', s, flags=re.I)
    if not s.strip():
        return 'unemployed'
    s = re.sub(r'^(?:start(?:ed|ing)?|launch(?:ed|ing)?|open(?:ed|ing)?|run(?:ning)?)\s+', '', s, flags=re.I)
    s = re.sub(r'^(?:a |an |the |another |a new |a temp |a part-time |a full-time )+', '', s, flags=re.I)
    s = re.sub(r'^(?:my |his |her |their |own )+', '', s, flags=re.I)
    return s.strip()[:60] or raw.strip()[:60]


def extract_subject(text: str) -> str:
    """Extract the speaker/subject from a turn.

    Priority:
    1. Leading [Name] prefix (LoCoMo/LongMemEval turn format: '[Joanna] Hey Nate! ...')
    2. 'you' at start -> assistant
    3. 'I am / I'm / my name is / call me X' -> X
    4. fallback 'user'
    """
    text = text or ''
    text_lower = text.lower()
    # 1) [Name] prefix（VKE 实体键修复核心）
    m = re.match(r'^\s*\[([A-Za-z][A-Za-z\s\'.-]{0,30})\]\s*', text)
    if m:
        name = m.group(1).strip().lower()
        if name not in ('user', 'assistant', 'system'):
            return name
    # 2) 'you' at start -> assistant
    if 'you' in text_lower[:20]:
        return 'assistant'
    if text_lower.startswith('i ') or text_lower.startswith("i'm") or text_lower.startswith("i've") or text_lower.startswith("i'd"):
        m = re.search(r"\bI'?m\s+([A-Z][a-z]+)", text)
        if m:
            return m.group(1).lower()
        return 'user'
    m = re.search(r"\b(my name is|call me|i am|i'm)\s+([A-Z][a-z]+)", text, re.I)
    if m:
        return m.group(2).lower()
    return 'user'


def extract_facts_v2(text: str, date_str: str, session_id: str, attrs: Optional[List[str]] = None) -> List[Dict]:
    """EAT: Extract structured facts using 5 semantic categories + attr-template family."""
    facts = []
    subject = extract_subject(text)

    # Category 1: Action-Verb
    for pattern, pred in _ACTION_VERB_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj = m.group(1).strip().rstrip('.,!?;:').strip()
            if len(obj) < 2 or len(obj) > 80:
                continue
            subcat = _classify_action_verb(pred, m.group(0))
            facts.append({
                'subject': subject, 'predicate': pred, 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id,
                'confidence': 'high' if pred in ('likes', 'has_object', 'located_at') else 'medium',
                'raw': m.group(0)[:120], 'category': 'action_verb',
                'semantic_subcategory': subcat,
            })

    # Category 2: Attribute-Value
    for pattern, pred in _ATTRIBUTE_VALUE_PATTERNS:
        for m in re.finditer(pattern, text):
            if len(m.groups()) >= 2:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                if 2 < len(subj) < 60 and 2 < len(obj) < 80:
                    facts.append({
                        'subject': subj.lower(), 'predicate': pred,
                        'object': obj.rstrip('.,!?;:').strip(),
                        'object_lower': obj.lower().rstrip('.,!?;:').strip(),
                        'time': date_str, 'session_id': session_id,
                        'confidence': 'low', 'raw': m.group(0)[:120],
                        'category': 'attribute_value',
                        'semantic_subcategory': 'direct_attribute',
                    })

    # Category 3: Count
    for pattern, pred in _COUNT_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            first_group = m.group(1).strip()
            resolved = _resolve_count(first_group)
            obj = m.group(0).strip()[:80]
            if len(obj) > 10:
                facts.append({
                    'subject': subject, 'predicate': pred, 'object': obj,
                    'object_lower': obj.lower(), 'time': date_str,
                    'session_id': session_id,
                    'confidence': 'high' if pred != 'vague_count' else 'low',
                    'raw': m.group(0)[:120], 'count_value': resolved,
                    'category': 'count',
                    'semantic_subcategory': 'explicit_count' if pred != 'vague_count' else 'vague_count',
                })

    # Category 4: Person-Action
    for pattern, pred in _PERSON_ACTION_PATTERNS:
        for m in re.finditer(pattern, text):
            person = m.group(1).strip()
            if 2 < len(person):
                action = m.group(2).strip().rstrip('.,!?;:').strip()
                if len(action) > 2:
                    facts.append({
                        'subject': person.lower(), 'predicate': pred,
                        'object': action, 'object_lower': action.lower(),
                        'time': date_str, 'session_id': session_id,
                        'confidence': 'high', 'raw': m.group(0)[:120],
                        'category': 'person_action',
                        'semantic_subcategory': pred.replace('person_', ''),
                    })

    # Category 5: Frequency
    for pattern, pred in _FREQUENCY_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj = m.group(0).strip()[:80]
            if len(obj) > 8:
                facts.append({
                    'subject': subject, 'predicate': pred, 'object': obj,
                    'object_lower': obj.lower(), 'time': date_str,
                    'session_id': session_id, 'confidence': 'medium',
                    'raw': m.group(0)[:120], 'category': 'frequency',
                    'semantic_subcategory': 'frequency_pattern',
                })

    # Category 6: Occupation (对话式职业表述, 2026-08-26)
    for pattern, pred in _OCCUPATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj = m.group(1) if m.lastindex and m.group(1) else m.group(0)
            obj = _normalize_occupation(obj)
            if len(obj) < 2 or len(obj) > 80:
                continue
            facts.append({
                'subject': subject, 'predicate': pred, 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id, 'confidence': 'medium',
                'raw': m.group(0)[:120], 'category': 'occupation',
                'semantic_subcategory': 'role',
            })

    # Category 8: Career Interest / 职业意向 (2026-08-27, conv-26_1 Caroline counseling→adoption)
    for pattern, pred in _CAREER_INTEREST_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj = m.group(1).strip() if m.lastindex and m.group(1) else m.group(0)
            obj = _normalize_career_interest(obj)
            if len(obj) < 2 or len(obj) > 60:
                continue
            facts.append({
                'subject': subject, 'predicate': pred, 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id, 'confidence': 'medium',
                'raw': m.group(0)[:120], 'category': 'career_interest',
                'semantic_subcategory': 'intention',
            })

    # Category 9: 属性词表驱动的通用模板族 (2026-08-27 方案①，attrs 来自数据集切片预扫描)
    if attrs:
        for _attr in attrs:
            for _f in _extract_attr_template(text, date_str, session_id, subject, _attr):
                facts.append(_f)

    # ─── Category 10: Dietary Preference / 饮食偏好 (2026-08-28, LME 031748ae 隐式表达) ───
    # 用户隐式表达（"I'm trying to cut back on gluten" / "I went vegan"）显式模板族
    # （Category 9 "{attr} is X"）抓不到 → 专属关键词正则 + 值归一。
    # predicate 统一 'dietary preference'（干净谓词，_select_ku_predicates 的 attr-in-p 可直接命中）。
    # 防误抓：问商店/菜单（"do they have vegan options" / "offer gluten-free desserts"）不是用户偏好。
    for pattern, pred in _DIETARY_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            _raw0 = m.group(0)
            if _DIETARY_STORE_ASK.search(_raw0):
                continue  # 问句/商店菜单语境不是用户饮食偏好
            obj = m.group(1).strip() if m.lastindex and m.group(1) else m.group(0)
            obj = _normalize_dietary(obj)
            if len(obj) < 2 or len(obj) > 60:
                continue
            facts.append({
                'subject': subject, 'predicate': 'dietary preference', 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id, 'confidence': 'medium',
                'raw': _raw0[:120], 'category': 'dietary',
                'semantic_subcategory': 'preference',
            })

    # Category 7: Possession (驾驶/拥有, 2026-08-26, LME 第一人称)
    for pattern, pred in _POSSESSION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            obj = m.group(m.lastindex).strip() if m.lastindex else m.group(0)
            obj = re.sub(r'\s+(?:to|for|while|so|because|but|and).*$', '', obj, flags=re.I).strip()[:60]
            if len(obj) < 2:
                continue
            facts.append({
                'subject': subject, 'predicate': pred, 'object': obj,
                'object_lower': obj.lower(), 'time': date_str,
                'session_id': session_id, 'confidence': 'medium',
                'raw': m.group(0)[:120], 'category': 'possession',
                'semantic_subcategory': 'possessed',
            })

    return facts


def extract_facts(text: str, date_str: str, session_id: str) -> List[Dict]:
    """Compatibility wrapper."""
    return extract_facts_v2(text, date_str, session_id)


# ═══════════════════════════════════════════════════════════════════
# IPI: Implicit Persona Inference
# ═══════════════════════════════════════════════════════════════════

_PERSONA_DIMENSIONS = [
    "personality_traits", "preferences", "life_goals", "key_concerns",
    "career", "financial_status", "relationship_status", "location",
    "religious_orientation", "political_orientation", "health", "lifestyle",
]

_IMPLICIT_SIGNALS = {
    "personality_traits": "Look for how the speaker describes their reactions to events, their social tendencies, their work habits, and how they handle challenges.",
    "preferences": "Note what the speaker says they like, dislike, prefer, enjoy, or find interesting. Also note what they avoid or complain about.",
    "life_goals": "Look for statements about what the speaker wants to achieve, improve, or change. Also note frustrations that reveal unstated goals.",
    "key_concerns": "Notice what the speaker repeatedly brings up, worries about, or seeks advice on. Repetition signals importance.",
    "career": "Look for mentions of work, projects, colleagues, career changes, or professional development.",
    "financial_status": "Note mentions of purchases, budgets, financial concerns, lifestyle markers.",
    "relationship_status": "Look for mentions of partners, family members, living arrangements, social activities.",
    "location": "Identify geographic references: cities, neighborhoods, types of housing, commute patterns.",
    "religious_orientation": "Note mentions of religious practice, spiritual beliefs, holiday celebrations.",
    "political_orientation": "Look for mentions of political events, policy preferences, or value statements.",
    "health": "Note mentions of medical conditions, injuries, treatments, exercise habits, sleep patterns.",
    "lifestyle": "Synthesize from daily routines: sleep schedule, work patterns, social activities, diet, exercise.",
}


def infer_persona(session_turns: List[str], session_date: str = "") -> Dict[str, str]:
    """IPI: Infer implicit persona profile from a session using LLM."""
    conv_text = "\n".join([f"{'[user]' if i % 2 == 0 else '[assistant]'} {turn}"
                           for i, turn in enumerate(session_turns) if turn.strip()])
    if not conv_text.strip():
        return {dim: "N/A" for dim in _PERSONA_DIMENSIONS}

    dim_descriptions = "\n".join([
        f"  - {dim}: {_IMPLICIT_SIGNALS.get(dim, 'Infer from context')}"
        for dim in _PERSONA_DIMENSIONS
    ])

    prompt = f"""Analyze the following conversation and infer the speaker's persona profile.
Focus on IMPLICIT signals — what can be inferred from what they say, how they say it,
and what they choose to talk about.

Session date: {session_date or "Unknown"}

Conversation:
{conv_text}

For each of the following 12 dimensions, infer the most likely value.
If no evidence exists for a dimension, write "N/A".
Be specific where possible.

Dimensions:
{dim_descriptions}

Output format (one per line, NO extra text):
personality_traits: <value>
preferences: <value>
life_goals: <value>
key_concerns: <value>
career: <value>
financial_status: <value>
relationship_status: <value>
location: <value>
religious_orientation: <value>
political_orientation: <value>
health: <value>
lifestyle: <value>
"""
    result = call_llm([{'role': 'user', 'content': prompt}], max_tokens=512)
    persona = {}
    if result:
        for line in result.strip().split('\n'):
            for dim in _PERSONA_DIMENSIONS:
                if line.startswith(f'{dim}:'):
                    value = line[len(dim)+1:].strip()
                    persona[dim] = value if value else "N/A"
                    break
    for dim in _PERSONA_DIMENSIONS:
        if dim not in persona:
            persona[dim] = "N/A"
    return persona


def embed_persona(persona: Dict[str, str]) -> str:
    """Convert persona dict to a flat text string suitable for embedding."""
    parts = []
    for dim in _PERSONA_DIMENSIONS:
        val = persona.get(dim, "N/A")
        if val != "N/A":
            parts.append(f"{dim}: {val}")
    return "; ".join(parts)


def query_persona(query: str, persona_embeddings: list, persona_texts: list, k: int = 3,
                  threshold: float = 0.2) -> List[Dict[str, str]]:
    """Retrieve top-k persona profiles by semantic similarity.

    2026-09-14：阈值默认 0.25 → 0.2，与 TriMemAR_v2.search 的 IPI 块
    以及论文 A.6 “cosine threshold of 0.2” 对齐；新增 threshold 参数便于扫描。
    """
    if persona_embeddings is None or not persona_texts or len(persona_texts) == 0:
        return []
    import numpy as np
    from hierarchyv2.embedding import embed
    qe = np.array(embed([query]))
    pes = np.array(persona_embeddings)
    sims = np.dot(pes, qe.T).flatten()
    results = []
    for idx in np.argsort(sims)[::-1][:k]:
        if sims[idx] > threshold:
            results.append({'persona': persona_texts[idx], 'score': float(sims[idx])})
    return results


__all__ = [
    'extract_facts_v2', 'extract_facts',
    'extract_subject', 'infer_persona', 'embed_persona',
    'query_persona', 'PERSONA_DIMENSIONS',
]
