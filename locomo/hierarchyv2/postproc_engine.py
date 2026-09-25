#!/usr/bin/env python3
"""
postproc_engine.py — 声明式规则引擎

将 run_item.py 中硬编码的后处理补丁迁移为声明式规则。
Phase 2: FixCat2.5, 2.6, 2.10, 2.11
"""
import json, os, re
from datetime import datetime
from .config import KM_PATH


# ─── Inline P2 rules (phase 2) ──────────────────────────────────
# These are defined inline for now; future phases will load from KM JSON.
_P2_RULES = [
    {
        'id': 'FixCat2.11-protect-relative-time',
        'description': 'M6-Guard: protect relative time expressions from later overrides',
        'trigger': {
            'question_pattern': None,
            'answer_pattern': None,
            'answer_contains': None,  # match all temporal-reasoning answers; handler checks internally
            'question_types': ['temporal-reasoning'],
        },
        'action': {
            'type': 'protect_relative_time',
            'params': {},
        },
        'priority': 1,
    },
    {
        'id': 'FixCat2.5-year-extract',
        'description': 'Year-only extraction for "which year"/"what year" questions',
        'trigger': {
            'question_pattern': r'(?:which|what)\s+year\b',
            'answer_pattern': None,
            'answer_contains': None,
            'question_types': ['temporal-reasoning'],
        },
        'action': {
            'type': 'extract_year',
            'params': {
                'source': 'session_dates',
                'prefer': 'min',
            },
        },
        'priority': 5,
    },
    {
        'id': 'FixCat2.10-last-year',
        'description': 'Resolve "last year" text to actual year from session dates',
        'trigger': {
            'question_pattern': None,
            'answer_pattern': None,
            'answer_contains': 'last year',
            'question_types': ['temporal-reasoning'],
        },
        'action': {
            'type': 'year_offset',
            'params': {
                'offset': -1,
                'source': 'session_dates',
                'method': 'max_year',
            },
        },
        'priority': 7,
    },
    {
        'id': 'FixCat2.6-strip-prefix',
        'description': 'Strip verbose "Based on..." / "According to..." wrappers from answer',
        'trigger': {
            'question_pattern': None,
            'answer_pattern': (
                r'^(?:Based on the extracted timeline|'
                r'Based on the provided context|'
                r'Based solely on the provided context|'
                r'Based on the conversation|'
                r'Based on the context|'
                r'The context does not specify|'
                r'The context does not mention|'
                r'There is no information|'
                r'No information)[.,:]*\s*'
            ),
            'answer_contains': None,
            'question_types': ['temporal-reasoning'],
        },
        'action': {
            'type': 'strip_prefix',
            'params': {},
        },
        'priority': 9,
    },
]


class PostProcEngine:
    """声明式后处理规则引擎"""

    def __init__(self, rules=None):
        """
        Args:
            rules: Optional list of rule dicts. If None, uses inline P2 rules.
        """
        self.rules = rules if rules is not None else _P2_RULES
        # Sort once by priority (ascending: lower = higher priority)
        self.rules_sorted = sorted(self.rules, key=lambda r: r.get('priority', 100))
        # Action handler dispatch table
        self.handlers = {
            'protect_relative_time': self._action_protect_relative_time,
            'extract_year': self._action_extract_year,
            'year_offset': self._action_year_offset,
            'strip_prefix': self._action_strip_prefix,
        }

    def apply(self, answer, question, question_type, sessions, dates, ctx_str):
        """
        对 LLM 输出的 answer 应用所有匹配的后处理规则。

        Args:
            answer: str — LLM 当前输出的答案
            question: str — 原始问题
            question_type: str — 问题类型（如 temporal-reasoning）
            sessions: list — 会话列表
            dates: list — 会话日期列表
            ctx_str: str — 上下文文本

        Returns:
            str — 处理后的答案
        """
        context = {
            'answer': answer or '',
            'question': question or '',
            'question_type': question_type or '',
            'sessions': sessions or [],
            'dates': dates or [],
            'ctx_str': ctx_str or '',
            '_protected': False,  # M6-Guard flag
        }

        for rule in self.rules_sorted:
            if self._match(rule, context):
                context = self._execute(rule, context)
                if context.get('_halt', False):
                    break

        return context.get('answer', answer or '')

    def _match(self, rule, context):
        """检查规则是否匹配当前上下文"""
        trigger = rule.get('trigger', {})
        qt = context.get('question_type', '')
        q = context.get('question', '')
        answer = context.get('answer', '')

        # question_types 过滤
        types = trigger.get('question_types', None)
        if types and qt not in types:
            return False

        # question_pattern 正则匹配
        qpat = trigger.get('question_pattern', None)
        if qpat:
            if not re.search(qpat, q, re.I):
                return False

        # answer_contains 关键词检查
        ac = trigger.get('answer_contains', None)
        if ac is not None:
            if re.search(ac, answer, re.I):
                pass  # matched
            elif ac in answer.lower():
                pass  # simple substring match
            else:
                # Check if ac is a plain string (not regex)
                if ac.lower() not in answer.lower():
                    return False

        # answer_pattern 正则匹配
        apat = trigger.get('answer_pattern', None)
        if apat:
            if not re.match(apat, answer.strip(), re.I):
                return False

        return True

    def _execute(self, rule, context):
        """执行单条规则的动作"""
        action = rule.get('action', {})
        action_type = action.get('type', '')
        params = action.get('params', {})

        handler = self.handlers.get(action_type)
        if handler:
            result = handler(context, params)
            if result is not None:
                context = result
            rid = rule.get('id', '?')
            ans_preview = context.get('answer', '')[:60]
            print(f'    [PostProc] {rid}: {ans_preview}', flush=True)

        return context

    # ─── Action Handlers ─────────────────────────────────────────

    def _action_protect_relative_time(self, context, params):
        """FixCat2.11: M6-Guard — set protected flag if answer has relative time expression.
        Also handle trailing 'context' word cleanup (original FixCat2.11 behavior)."""
        ans = context.get('answer', '')

        # M6-Guard: set protected flag for later rules (FixCat2.1)
        rel_pat = re.compile(r'(?i)\b(?:week\s+before|weekend\s+before|last\s+week(?:end)?)\b')
        if rel_pat.search(ans):
            context['_protected'] = True
            print(f'    [PostProc] FixCat2.11 M6-Guard: protected relative time expression', flush=True)

        # Original FixCat2.11 cleanup: strip trailing "context" from short answers
        # Only when "context" is the last word (pattern: "YYYY-MM-DD context")
        if ans and len(ans) < 50:
            _trail = re.sub(r'\s+context\s*$', '', ans)
            if _trail and _trail != ans and len(_trail) > 3:
                # But don't strip if it would break FixCat2.6's prefix patterns
                # (e.g. "The context does not..." should preserve "context" for FixCat2.6)
                _prefix_matches_2_6 = re.search(
                    r'^(?:Based on|The context|There is|No information)',
                    ans, re.I
                )
                if not _prefix_matches_2_6:
                    context['answer'] = _trail

        return context

    def _action_extract_year(self, context, params):
        """FixCat2.5: For 'which year' / 'what year' questions,
        extract year from session dates or timeline context."""
        ans = context.get('answer', '')
        q = context.get('question', '')
        dates = context.get('dates', [])
        ctx_str = context.get('ctx_str', '')

        _years_in_ans = re.findall(r'\b(202\d|201\d)\b', str(ans))
        if not _years_in_ans:
            # LLM may have answered without a year — extract best year from dates
            if dates:
                _all_years = set(d[:4] for d in dates if d and len(d) >= 4)
                if _all_years:
                    _best_yr = min(_all_years) if params.get('prefer') == 'min' else max(_all_years)
                    context['answer'] = _best_yr
                    return context
            return context

        _candidate_years = set()

        # Scan session metadata for stronger confirmation
        if dates:
            _all_years = set(d[:4] for d in dates if d and len(d) >= 4)
            for _y in _years_in_ans:
                if _y in _all_years:
                    _candidate_years.add(_y)

        # Check context (timeline text) for year-level event mentions
        _ev_words = set(re.findall(r'[a-zA-Z]{4,}', q.lower()))
        _ev_words -= {'when', 'what', 'which', 'year', 'did', 'was', 'has', 'had', 'where', 'how', 'long'}
        if _ev_words and ctx_str:
            for _yr in list(_years_in_ans):
                for _chunk in ctx_str.split('\n'):
                    _cs = _chunk.lower()
                    if _yr in _cs and any(w in _cs for w in _ev_words):
                        _candidate_years.add(_yr)
                        break

        if _candidate_years:
            _best_yr = max(_candidate_years, key=lambda x: ctx_str.count(x) if ctx_str else 1)
            context['answer'] = _best_yr
        elif _years_in_ans:
            context['answer'] = max(set(_years_in_ans), key=_years_in_ans.count)

        return context

    def _action_year_offset(self, context, params):
        """FixCat2.10: Resolve 'last year' / 'this year' text to actual years from session dates."""
        ans = context.get('answer', '')
        dates = context.get('dates', [])

        _ans_lower = str(ans or '').lower()[:120]
        if 'last year' not in _ans_lower:
            return context

        if not dates:
            return context

        _unique_years = set()
        for _d in dates:
            if _d and len(_d) >= 4:
                try:
                    _unique_years.add(int(_d[:4]))
                except:
                    pass

        if not _unique_years:
            return context

        offset = params.get('offset', -1)
        method = params.get('method', 'max_year')

        if method == 'max_year':
            _base_year = max(_unique_years)
        else:
            _base_year = min(_unique_years)

        context['answer'] = str(_base_year + offset)
        return context

    def _action_strip_prefix(self, context, params):
        """FixCat2.6: Strip verbose 'Based on...' / 'According to...' wrappers from answer."""
        ans = context.get('answer', '')
        if not ans:
            return context

        _ans_clean = str(ans).strip()
        # Normalize line separators
        _ans_clean = _ans_clean.replace('\n', ' ').replace('\r', ' ')
        # Collapse multiple spaces
        _ans_clean = re.sub(r'\s+', ' ', _ans_clean)

        _verbose_prefixes = [
            r'^Based on the extracted timeline[.,:]*\s*',
            r'^Based on the provided context[.,:]*\s*',
            r'^Based solely on the provided context[.,:]*\s*',
            r'^Based on the conversation[.,:]*\s*',
            r'^Based on the context[.,:]*\s*',
            r'^The context does not specify[.,:]*\s*',
            r'^The context does not mention[.,:]*\s*',
            r'^There is no information[.,:]*\s*',
            r'^No information[.,:]*\s*',
        ]

        for _vp in _verbose_prefixes:
            _trimmed = re.sub(_vp, '', _ans_clean, flags=re.I).strip()
            if len(_trimmed) < len(_ans_clean) and len(_trimmed) > 5:
                _ans_clean = _trimmed
                break

        context['answer'] = _ans_clean
        return context


# ─── Standalone usage: load rules from KM JSON ─────────────────
def load_rules_from_km(path=None):
    """Load postproc_rules from KM JSON file. Falls back to inline P2 rules."""
    if path is None:
        path = KM_PATH
    try:
        km = json.load(open(path))
        rules = km.get('postproc_rules', [])
        if rules:
            return rules
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return _P2_RULES
