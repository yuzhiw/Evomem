#!/usr/bin/env python3
"""ms_counter.py — MS(多会话计数/聚合)覆盖不足的三项改进（cod2）。

对应文档 MS_COVERAGE_BRAINSTORM_20260918.md：
  P4  相对时间显式化：把 "in the past two weeks" / "this year" / "since the start of
      the year" 等解析成绝对日期区间，用于过滤事件（现仅 TR 分支有月份过滤）。
  P2  谓词+时间窗的穷尽式候选召回：不做 top-k 相似度截断，而是对**全部会话**做词形
      变体词法扫描（问题内容词 + 动作词），把命中句子连日期一起作为 "[CANDIDATE EVENT
      MENTIONS]" 注入。目标是覆盖，不是精度；精度交给下一步的 LLM 判定。
  P1  先列清单、再由代码计数：LLM 只负责枚举逐条证据（禁止心算），计数/求和由
      Python 确定性完成 —— 避免"提及≠事件"造成的高估与心算失误。

所有功能可用环境变量关闭：V2_MS_LEDGER=0（默认 1）
"""
import os, re
from datetime import datetime, timedelta

ENABLED = os.environ.get('V2_MS_LEDGER', '1') == '1'
MAX_MENTIONS = int(os.environ.get('V2_MS_MAX_MENTIONS', '90'))

_QWORDS = {'how', 'many', 'much', 'what', 'which', 'when', 'where', 'who', 'whom', 'whose', 'why',
           'did', 'do', 'does', 'have', 'has', 'had', 'am', 'is', 'are', 'was', 'were',
           'i', 'my', 'me', 'you', 'your', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
           'of', 'with', 'and', 'or', 'this', 'that', 'these', 'those', 'total', 'number',
           'times', 'time', 'been', 'over', 'during', 'from', 'since', 'about', 'last',
           'past', 'next', 'all', 'any', 'some', 'it', 'its', 'they', 'them', 'we', 'our'}

# ── P4: 相对时间 → 绝对区间 ──────────────────────────────────────────────
_NUM_WORDS = {'a': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
              'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'couple of': 2, 'few': 3, 'several': 4}


def _to_date(s):
    """宽容日期解析：接受 2023-05-30 / 2023/05/30 / 2023-05-30 (Tue) 14:15 等。"""
    if not s:
        return None
    t = str(s).strip().replace('/', '-')[:10]
    try:
        return datetime.strptime(t, '%Y-%m-%d').date()
    except Exception:
        return None


def parse_time_window(q: str, qdate: str):
    """返回 (start_date, end_date)（datetime.date），无法解析返回 (None, None)。"""
    qd = _to_date(qdate)
    if qd is None:
        return None, None
    t = (q or '').lower()

    m = re.search(r'\b(?:in |over |during )?(?:the )?(?:past|last|previous)\s+(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|couple of|few|several)\s+(day|week|month|year)s?\b', t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS.get(m.group(1), 1)
        unit = m.group(2)
        days = {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit] * n
        return qd - timedelta(days=days), qd

    if re.search(r'\b(?:this|so far this|since the start of the|since the beginning of the)\s+year\b', t) or \
       re.search(r'\bsince (the )?(start|beginning) of (the )?year\b', t):
        return datetime(qd.year, 1, 1).date(), qd
    if re.search(r'\bthis month\b|\bso far this month\b', t):
        return datetime(qd.year, qd.month, 1).date(), qd
    # 无数词形式：last/past week|month|year（2026-09-18 补）
    _m3 = re.search(r'\b(?:in |over |during )?(?:the )?(last|past|previous)\s+(week|month|year)\b', t)
    if _m3 and not re.search(r'\b(?:last|past|previous)\s+(?:few|couple of|several)\b', t):
        days = {'week': 7, 'month': 30, 'year': 365}[_m3.group(2)]
        return qd - timedelta(days=days), qd
    if re.search(r'\bthis week\b', t):
        return qd - timedelta(days=qd.weekday()), qd
    if re.search(r'\b(?:in the )?(?:past|last) (?:few|couple of|several) (day|week|month|year)s\b', t):
        m2 = re.search(r'(?:past|last) (?:few|couple of|several) (day|week|month|year)s', t)
        days = {'day': 3, 'week': 21, 'month': 90, 'year': 730}[m2.group(1)]
        return qd - timedelta(days=days), qd
    if re.search(r'\brecently\b|\blast few\b', t):
        return qd - timedelta(days=30), qd
    if re.search(r'\btoday\b', t):
        return qd, qd
    return None, None


# ── P2: 词形变体 + 穷尽式会话扫描 ─────────────────────────────────────────
_MORPH_EXTRA = {
    'buy': ['bought', 'buying', 'purchase', 'purchased'],
    'bake': ['baked', 'baking'],
    'attend': ['attended', 'attending'],
    'bike': ['bicycle', 'cycling', 'cycled'],
    'camp': ['camping', 'camped'],
    'plant': ['plants', 'planted'],
    'game': ['games', 'gaming', 'played'],
    'play': ['played', 'playing', 'playthrough'],
}


def query_terms(q: str):
    """问题里的内容词（含粗词干与近义扩展）。"""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", (q or '').lower())
    out = []
    for w in words:
        if w in _QWORDS or len(w) < 4:
            continue
        out.append(w)
        if w.endswith('s') and len(w) > 4:
            out.append(w[:-1])
        if w.endswith('ing') and len(w) > 5:
            out.append(w[:-3])
            out.append(w[:-3] + 'e')
        if w.endswith('ed') and len(w) > 4:
            out.append(w[:-2])
        for k, vs in _MORPH_EXTRA.items():
            if w.startswith(k[:max(3, len(k) - 1)]):
                out.extend(vs)
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text or '')
    out = []
    for p in parts:
        p = p.strip()
        if 25 <= len(p) <= 500:
            out.append(p)
    return out


def candidate_mentions(sessions, dates, q: str, qdate: str = ''):
    """穷尽式扫描全部会话，返回命中问题的句子列表 [(date_str, sentence, session_idx)]。

    只做召回，不做 top-k 截断；用 IDF 加权压噪音（如 "united/states" 这类在全库遍地
    都有的词），要求命中“核心词”至少 1 个且命中总词数 ≥ 2。
    """
    import math
    terms = query_terms(q)
    if not terms:
        return [], (None, None)
    win_start, win_end = parse_time_window(q, qdate)
    sessions = sessions or []
    dates = dates or []

    # 逐会话建立词集 → 计算 df/idf
    sess_words = []
    df = {t: 0 for t in terms}
    for session in sessions:
        words = set()
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict) and str(turn.get('role', 'user')) == 'user':
                    words |= set(re.findall(r"[a-z0-9']+", str(turn.get('content', '')).lower()))
        sess_words.append(words)
        for t in terms:
            if t in words or any(w.startswith(t[:max(4, len(t) - 2)]) for w in words if len(w) >= 4):
                df[t] += 1
    N = max(1, len(sessions))
    idf = {t: math.log(1 + N / (1 + df.get(t, 0))) for t in terms}
    eligible = [t for t in terms if df.get(t, 0) <= 0.60 * N] or terms
    eligible.sort(key=lambda t: -idf[t])
    core = set(eligible[:3])                     # 最具区分度的 3 个词

    hits, seen = [], set()
    for si, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        dstr = str(dates[si])[:10].replace('/', '-') if si < len(dates) else ''
        if win_start and dstr:
            sd = _to_date(dstr)
            if sd and not (win_start <= sd <= win_end):
                continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            if str(turn.get('role', 'user')) != 'user':
                continue
            for sent in _sentences(str(turn.get('content', ''))):
                low = sent.lower()
                matched = [t for t in terms if t in low]
                if len(matched) < 2:
                    continue
                if not (core & set(matched)):        # 必顶命中一个核心词
                    continue
                score = sum(idf[t] for t in matched)
                key = (dstr, low[:80])
                if key in seen:
                    continue
                seen.add(key)
                hits.append((dstr, sent, si, score))
    hits.sort(key=lambda x: (-x[3], x[0]))
    hits = hits[:MAX_MENTIONS]
    hits.sort(key=lambda x: x[0])
    return [h[:3] for h in hits], (win_start, win_end)


def render_mentions(mentions, window=None):
    if not mentions:
        return ''
    if window and window[0]:
        head = (f'[CANDIDATE EVENT MENTIONS (exhaustive lexical scan over ALL sessions; '
                f'time window {window[0]}..{window[1]})]')
    else:
        head = '[CANDIDATE EVENT MENTIONS (exhaustive lexical scan over ALL sessions)]'
    lines = [head]
    for dstr, sent, si in mentions:
        dt = f'[{dstr}] ' if dstr else ''
        lines.append(f'  {dt}(s{si}) {sent[:220]}')
    return '\n'.join(lines)


# ── P1: 清单 → 代码计数/求和 ─────────────────────────────────────────────
_LINE_RE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s*(.+)$')
_VAL_RE = re.compile(r'\|\s*([-+]?\$?\d[\d,]*\.?\d*)\s*$')


def parse_item_list(ans: str):
    """解析 LLM 枚举出来的清单，返回 [(desc, value_or_None)]。"""
    items = []
    for raw in (ans or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        body = m.group(1).strip() if m else (line if len(line) > 12 and '|' in line else None)
        if not body:
            continue
        val = None
        mv = _VAL_RE.search(body)
        if mv:
            try:
                val = float(mv.group(1).replace('$', '').replace(',', ''))
            except Exception:
                val = None
            body = body[:mv.start()].strip()
        if not body:
            continue
        items.append((body, val))
    return items


def _norm_desc(d: str) -> str:
    d = re.sub(r'[^a-z0-9 ]', ' ', (d or '').lower())
    return ' '.join(d.split()[:12])


_MONEY_RE = re.compile(r'\b(money|amount|cost|costs|pay|paid|price|dollars?|usd|expenses?|spending|budget)\b|\$')


def is_money_q(q: str) -> bool:
    """是否为金额类问题（决定要不要加 $ 前缀）。
    注意不要用单独的 'spent'——"how many hours have I spent" 不是金额题。"""
    t = (q or '').lower()
    if _MONEY_RE.search(t):
        return True
    return bool(re.search(r'how much', t) and re.search(r'\bspen[td]\b', t))


def deterministic_answer(ans: str, q: str, mode: str):
    """由清单确定性地给出答案。mode ∈ {'count','sum'}。失败返回 None。"""
    items = parse_item_list(ans)
    if len(items) < 1:
        return None
    seen, uniq = set(), []
    for desc, val in items:
        k = _norm_desc(desc)
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append((desc, val))
    if mode == 'sum':
        vals = [v for _, v in uniq if v is not None]
        vals = [v for v in vals if v > 0]
        if not vals:
            return None
        s = sum(vals)
        s = int(s) if abs(s - round(s)) < 1e-6 else round(s, 2)
        return f'${s}' if is_money_q(q) else str(s)
    return str(len(uniq))


def detect_mode(q: str) -> str:
    """计数 vs 求和（金额/时长累加）。"""
    t = (q or '').lower()
    if re.search(r'how much (money|total|did i (spend|pay))|total (amount|money|cost)|have i spent|did i spend|total.*(spend|spent)', t):
        return 'sum'
    if re.search(r'how (many|much) (days|hours|weeks|minutes|months)\b', t) and \
       re.search(r'\b(spend|spent|take|took|lasted)\b', t):
        return 'sum'
    return 'count'


LIST_PROMPT = (
    'You are given candidate event mentions retrieved from the whole conversation history. '
    'Enumerate EVERY DISTINCT occurrence that matches the question. Output ONE line per occurrence, '
    'formatted exactly as:\n'
    '- <short description of THIS occurrence> | <numeric value if the question asks for an amount/duration, '
    'otherwise 1>\n'
    'Rules: include only COMPLETED occurrences stated by the user; exclude plans, intentions, suggestions, '
    'advice, hypotheticals, and events belonging to other people unless asked. '
    'If two mentions describe the SAME occurrence, output it only once. '
    'DO NOT output a total, a count, or any prose. Output ONLY the list.'
)

# ── P3: 会话级 Map-Reduce（覆盖率兜底）──────────────────────────────────
MAP_BATCH = int(os.environ.get('V2_MS_MAP_BATCH', '8'))
MAP_ENABLED = os.environ.get('V2_MS_MAPREDUCE', '1') == '1'

_MAP_SYS = ('You are a retrieval filter. For each numbered session below, decide whether it contains '
            'at least one occurrence of the event the user asks about. Be inclusive: a mention, a plan, '
            'a summary, or a discussion of the event all count as a mention. '
            'Reply with ONLY a comma-separated list of the session numbers that contain it, '
            'or the single word NONE.')


def map_select_sessions(sessions, dates, q: str, qdate: str, call_llm, batch_size: int = None,
                        max_batches: int = 12):
    """P3: 逐批问“这批会话里有没有该事件的提及”，返回命中的会话下标集合。

    动机：词法召回会因同义词/表述差异漏掉证据（如 "tank" vs "aquarium"）；
    让 LLM 对**每个会话**做一次 bin 判定，才能把覆盖率从 top-k 中提到“全部会话”。
    成本：ceil(N_sessions / batch) 次调用（LME 约 45 会话 → 6 次）。
    """
    if not MAP_ENABLED or not sessions:
        return []
    batch_size = batch_size or MAP_BATCH
    win_start, win_end = parse_time_window(q, qdate)
    idx_pool = []
    for si in range(len(sessions)):
        d = _to_date(str(dates[si])[:10].replace('/', '-')) if si < len(dates or []) else None
        if win_start and d and not (win_start <= d <= win_end):
            continue
        idx_pool.append(si)
    if not idx_pool:
        idx_pool = list(range(len(sessions)))
    hit = set()
    n_batches = 0
    for b in range(0, len(idx_pool), batch_size):
        if n_batches >= max_batches:
            break
        chunk = idx_pool[b:b + batch_size]
        blocks = []
        for n, si in enumerate(chunk, 1):
            txt = _session_text(sessions[si], max_chars=1400)
            dstr = str(dates[si])[:10] if si < len(dates or []) else '?'
            blocks.append(f'[{n}] ({dstr}) {txt}')
        user = ('Session batch:\n' + '\n\n'.join(blocks) +
                f'\n\nQuestion: {q}\nWhich session numbers mention it? (comma-separated, or NONE)')
        try:
            r = call_llm([{'role': 'system', 'content': _MAP_SYS},
                          {'role': 'user', 'content': user}], max_tokens=64) or ''
        except Exception:
            r = ''
        n_batches += 1
        nums = [int(x) for x in re.findall(r'\d+', r)]
        for n in nums:
            if 1 <= n <= len(chunk):
                hit.add(chunk[n - 1])
    return sorted(hit)


def _session_text(session, max_chars=1400):
    parts = []
    for turn in (session or []):
        if isinstance(turn, dict) and str(turn.get('content', '')).strip():
            role = turn.get('role', '?')
            parts.append(f'{role}: {str(turn["content"]).strip()}')
    return ('  '.join(parts))[:max_chars]


def render_sessions(sessions, dates, idxs, max_chars=60000):
    """把被 map 命中的会话渲染成上下文（用于 P1 枚举）。"""
    out, used = [], 0
    for si in idxs:
        dstr = str(dates[si])[:10] if si < len(dates or []) else '?'
        txt = _session_text(sessions[si], max_chars=2600)
        block = f'[session {si}] ({dstr}) {txt}'
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return '\n\n'.join(out)
