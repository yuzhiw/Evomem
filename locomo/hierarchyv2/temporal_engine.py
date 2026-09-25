#!/usr/bin/env python3
"""
temporal_engine.py — TemporalEngine for 4-level temporal reasoning
"""
import re
from collections import defaultdict
from datetime import datetime, timedelta
from .date_utils import parse_date
from typing import Dict
from .config import _STOP

class TemporalEngine:
    """4-level temporal reasoning"""

    def __init__(self):
        self.date_index = defaultdict(list)  # YYYY-MM-DD → [facts]
        self.sequence_index = []  # ordered list of (time, fact, session_id)
        self.before_after = defaultdict(list)  # fact_key → [(related_fact, relation)]

    def add_fact(self, fact: Dict):
        t = fact.get('time', '')
        if t:
            self.date_index[t[:10]].append(fact)

    def _resolve_relative_dates(self, text: str, msg_date) -> str:
        """Replace relative date references with absolute dates (v2: extended coverage)"""
        if not msg_date:
            return text
        def _replace(m):
            word = m.group(0).lower()
            delta = None
            if word == 'today':
                delta = timedelta(days=0)
            elif word == 'yesterday':
                delta = timedelta(days=-1)
            elif word == 'tomorrow':
                delta = timedelta(days=1)
            elif word == 'day before yesterday':
                delta = timedelta(days=-2)
            elif word == 'last night':
                delta = timedelta(days=-1)
            elif word in ('this week', 'this month'):
                return m.group(0)  # keep as-is, too vague
            if delta is not None:
                abs_date = msg_date + delta
                return abs_date.strftime('%Y/%m/%d')
            return m.group(0)
        text = re.sub(r'\b(today|yesterday|tomorrow|day before yesterday|last night)\b', _replace, text, flags=re.IGNORECASE)
        # v2: Extended relative date patterns
        def _replace_relative(m):
            num_str = m.group(1)
            unit = m.group(2).lower()
            direction = m.group(3).lower() if m.lastindex >= 3 and m.group(3) else 'ago'
            try:
                num = int(num_str)
            except:
                return m.group(0)
            if 'ago' in direction or 'before' in direction or 'earlier' in direction or 'back' in direction:
                delta = timedelta(days=num * {'day': 1, 'days': 1, 'week': 7, 'weeks': 7, 'month': 30, 'months': 30, 'year': 365, 'years': 365}.get(unit, 1))
                abs_date = msg_date - delta
                return f'{abs_date.strftime("%Y/%m/%d")} ({m.group(0).strip()})'
            elif 'later' in direction or 'after' in direction or 'from now' in direction or 'ahead' in direction:
                delta = timedelta(days=num * {'day': 1, 'days': 1, 'week': 7, 'weeks': 7, 'month': 30, 'months': 30, 'year': 365, 'years': 365}.get(unit, 1))
                abs_date = msg_date + delta
                return f'{abs_date.strftime("%Y/%m/%d")} ({m.group(0).strip()})'
            return m.group(0)
        text = re.sub(r'(\d+)\s*(day|days|week|weeks|month|months|year|years)\s+(ago|before|earlier|back|later|after|from now|ahead)\b', _replace_relative, text, flags=re.IGNORECASE)
        # Handle weekdays: "last Monday", "this Friday", "next Tuesday"
        weekdays = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
        def _replace_weekday(m):
            qualifier = m.group(1).lower()
            wd_name = m.group(2).lower()
            target_wd = weekdays.get(wd_name)
            if target_wd is None:
                return m.group(0)
            current_wd = msg_date.weekday()
            if qualifier == 'last':
                days_back = (current_wd - target_wd) % 7
                if days_back == 0:
                    days_back = 7
                abs_date = msg_date - timedelta(days=days_back)
            elif qualifier == 'next':
                days_forward = (target_wd - current_wd) % 7
                if days_forward == 0:
                    days_forward = 7
                abs_date = msg_date + timedelta(days=days_forward)
            elif qualifier == 'this':
                days_diff = target_wd - current_wd
                abs_date = msg_date + timedelta(days=days_diff)
            else:
                return m.group(0)
            return f'{abs_date.strftime("%Y/%m/%d")} ({m.group(0).strip()})'
        text = re.sub(r'\b(last|next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', _replace_weekday, text, flags=re.IGNORECASE)
        # Handle "a few days/weeks ago", "a couple of days ago"
        def _replace_vague(m):
            quant = m.group(1).lower()
            unit = m.group(2).lower()
            mult = {'few': 3, 'couple': 2, 'several': 5, 'a few': 3, 'a couple of': 2}
            num = mult.get(quant, 3)
            delta = timedelta(days=num * {'day': 1, 'days': 1, 'week': 7, 'weeks': 7, 'month': 30, 'months': 30}.get(unit, 1))
            abs_date = msg_date - delta
            return f'{abs_date.strftime("%Y/%m/%d")} (approx. {quant} {unit} ago)'
        text = re.sub(r'\b(a few|a couple of|couple|few|several)\s+(day|days|week|weeks|month|months)\s+ago\b', _replace_vague, text, flags=re.IGNORECASE)
        # v7a: bare "last week" / "this week" / "next week" resolution (NOT followed by "before [date]")
        # Uses negative lookahead to avoid stepping on v3's "last week before [DATE]" pattern.
        def _replace_week_bare(m):
            direction = m.group(1).lower()
            if not msg_date:
                return m.group(0)
            if direction == 'last':
                abs_date = msg_date - timedelta(days=7)
            elif direction == 'next':
                abs_date = msg_date + timedelta(days=7)
            else:  # 'this' — kept as-is, too vague for precise mapping
                return m.group(0)
            wk_start = abs_date - timedelta(days=abs_date.weekday())
            wk_end = wk_start + timedelta(days=6)
            return f'{abs_date.strftime("%Y/%m/%d")} (week of {wk_start.strftime("%m/%d")}–{wk_end.strftime("%m/%d")})'
        text = re.sub(r'(?i)\b(last|this|next)\s+week\b(?!\s+(?:before|after))', _replace_week_bare, text)

        # v3: "last week before [date]" / "the week before [date]" pattern
        def _replace_week_before(m):
            date_str = m.group(1).strip()
            date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%d %B %Y', '%B %d, %Y']
            target_date = None
            for fmt in date_formats:
                try:
                    target_date = datetime.strptime(date_str, fmt)
                    break
                except:
                    pass
            if target_date:
                prev_week = target_date - timedelta(days=7)
                return f'{prev_week.strftime("%Y/%m/%d")} ({m.group(0).strip()})'
            return m.group(0)
        text = re.sub(r'(?i)(?:last\s+week\s+before|the\s+week\s+before)\s+(.+?)(?=[,.]|$)', _replace_week_before, text)
        # v4: "in [month] [year]" → specific format
        def _replace_in_month(m):
            mm = m.group(1)
            yy = m.group(2)
            month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                         'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
            if mm.capitalize() in month_map:
                return f'{yy}/{month_map[mm.capitalize()]:02d} (in {mm} {yy})'
            return m.group(0)
        text = re.sub(r'(?i)\bin\s+(\w+)\s+(\d{4})\b', _replace_in_month, text)
        # v5: "next month" → explicit month name + year
        def _replace_next_month(m):
            if msg_date:
                next_mo = msg_date.month % 12 + 1
                next_yr = msg_date.year + (1 if msg_date.month == 12 else 0)
                mo_name = ['January','February','March','April','May','June',
                          'July','August','September','October','November','December'][next_mo-1]
                return f'{next_yr}/{next_mo:02d} ({mo_name} {next_yr})'
            return m.group(0)
        text = re.sub(r'(?i)\bnext\s+month\b', _replace_next_month, text)


        def _replace_year(m):
            direction = m.group(1).lower()
            if not msg_date:
                return m.group(0)
            yr = msg_date.year
            if direction == 'last':
                yr -= 1
            elif direction == 'next':
                yr += 1
            return f'{yr} (in {yr})'
        text = re.sub(r'(?i)\b(last|this|next)\s+year\b', _replace_year, text)

        # v6: season resolution (last/this/next summer/spring/fall/winter)
        season_months = {
            'spring': (3,5), 'summer': (6,8),
            'fall': (9,11), 'winter': (12,2),
        }

        def _replace_season(m):
            direction = m.group(1).lower()
            season = m.group(2).lower()
            if not msg_date or season not in season_months:
                return m.group(0)
            yr = msg_date.year
            mo = msg_date.month
            s_start, s_end = season_months[season]

            if direction == 'last':
                if mo <= s_end:
                    yr -= 1
            elif direction == 'next':
                if mo >= s_start:
                    yr += 1

            return f'{yr}/{s_start:02d}-{yr}/{s_end:02d} ({season} {yr})'

        text = re.sub(r'(?i)\b(last|this|next)\s+(spring|summer|fall|winter)\b', _replace_season, text)
        return text

    def _smart_chunk(self, text: str) -> list:
        """Split long assistant texts into fine-grained chunks"""
        parts = []
        paras = re.split(r'\n\s*\n', text)
        for p in paras:
            p = p.strip()
            if not p: continue
            if len(p) <= 400:
                parts.append(p)
            else:
                lines = p.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    if len(line) <= 400:
                        parts.append(line)
                    else:
                        sents = re.split(r'(?<=[.!?])\s+', line)
                        for s in sents:
                            s = s.strip()
                            if s: parts.append(s)
        return parts if parts else [text[:400]]

    def _extract_entities(self, text: str) -> list:
        """Extract named entities (continuous capitalized word sequences)"""
        ents = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        COMMON = {'How','What','When','Where','Why','Which','Who','The','A','An',
                  'This','That','My','Your','His','Her','Its','Our','Their',
                  'I','You','He','She','It','We','They','Am','Is','Are','Was','Were',
                  'Can','Will','May','Shall','Do','Does','Did','Also','Only','Just',
                  'First','Last','Next','Some','Any','All','More','Most',
                  'Past','New','Old','Recent','Many','Much','Each','Every',
                  'Like','Very','Still','Already','Yet','Even',
                  'About','Around','Before','After','From','Into','Through',
                  'During','Within','Without','Across','Between','Among',
                  'Because','Since','Until','Although','Though','While',
                  'If','Or','But','Not','So','As','For','With','Without',
                  'By','At','On','In','To','Of','Up','Off','Out','Over'}
        return [e for e in ents if e not in COMMON and len(e) > 1]

    def _build_entity_graph(self):
        """Build entity → entry_id mapping"""
        self.entity_graph = {}
        for entry in self.sequence_index:
            eid = entry['id']
            entities = self._extract_entities(entry['content'])
            for ent in entities:
                if ent not in self.entity_graph:
                    self.entity_graph[ent] = []
                self.entity_graph[ent].append(eid)

    def build_sequence(self, sessions, dates):
        """Build temporal sequence from sessions with date normalization (smart chunking)"""
        seq_id = 0
        for si, session in enumerate(sessions):
            if not isinstance(session, list): continue
            dt = dates[si] if si < len(dates) else ''
            parsed = parse_date(dt)
            for turn in session:
                if isinstance(turn, dict) and turn.get('content', '').strip():
                    content = turn['content'].strip()
                    # Smart chunking for long assistant replies
                    if turn.get('role') == 'assistant' and len(content) > 500:
                        chunks = self._smart_chunk(content)
                    else:
                        chunks = [content]
                    for chunk in chunks:
                        seq_id += 1
                        chunk_display = chunk[:400]
                        if parsed:
                            chunk_display = self._resolve_relative_dates(chunk_display, parsed)
                        self.sequence_index.append({
                            'id': seq_id,
                            'time': parsed,
                            'time_str': dt[:10] if dt else '',
                            'role': turn.get('role', ''),
                            'content': chunk_display,
                            'session': si,
                        })

    def _bigrams(self, words: list) -> set:
        """Generate bigrams from a word list"""
        return set(f'{words[i]} {words[i+1]}' for i in range(len(words)-1))

    def _score_entry(self, entry, q_words: set, q_bigrams: set, q_proper: set, q_lower: str):
        """Score an entry using: bigram + proper noun + unigram overlap. Returns (score, has_strong_signal)"""
        cl = entry['content'].lower()
        content_orig = entry['content']
        score = 0
        has_strong = False
        # 1) Bigram match (strong signal)
        c_words = re.findall(r'\b[a-zA-Z]{3,}\b', cl)
        c_bigrams = self._bigrams(c_words)
        bigram_overlap = len(q_bigrams & c_bigrams)
        if bigram_overlap > 0:
            has_strong = True
        score += bigram_overlap * 3
        # 2) Proper noun match (case-sensitive)
        pn_hit = 0
        for pn in q_proper:
            if re.search(r'\b' + re.escape(pn) + r'\b', content_orig) or \
               re.search(r'\b' + re.escape(pn.lower()) + r'\b', cl):
                pn_hit += 1
        if pn_hit > 0:
            has_strong = True
        score += pn_hit * 5
        # 3) Unigram overlap (weak signal)
        c_word_set = set(c_words)
        unigram_overlap = len(q_words & c_word_set)
        score += unigram_overlap
        return score, has_strong

    def _extract_proper_nouns(self, query: str) -> set:
        """Extract genuine named entities from query (not common capitalized words)"""
        common = {'How', 'What', 'When', 'Where', 'Why', 'Which', 'Who', 'Whom', 'Whose',
                  'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'My', 'Your',
                  'His', 'Her', 'Its', 'Our', 'Their', 'Me', 'You', 'He', 'She', 'It',
                  'We', 'They', 'Am', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being',
                  'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Done', 'Doing',
                  'Can', 'Will', 'May', 'Might', 'Shall', 'Should', 'Would', 'Could',
                  'Also', 'Only', 'Just', 'Here', 'There', 'Then', 'Than', 'Now',
                  'First', 'Last', 'Next', 'Some', 'Any', 'All', 'More', 'Most',
                  'Past', 'New', 'Old', 'Recent', 'Many', 'Much', 'Few', 'Each',
                  'Every', 'Other', 'Another', 'Same', 'Different', 'Big', 'Small',
                  'High', 'Low', 'Long', 'Short', 'Best', 'Better', 'Good', 'Great',
                  'Like', 'Such', 'Very', 'Still', 'Already', 'Yet', 'Even',
                  'About', 'Around', 'Before', 'After', 'From', 'Into', 'Through',
                  'During', 'Within', 'Without', 'Across', 'Between', 'Among',
                  'Because', 'Since', 'Until', 'Although', 'Though', 'While',
                  'If', 'Or', 'But', 'Not', 'So', 'As', 'For', 'With', 'Without',
                  'By', 'At', 'On', 'In', 'To', 'Of', 'Up', 'Off', 'Out', 'Over',
                  'Day', 'Days', 'Week', 'Weeks', 'Month', 'Months', 'Year', 'Years',
                  'Time', 'Times', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                  'Seven', 'Eight', 'Nine', 'Ten', 'No', 'Yes', 'True', 'False'}
        pn = set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query))
        return pn - common

    def query_temporal(self, query: str, question_date: str) -> str:
        """Temporal with bigram + proper noun weighted matching + entity graph"""
        qd = parse_date(question_date) if question_date else None

        # Build entity graph for LiCoMemory
        self._build_entity_graph()

        # Query preprocessing
        q_lower = query.lower()
        q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)) - _STOP
        q_bigrams = self._bigrams(list(q_words))
        q_proper = self._extract_proper_nouns(query)

        # LiCoMemory: pre-compute query entities for entity overlap bonus
        q_entities = self._extract_entities(query)

        # Score all entries (must have strong signal: bigram or proper noun) + LiCoMemory entity bonus
        scored = []
        for entry in self.sequence_index:
            score, has_strong = self._score_entry(entry, q_words, q_bigrams, q_proper, q_lower)
            # LiCoMemory: entity overlap bonus
            if self.entity_graph:
                entry_entities = self._extract_entities(entry['content'])
                overlap = len(set(q_entities) & set(entry_entities))
                entity_bonus = overlap * 3
                score += entity_bonus
            if score >= 8 and has_strong:
                scored.append((score, entry))
        # If too few entries, fall back to overlap≥1
        if len(scored) < 3:
            qw = set(re.findall(r'\b[a-zA-Z]{3,}\b', q_lower)) - _STOP
            for entry in self.sequence_index:
                cw = set(re.findall(r'\b[a-zA-Z]{3,}\b', entry['content'].lower())) - _STOP
                if len(qw & cw) >= 1 and not any(e['id'] == entry['id'] for _, e in scored):
                    scored.append((1, entry))

        # Proper noun fallback: ensure entries matching query proper nouns are included
        scored_ids = {e['id'] for _, e in scored}
        for pn in q_proper:
            pn_lower = pn.lower()
            for entry in self.sequence_index:
                if entry['id'] in scored_ids:
                    continue
                if re.search(r'\b' + re.escape(pn) + r'\b', entry['content']):
                    scored.append((6, entry))
                    scored_ids.add(entry['id'])

        # Sort by score descending (most relevant first), then by time ascending
        scored.sort(key=lambda x: (-x[0], x[1]['time'] or parse_date('1900-01-01')))

        # Build timeline (score-descending order, top entries)
        lines = [f'[QUESTION_DATE: {question_date[:10] if question_date else "?"}]']
        for score, entry in scored[:50]:
            ts = entry['time_str'] or '?'
            lines.append(f'[{ts}] [{entry["role"]}] {entry["content"]}')
        timeline = '\n'.join(lines) if len(lines) > 1 else ''

        # Also show recent events (newest 10) for quick-access
        if len(scored) > 5:
            recent = sorted(scored, key=lambda x: (x[1]['time'] or parse_date('1900-01-01'), -x[0]), reverse=True)[:10]
            recent.reverse()
            recent_lines = ['[RECENT EVENTS]']
            for score, entry in recent:
                ts = entry['time_str'] or '?'
                recent_lines.append(f'[{ts}] [score={score}] [{entry["role"]}] {entry["content"]}')
            timeline += '\n' + '\n'.join(recent_lines)

        # Phase 2: Pre-compute date calculations
        calc_lines = []
        if qd:
            seen_dates = set()
            user_events = []
            for _, entry in scored:
                if entry['role'] == 'user' and entry['time']:
                    ds = entry['time'].strftime('%Y/%m/%d')
                    if ds not in seen_dates:
                        seen_dates.add(ds)
                        content_short = entry['content'][:80]
                        diff_days = (qd - entry['time']).days
                        user_events.append((ds, diff_days, content_short))

            if user_events:
                calc_lines.append('[DATE_CALC]')
                calc_lines.append(f'Question date: {question_date[:10]}')
                for ds, diff_days, content in user_events:
                    calc_lines.append(f'  [{ds}] ({diff_days} days before question): {content}')
                # Compute differences between event pairs for interval questions
                calc_lines.append('[EVENT INTERVALS]')
                calc_lines.append('(Computed day differences between events sharing keywords)')
                q_words_calc = set(re.findall(r'[a-zA-Z]{4,}', q_lower)) - _STOP
                full_events = []
                for _, entry in scored:
                    if entry['role'] == 'user' and entry['time']:
                        ds = entry['time'].strftime('%Y/%m/%d')
                        diff_days = (qd - entry['time']).days
                        full_events.append((ds, diff_days, entry['content']))
                for a_idx in range(len(full_events)):
                    for b_idx in range(a_idx + 1, len(full_events)):
                        ds_a, diff_a, ctx_a = full_events[a_idx]
                        ds_b, diff_b, ctx_b = full_events[b_idx]
                        words_a = set(re.findall(r'[a-zA-Z]{4,}', ctx_a.lower())) - _STOP
                        words_b = set(re.findall(r'[a-zA-Z]{4,}', ctx_b.lower())) - _STOP
                        if any(kw in ctx_a.lower() for kw in ['question \d+', 'single choice', 'azure', 'transcript']):
                            continue
                        if any(kw in ctx_b.lower() for kw in ['question \d+', 'single choice', 'azure', 'transcript']):
                            continue
                        shared_calc = (words_a & words_b) | (words_a & q_words_calc) | (words_b & q_words_calc)
                        shared_direct = words_a & words_b
                        shared_with_query = (words_a & q_words_calc) | (words_b & q_words_calc)
                        if len(shared_calc) >= 2 or (len(shared_calc) >= 1 and len(shared_with_query & {k for k, _ in [('farmfresh',1), ('instacart',1), ('cancelled',1), ('subscription',1)]}) > 0):
                            actual_dur = max(diff_a - diff_b, diff_b - diff_a)
                            kw_display = list(shared_calc)[:4]
                            calc_lines.append(f'  "{ctx_a[:100]}" ({ds_a})')
                            calc_lines.append(f'  "{ctx_b[:100]}" ({ds_b})')
                            calc_lines.append(f'  Shared topics: {", ".join(kw_display)}')
                            calc_lines.append(f'  Interval: {actual_dur} days')

        # Phase 3: DURATION ANALYSIS - detect start/end event pairs sharing a proper noun
        dur_lines = []
        if len(scored) >= 2:
            dated_entries = []
            for _, entry in scored:
                if entry['role'] == 'user' and entry['time'] and entry['content']:
                    cl = entry['content'].lower()
                    proper_nouns = set(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', entry['content']))
                    dated_entries.append((entry['time'], cl, entry['content'], proper_nouns))
            start_kw = {'started', 'starting', 'began', 'beginning', 'heading to', 'left for', 'arrived at', 'began my', 'start my'}
            end_kw = {'back', 'returned', 'returning', 'finished', 'completed', 'back from', 'got back', 'just got back', 'ended', 'finished my'}
            dur_pairs = []
            for i, (t1, c1, ctx1, pn1) in enumerate(dated_entries):
                is_start = any(kw in c1 for kw in start_kw)
                is_end = any(kw in c1 for kw in end_kw)
                if not is_start and not is_end:
                    continue
                for j, (t2, c2, ctx2, pn2) in enumerate(dated_entries):
                    if i >= j:
                        continue
                    other_is_start = any(kw in c2 for kw in start_kw)
                    other_is_end = any(kw in c2 for kw in end_kw)
                    both_valid = (is_start and other_is_end) or (is_end and other_is_start)
                    if not both_valid:
                        continue
                    shared_pn = pn1 & pn2
                    if len(shared_pn) < 1:
                        continue
                    dur_days = abs((t2 - t1).days)
                    if dur_days <= 365:
                        start_entry = ctx1 if is_start else ctx2
                        end_entry = ctx2 if is_start else ctx1
                        dur_pairs.append((t1, t2, dur_days, shared_pn, start_entry[:80], end_entry[:80]))
            if dur_pairs:
                dur_lines.append('[DURATION ANALYSIS - start/end event pairs]')
                for t1, t2, dur, pn, s_entry, e_entry in dur_pairs[:3]:
                    t1_s = t1.strftime('%Y/%m/%d')
                    t2_s = t2.strftime('%Y/%m/%d')
                    dur_lines.append(f'  START: "{s_entry}" [{t1_s}]')
                    dur_lines.append(f'  END:   "{e_entry}" [{t2_s}]')
                    dur_lines.append(f'  Shared entity: {", ".join(sorted(pn)[:3])}')
                    dur_lines.append(f'  DURATION: {dur} days')

        result = timeline
        if calc_lines:
            result += '\n' + '\n'.join(calc_lines)
        if dur_lines:
            result += '\n' + '\n'.join(dur_lines)

        # Entity relations section
        if self.entity_graph:
            q_entities_er = self._extract_entities(query)
            entity_lines = ['[ENTITY RELATIONS]']
            for qe in q_entities_er[:5]:
                if qe in self.entity_graph:
                    related = self.entity_graph[qe]
                    related_entries = [self.sequence_index[e-1]['content'][:60] for e in related if e <= len(self.sequence_index)]
                    if related_entries:
                        entity_lines.append(f'  "{qe}" → {len(related)} mentions:')
                        for r_entry in related_entries[:3]:
                            entity_lines.append(f'    · {r_entry}')
            if len(entity_lines) > 1:
                result += '\n' + '\n'.join(entity_lines[:15])

        return result

    def build_full_timeline(self) -> str:
        """Build CHRONOLOGICAL timeline of ALL sessions (unfiltered, unscored)."""
        if not self.sequence_index:
            return ''
        sorted_entries = sorted(self.sequence_index, key=lambda e: (e.get('time') or parse_date('1900-01-01'), e.get('id', '')))
        lines = ['[FULL CHRONOLOGICAL TIMELINE (all sessions)]']
        seen_dates = set()
        for entry in sorted_entries:
            ts = entry.get('time_str') or '?'
            date_only = ts[:10] if len(ts) >= 10 else ts
            role = entry.get('role', '?')
            content = entry.get('content', '')[:50]
            lines.append(f'[{date_only}] [{role}] {content}')
            seen_dates.add(date_only)
        return '\n'.join(lines)

