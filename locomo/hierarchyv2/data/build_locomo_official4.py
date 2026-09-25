#!/usr/bin/env python3
"""Build the LoCoMo-120 subset with LoCoMo's OFFICIAL four categories.

Official LoCoMo categories:
  1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial

We keep only categories 1-4 (adversarial excluded, as is common practice) and
sample 30 questions per category (120 total), seed 42.

Output format matches the flattened QA format used by run_item_v2.
"""
import json, re, random
from collections import Counter

SRC = '/root/autodl-fs/newpy0522/lunwen_latest/hierarchyv2_lunwen/paperwriting/paperwriting/lunwen1/reproducibility/data/locomo10.json'
DST = '/root/autodl-fs/newpy0522/lunwen_latest/hierarchyv2_lunwen/paperwriting/paperwriting/lunwen1/reproducibility/code_locomo/data/locomo120_official4.json'

CAT_TO_TYPE = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop'}
PER_CAT = 30
SEED = 42

MONTHS = {m: i + 1 for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july',
     'august', 'september', 'october', 'november', 'december'])}
DT_RE = re.compile(r'(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})')


def parse_dt(s):
    m = DT_RE.search(s)
    if m:
        day, mon, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if mon in MONTHS:
            return f'{year}-{MONTHS[mon]:02d}-{day:02d}'
    return ''


def main():
    data = json.load(open(SRC))
    pool = {c: [] for c in CAT_TO_TYPE}
    for conv in data:
        cid = conv.get('sample_id', 'conv')
        convd = conv['conversation']
        session_keys = sorted([k for k in convd if re.match(r'^session_\d+$', k)],
                              key=lambda k: int(k.split('_')[1]))
        sessions, dates = [], []
        for sk in session_keys:
            turns_out = []
            for t in convd[sk]:
                if not isinstance(t, dict):
                    continue
                content = t.get('text', '')
                if not content:
                    continue
                role = 'user' if t.get('speaker') == convd.get('speaker_a') else 'assistant'
                turns_out.append({'role': role, 'content': f"[{t.get('speaker','user')}] {content}"})
            if turns_out:
                sessions.append(turns_out)
                dates.append(parse_dt(convd.get(sk + '_date_time', '')))
        for qa in conv['qa']:
            cat = int(qa.get('category', 1))
            if cat not in CAT_TO_TYPE:
                continue
            ans = qa.get('answer', '')
            if ans is None or not str(ans).strip():
                continue
            pool[cat].append({
                'cid': cid, 'cat': cat, 'question': qa.get('question', ''),
                'answer': ans, 'date': dates[-1] if dates else '',
                'sessions': sessions, 'dates': dates,
            })

    rng = random.Random(SEED)
    out = []
    for cat in sorted(CAT_TO_TYPE):
        items = pool[cat]
        pick = rng.sample(items, PER_CAT) if len(items) >= PER_CAT else items
        for r in pick:
            out.append({
                'question_id': f"{r['cid']}_c{cat}_{len(out)}",
                'question_type': CAT_TO_TYPE[cat],
                'category': str(cat),
                'question': r['question'],
                'question_date': r['date'],
                'answer': r['answer'],
                'haystack_sessions': r['sessions'],
                'haystack_dates': r['dates'],
            })
    json.dump(out, open(DST, 'w'), ensure_ascii=False, indent=1)
    print('written:', len(out), '->', DST)
    print('type dist:', dict(Counter(x['question_type'] for x in out)))
    print('category dist:', dict(Counter(x['category'] for x in out)))
    print('empty answers:', sum(1 for x in out if not str(x['answer']).strip()))
    print('conversations covered:', len(set(x['question_id'].split('_')[0] for x in out)))


if __name__ == '__main__':
    main()
