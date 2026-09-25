#!/usr/bin/env python3
"""
main.py — Main entry point for TriMem-AR
"""
import sys, os, json, re
from collections import defaultdict

from .config import DATA_PATH, KM_PATH
from .run_item_v2 import run_item_v2

# Alias for backward compatibility
run_item = run_item_v2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA_PATH)
    parser.add_argument('--tag', type=str, default='trimem_ar')
    parser.add_argument('--part', type=int, default=0, help='Part index for parallel execution (0-based)')
    parser.add_argument('--num_parts', type=int, default=1, help='Total number of parallel parts')
    args = parser.parse_args()

    data = json.load(open(args.data))
    # Slice data for parallel execution
    if args.num_parts > 1:
        part_size = (len(data) + args.num_parts - 1) // args.num_parts
        start = args.part * part_size
        end = min(start + part_size, len(data))
        data = data[start:end]
        print(f' Part {args.part}/{args.num_parts-1}: items {start}-{end-1}')

    print(f'\n{"="*60}', flush=True)
    print(f'  TriMem-AR: {len(data)} items', flush=True)
    print(f'{"="*60}', flush=True)

    results = []
    correct = 0
    for idx, item in enumerate(data):
        qid = item.get('question_id', '?')
        qt = item.get('question_type', '?')
        print(f'[{idx+1}/{len(data)}] {qid} ({qt})', flush=True)
        r = run_item(item)
        results.append(r)
        if r.get('correct', False):
            correct += 1
        status = '✅' if r.get('correct') else '❌' if not r.get('invalid') else '⚠️'
        print(f'  -> {status}', flush=True)

        # Print intermediate results every 2 questions
        if (idx + 1) % 2 == 0:
            processed = idx + 1
            acc = round(correct / processed * 100, 1)
            print(f'  [{processed}/{len(data)}] Partial: {correct}/{processed} correct ({acc}%)', flush=True)

    # Summary
    total = sum(1 for r in results if not r.get('invalid'))
    total_acc = round(correct / total * 100, 1) if total else 0

    from collections import defaultdict as dd
    by_type = dd(lambda: {'t': 0, 'c': 0})
    for r in results:
        qt = r.get('type', '?')
        by_type[qt]['t'] += 1
        if r.get('correct'):
            by_type[qt]['c'] += 1

    print(f'\n{"="*60}', flush=True)
    print(f'  TriMem-AR {args.tag}: {total_acc}% ({correct}/{total})', flush=True)
    for t, s in by_type.items():
        ta = round(s['c'] / s['t'] * 100, 1) if s['t'] else 0
        print(f'    {t}: {ta}% ({s["c"]}/{s["t"]})', flush=True)
    print(f'{"="*60}', flush=True)

    # Save
    out = {
        'tag': args.tag,
        'total': total,
        'correct': correct,
        'accuracy': total_acc,
        'by_type': {t: {'total': s['t'], 'correct': s['c'], 'accuracy': round(s['c'] / s['t'] * 100, 1) if s['t'] else 0} for t, s in by_type.items()},
        'results': results,
    }
    out_path = os.path.join(os.environ.get('OUT_DIR', 'results'), f'{args.tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'  Saved to {out_path}', flush=True)


if __name__ == '__main__':
    main()
