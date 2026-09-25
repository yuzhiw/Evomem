#!/usr/bin/env python3
"""
eval_base.py — Shared evaluation logic for hierarchyv2 ablation experiments.

Each ablation's run.py imports from this module, monkey-patches the relevant
components, then calls run_eval().
"""
import sys, os, json, re, time
import numpy as np
from collections import defaultdict
from pathlib import Path
import traceback

from bert_score import BERTScorer

# ── hierarchyv2 base imports (will be patched per ablation) ──────
from hierarchyv2.trimem_ar_v2 import TriMemAR_v2
from hierarchyv2.llm_utils import call_llm, classify_question_type, judge
from hierarchyv2.knowledge_memory_v2 import vke_retrieve, classify_question_type_vke
from hierarchyv2.run_item_v2 import classify_error, PatternRegistry, _register_error_pattern, _log_km_failure, _strip_wrappers

# ── Data paths ──────────────────────────────────────────────────────
DATA_PATH = os.environ.get('DATA_PATH', 'data/locomo10_input_50.json')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')

_fast_vke_imported = False  # Will be set to True after VKE import

def ensure_vke_imported():
    """Lazy-import VKE functions that depend on hierarchyv2 package."""
    global _fast_vke_imported
    if not _fast_vke_imported:
        # This ensures compatibility with ablation setups
        _fast_vke_imported = True


def convert_locomo_item(session_item: dict) -> list[dict]:
    """Convert a single LoCoMo session (with nested QAs) into flat items."""
    conversation = session_item.get('conversation', {})
    qa_list = session_item.get('qa', [])

    all_sessions = []
    all_dates = []
    session_keys = sorted(
        [k for k in conversation.keys() if re.match(r'^session_\d+$', k) and not k.endswith('_date_time')],
        key=lambda k: int(k.split('_')[1])
    )
    for sk in session_keys:
        sess = conversation[sk]
        if isinstance(sess, list):
            normalized = []
            for turn in sess:
                if isinstance(turn, dict):
                    t = {}
                    t['speaker'] = turn.get('speaker', '')
                    t['content'] = turn.get('content', turn.get('text', ''))
                    t['role'] = turn.get('role', 'user')
                    normalized.append(t)
            if normalized:
                all_sessions.append(normalized)
                date_key = f'{sk}_date_time'
                if date_key in conversation:
                    all_dates.append(conversation[date_key])
                else:
                    all_dates.append('')

    flat_items = []
    for qa in qa_list:
        answer = qa.get('adversarial_answer', qa.get('answer', ''))
        item = {
            'question': qa['question'],
            'answer': answer,
            'question_type': '',
            'question_id': f"{session_item.get('sample_id','?')}",
            'haystack_sessions': all_sessions,
            'haystack_dates': all_dates,
        }
        flat_items.append(item)
    return flat_items


def answer_with_memory(q: str, cor: str, qt: str, mem: TriMemAR_v2, km_instructions: str = '',
                       registry: PatternRegistry = None, qid: str = '?',
                       use_vke: bool = True) -> dict:
    """
    Answer a question using a pre-built TriMemAR_v2 memory.
    
    Args:
        use_vke: If False, skip VKE retrieval (for w/o VKE ablation)
    """
    try:
        context = mem.search(q, question_type=qt)
        if not context:
            context = ['No relevant memories.']

        _counting_hint = _get_counting_hint(q)

        if qt == 'temporal-reasoning':
            ctx_str = '\n'.join([str(c)[:500] for c in context[:25]])
            if len(ctx_str) > 28000:
                ctx_str = ctx_str[:28000] + '...'
            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Be precise. '
                 'For temporal questions, use the EXACT date or time expression from context. '
                 'Do NOT add narrative wrappers.' + km_instructions},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=256) or ''
            ans = _strip_wrappers(ans.strip())
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or ''}
            if not result['correct'] and registry:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result

        elif qt == 'knowledge-update':
            ctx_str = '\n'.join([str(c)[:600] for c in context[:15]])
            if len(ctx_str) > 25000:
                ctx_str = ctx_str[:25000] + '...'
            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Find the most recent relevant fact.' + km_instructions + _counting_hint},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=256) or ''
            ans = _strip_wrappers(ans.strip())
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or ''}
            if not result['correct'] and registry:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result

        else:
            # single-session-user / entity-adversarial / single-session-preference
            vke_type = classify_question_type_vke(q)
            ctx_str = '\n'.join([str(c)[:600] for c in context[:20]])
            if len(ctx_str) > 24000:
                ctx_str = ctx_str[:24000] + '...'

            # VKE retrieval (optional for ablation). Need to pass mem.efg.
            if use_vke:
                try:
                    vke_results = vke_retrieve(q, vke_type, mem.efg if hasattr(mem, 'efg') else None)
                    if vke_results:
                        vke_context = '\n'.join([f"  [{r.get('predicate','?')}] {r.get('object','')}"
                                                 for r in vke_results[:10]]).strip()
                        if vke_context:
                            ctx_str = ctx_str + '\n\nRelevant facts:\n' + vke_context
                            if len(ctx_str) > 28000:
                                ctx_str = ctx_str[:28000] + '...'
                except Exception as e:
                    print(f'    [VKE warn] {e}', flush=True)

            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Be concise.'
                 ' Do NOT add narrative wrappers.' + km_instructions + _counting_hint},
                {'role': 'user', 'content': f'Context:\n{ctx_str}\n\nQ: {q}\nA:'}
            ], max_tokens=256) or ''
            ans = _strip_wrappers(ans.strip())
            result = {'correct': judge(q, ans, str(cor)), 'invalid': False, 'type': qt, 'predicted': ans or ''}
            if not result['correct'] and registry:
                error_type = classify_error(q, str(cor), ans)
                _register_error_pattern(q, str(cor), ans, error_type, registry)
                _log_km_failure(q, cor, ans, qt, qid)
            return result

    except Exception as e:
        traceback.print_exc()
        print(f'    [ERR] {e}', flush=True)
        return {'correct': False, 'invalid': True, 'type': qt, 'predicted': ''}


def _get_counting_hint(q: str) -> str:
    q_lower = q.lower()
    if any(w in q_lower for w in ['how many', 'count', 'how often', 'how many times']):
        return '\nCOUNTING INSTRUCTIONS: Extract the PRECISE count. Give just the number.'
    return ''


def init_scorer():
    import torch
    print('  [BERTScore] Initializing BERTScorer (roberta-large)...', flush=True)
    scorer = BERTScorer(
        lang='en',
        model_type='roberta-large',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        rescale_with_baseline=True,
    )
    return scorer


def compute_bertscore(scorer, cands, refs):
    safe_cands = [c if c and c.strip() else '[no prediction]' for c in cands]
    safe_refs = [r if r and r.strip() else '[empty reference]' for r in refs]
    P, R, F1 = scorer.score(safe_cands, safe_refs)
    return P.tolist(), R.tolist(), F1.tolist()


def run_eval(tag: str, build_memory_callback=None):
    """
    Run the full evaluation pipeline.
    
    Args:
        tag: Name for the experiment (used in output filenames)
        build_memory_callback: Optional hook to build/modify memory after construction.
            Signature: build_memory_callback(mem, sessions, dates) -> None
    """
    print('=' * 60, flush=True)
    print(f'  hierarchyv2 + BERTScore evaluation: {tag}', flush=True)
    print('=' * 60, flush=True)

    # ── Load data ────────────────────────────────────────────────
    with open(DATA_PATH) as f:
        raw_data = json.load(f)
    print(f'  Loaded {len(raw_data)} sessions from {DATA_PATH}', flush=True)

    scorer = init_scorer()

    all_results = []
    all_predictions = []
    all_references = []
    processed_count = 0

    for conv_idx, sess in enumerate(raw_data):
        conv_id = sess.get('sample_id', f'conv_{conv_idx}')
        items = convert_locomo_item(sess)
        n_qas = len(items)

        print(f'\n  {"─"*50}', flush=True)
        print(f'  Conv {conv_idx+1}/{len(raw_data)}: {conv_id} ({n_qas} QAs)', flush=True)
        print(f'  {"─"*50}', flush=True)

        sample_item = items[0]
        sessions = sample_item.get('haystack_sessions', [])
        dates = sample_item.get('haystack_dates', [])
        qt_sample = classify_question_type(sample_item['question'])
        chunk_enabled = (qt_sample in ['single-session-assistant', 'single-session-user'])

        # Build memory
        print(f'  Building memory...', flush=True)
        t0 = time.time()
        mem = TriMemAR_v2()
        mem.add_sessions(sessions, dates, chunk_enabled=chunk_enabled)

        # Apply ablation-specific modifications to memory
        if build_memory_callback:
            build_memory_callback(mem, sessions, dates)

        print(f'  Memory built in {time.time()-t0:.1f}s', flush=True)
        print(f'  Docs: {len(mem.raw_docs)}, facts: {len(mem.facts)}, personas: {len(mem.personas)}', flush=True)

        # Answer questions
        for idx, item in enumerate(items):
            q = item['question']
            cor = item['answer']
            qid = item.get('question_id', '?')

            print(f'  [{conv_idx+1}.{idx+1}/{n_qas}] {q[:60]}...', flush=True)

            qt = classify_question_type(q)
            registry = None
            km_instructions = ''

            # EDPL: PatternRegistry (can be skipped in w/o EDPL)
            try:
                registry = PatternRegistry()
                relevant_patterns = registry.get_relevant(q)
                for p in relevant_patterns[:3]:
                    km_instructions += '\n' + p.instruction
            except Exception:
                pass

            t1 = time.time()
            r = answer_with_memory(
                q, cor, qt, mem,
                km_instructions=km_instructions,
                registry=registry,
                qid=qid,
            )
            elapsed = time.time() - t1

            pred = r.get('predicted', '')
            is_correct = r.get('correct', False) and not r.get('invalid', False)

            all_predictions.append(pred)
            all_references.append(cor)
            all_results.append({
                'question': q,
                'reference': cor,
                'prediction': pred,
                'judge_correct': is_correct,
                'invalid': r.get('invalid', False),
                'type': r.get('type', qt),
            })

            status = '✅' if is_correct else '❌' if not r.get('invalid', False) else '⚠️'
            print(f'    {status} ({elapsed:.1f}s) {pred[:60] if pred else "[empty]"}', flush=True)
            processed_count += 1

        # Save progress
        _save_progress(tag, scorer, all_results, all_predictions, all_references)

    # Final results
    _final_results(tag, scorer, all_results, all_predictions, all_references)


def _save_progress(tag, scorer, results, predictions, references):
    md = _build_md(tag, results, intermediate=True)
    out_path = os.path.join(OUTPUT_DIR, f'result_{tag}.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(md)
    print(f'  [PROGRESS] Saved to {out_path}', flush=True)


def _final_results(tag, scorer, results, predictions, references):
    total_valid = sum(1 for r in results if not r['invalid'])
    correct_judge = sum(1 for r in results if r['judge_correct'] and not r['invalid'])
    judge_acc = round(correct_judge / total_valid * 100, 1) if total_valid else 0.0

    # BERTScore
    P_list, R_list, F1_list = compute_bertscore(scorer, predictions, references)
    avg_f1 = float(np.mean(F1_list))
    avg_p = float(np.mean(P_list))
    avg_r = float(np.mean(R_list))

    valid_indices = [i for i, r in enumerate(results) if not r['invalid']]
    valid_preds = [predictions[i] for i in valid_indices]
    valid_refs = [references[i] for i in valid_indices]
    if valid_preds:
        vP, vR, vF1 = compute_bertscore(scorer, valid_preds, valid_refs)
        avg_f1_valid = float(np.mean(vF1))
        avg_p_valid = float(np.mean(vP))
        avg_r_valid = float(np.mean(vR))
    else:
        avg_f1_valid = avg_p_valid = avg_r_valid = 0.0

    # Print summary
    print(f'\n{"="*60}', flush=True)
    print(f'  Experiment: {tag}', flush=True)
    print(f'  {"─"*40}', flush=True)
    print(f'  {"Total items":<25} {len(results):<8}', flush=True)
    print(f'  {"Valid items":<25} {total_valid:<8}', flush=True)
    print(f'  {"Judge Accuracy (%)":<25} {judge_acc:<8}', flush=True)
    print(f'  {"BERTScore F1 (all)":<25} {avg_f1:<8.4f}', flush=True)
    print(f'  {"BERTScore F1 (valid)":<25} {avg_f1_valid:<8.4f}', flush=True)
    print(f'  {"─"*40}', flush=True)

    # Per-type breakdown
    by_type = defaultdict(list)
    for i, r in enumerate(results):
        by_type[r['type']].append(i)

    print(f'\n  ── By question type ──', flush=True)
    for t, indices in sorted(by_type.items()):
        t_correct = sum(1 for i in indices if results[i]['judge_correct'] and not results[i]['invalid'])
        t_total = sum(1 for i in indices if not results[i]['invalid'])
        t_acc = round(t_correct / t_total * 100, 1) if t_total else 0
        t_f1 = float(np.mean([F1_list[i] for i in indices])) if indices else 0
        print(f'    {t:<25} acc={t_acc:>5.1f}%  F1={t_f1:.4f}  ({t_correct}/{t_total})', flush=True)

    # Build markdown
    md = _build_md(tag, results, intermediate=False)
    md += f"""
## BERTScore

| Metric | Value |
|--------|-------|
| BERTScore Avg P (all) | {avg_p:.4f} |
| BERTScore Avg R (all) | {avg_r:.4f} |
| **BERTScore Avg F1 (all)** | **{avg_f1:.4f}** |
| BERTScore Avg P (valid) | {avg_p_valid:.4f} |
| BERTScore Avg R (valid) | {avg_r_valid:.4f} |
| **BERTScore Avg F1 (valid)** | **{avg_f1_valid:.4f}** |

## Per-Item BERT-F1

| # | Question | Type | Judge | BERT-F1 |
|---|----------|------|-------|---------|
"""
    for i, r in enumerate(results):
        q_short = r['question'][:50].replace('|', '/') if r['question'] else ''
        judge_str = '✅' if r['judge_correct'] else ('⚠️' if r['invalid'] else '❌')
        md += f"| {i+1} | {q_short} | {r['type']} | {judge_str} | {F1_list[i]:.4f} |\n"

    md += "\n## Per-Type BERT-F1\n\n| Type | Items | Avg BERT-F1 |\n|------|-------|-------------|\n"
    for t, indices in sorted(by_type.items()):
        t_f1 = float(np.mean([F1_list[i] for i in indices])) if indices else 0
        md += f"| {t} | {len(indices)} | {t_f1:.4f} |\n"

    out_path = os.path.join(OUTPUT_DIR, f'result_{tag}.md')
    with open(out_path, 'w') as f:
        f.write(md)
    print(f'\n  Final results saved to {out_path}', flush=True)

    # Also update aaai_review.md with full ablation table
    _update_ablation_table(tag, judge_acc, avg_f1, avg_f1_valid)


def _update_ablation_table(tag, accuracy, f1_all, f1_valid):
    """Append this experiment's result to the running ablation table."""
    table_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    table = {}
    if os.path.exists(table_path):
        try:
            table = json.load(open(table_path))
        except Exception:
            pass
    table[tag] = {'accuracy': accuracy, 'f1_all': f1_all, 'f1_valid': f1_valid}
    json.dump(table, open(table_path, 'w'), indent=2)


def _build_md(tag, results, intermediate=False):
    total_valid = sum(1 for r in results if not r['invalid'])
    correct_judge = sum(1 for r in results if r['judge_correct'] and not r['invalid'])
    judge_acc = round(correct_judge / total_valid * 100, 1) if total_valid else 0.0

    by_type = defaultdict(list)
    for i, r in enumerate(results):
        by_type[r['type']].append(i)

    title = f"hierarchyv2 Ablation: {tag}"
    if intermediate:
        title += " (IN PROGRESS)"

    md = f"""# {title}

**Date:** 2026-07-27
**Data:** {DATA_PATH}
**Status:** {"In progress" if intermediate else "Complete"}

## Summary

| Metric | Value |
|--------|-------|
| Total items | {len(results)} |
| Valid items | {total_valid} |
| **Judge Accuracy** | **{judge_acc}%** |

## Per-Type Breakdown

| Type | Accuracy | Correct/Total |
|------|----------|---------------|
"""
    for t, indices in sorted(by_type.items()):
        t_correct = sum(1 for i in indices if results[i]['judge_correct'] and not results[i]['invalid'])
        t_total = sum(1 for i in indices if not results[i]['invalid'])
        t_acc = round(t_correct / t_total * 100, 1) if t_total else 0
        md += f"| {t} | {t_acc}% | {t_correct}/{t_total} |\n"

    md += """
## Per-Item Results

| # | Question | Type | Judge | Prediction | Reference |
|---|----------|------|-------|------------|-----------|
"""
    for i, r in enumerate(results):
        q_short = r['question'][:50].replace('|', '/') if r['question'] else ''
        pred_short = r['prediction'][:60].replace('|', '/') if r['prediction'] else '[empty]'
        ref_short = r['reference'][:60].replace('|', '/') if r['reference'] else '[empty]'
        judge_str = '✅' if r['judge_correct'] else ('⚠️' if r['invalid'] else '❌')
        md += f"| {i+1} | {q_short} | {r['type']} | {judge_str} | {pred_short} | {ref_short} |\n"

    return md
