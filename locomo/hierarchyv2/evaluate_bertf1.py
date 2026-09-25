#!/usr/bin/env python3
"""
evaluate_bertf1.py — Run hierarchyv2 on locomo50, compute judge accuracy + BERTScore F1

Optimized: builds memory once per conversation, then answers all questions.
"""
import sys, os, json, re, time
import numpy as np
from collections import defaultdict
from pathlib import Path
import traceback

# ── bert-score ────────────────────────────────────────────────────
from bert_score import BERTScorer

# ── hierarchyv2 imports ────────────────────────────────────────────
from hierarchyv2.run_item_v2 import run_item_v2, classify_error, PatternRegistry, _register_error_pattern, _log_km_failure, _strip_wrappers
from hierarchyv2.llm_utils import call_llm, classify_question_type, judge
from hierarchyv2.trimem_ar_v2 import TriMemAR_v2
from hierarchyv2.knowledge_memory_v2 import (
    vke_retrieve, vke_format_context,
    classify_question_type_vke,
    bayesian_confidence_calibrate,
    compute_keep_score, should_prune,
)

# ── Data path ──────────────────────────────────────────────────────
DATA_PATH = os.environ.get('DATA_PATH', 'data/locomo10_input_50.json')
OUTPUT_MD = os.environ.get('OUTPUT_MD', 'outputs/aaai_review.md')


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
                    t['role'] = 'user' if turn.get('speaker', '') == conversation.get('speaker_a', '') else 'assistant'
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
                       registry: PatternRegistry = None, qid: str = '?') -> dict:
    """Answer a question using a pre-built TriMemAR_v2 memory."""
    try:
        context = mem.search(q, question_type=qt)
        if not context:
            context = ['No relevant memories.']

        if qt == 'temporal-reasoning':
            vke_type = classify_question_type_vke(q)
            ctx_str = '\n'.join([str(c)[:500] for c in context[:25]])
            if len(ctx_str) > 28000:
                ctx_str = ctx_str[:28000] + '...'
            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Be precise. '
                 'For temporal questions, use the EXACT date or time expression from context. '
                 'If the context has a relative time expression, preserve it EXACTLY. '
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
            _counting_hint = ''
            if any(w in q.lower() for w in ['how many', 'count', 'how often', 'how many times']):
                _counting_hint = '\nCOUNTING INSTRUCTIONS: Extract the PRECISE count. ...'
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
            # single-session-user or others: use VKE retrieval + LLM
            vke_type = classify_question_type_vke(q)
            ctx_str = '\n'.join([str(c)[:600] for c in context[:20]])
            if len(ctx_str) > 24000:
                ctx_str = ctx_str[:24000] + '...'

            q_lower = q.lower()
            _counting_hint = ''
            if any(w in q_lower for w in ['how many', 'count', 'how often', 'how many times']):
                _counting_hint = '\nCOUNTING INSTRUCTIONS: ...'

            # VKE retrieval for facts
            vke_results = vke_retrieve(q, vke_type, mem.efg)
            if vke_results:
                vke_context = '\n'.join([f"  [{r.get('predicate','?')}] {r.get('object','')}"
                                         for r in vke_results[:10]]).strip()
                if vke_context:
                    ctx_str = ctx_str + '\n\nRelevant facts:\n' + vke_context
                    if len(ctx_str) > 28000:
                        ctx_str = ctx_str[:28000] + '...'

            ans = call_llm([
                {'role': 'system', 'content': 'Answer based ONLY on context. Be concise.'
                 ' If numbers are in context, use them precisely.'
                 ' If asked for names, lists, or items, list them.'
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


def init_scorer():
    """Initialize BERTScorer."""
    import torch
    print('  [BERTScore] Initializing BERTScorer (roberta-large)...', flush=True)
    scorer = BERTScorer(
        lang='en',
        model_type='roberta-large',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        rescale_with_baseline=True,
    )
    return scorer


def compute_bertscore(scorer, cands: list[str], refs: list[str]) -> tuple:
    """Compute BERTScore P, R, F1 for lists of candidate and reference strings."""
    safe_cands = [c if c and c.strip() else '[no prediction]' for c in cands]
    safe_refs = [r if r and r.strip() else '[empty reference]' for r in refs]
    P, R, F1 = scorer.score(safe_cands, safe_refs)
    return P.tolist(), R.tolist(), F1.tolist()


def main():
    print('=' * 60, flush=True)
    print('  hierarchyv2 + BERTScore evaluation on LoCoMo 50', flush=True)
    print('=' * 60, flush=True)

    # ── Load data ────────────────────────────────────────────────
    with open(DATA_PATH) as f:
        raw_data = json.load(f)
    print(f'  Loaded {len(raw_data)} sessions from {DATA_PATH}', flush=True)

    # ── Initialize BERTScore ──────────────────────────────────────
    scorer = init_scorer()

    # ── Process conversation by conversation ──────────────────────
    all_results = []
    all_predictions = []
    all_references = []

    for conv_idx, sess in enumerate(raw_data):
        conv_id = sess.get('sample_id', f'conv_{conv_idx}')
        qa_list = sess.get('qa', [])

        # Convert this conversation's data
        items = convert_locomo_item(sess)
        print(f'\n  {"─"*50}', flush=True)
        print(f'  Conversation {conv_idx+1}/10: {conv_id} ({len(items)} QAs)', flush=True)
        print(f'  {"─"*50}', flush=True)

        # Build memory once for this conversation
        print(f'  Building memory...', flush=True)
        t0 = time.time()

        sample_item = items[0]
        sessions = sample_item.get('haystack_sessions', [])
        dates = sample_item.get('haystack_dates', [])
        qt_sample = classify_question_type(sample_item['question'])
        chunk_enabled = (qt_sample in ['single-session-assistant', 'single-session-user'])

        mem = TriMemAR_v2()
        mem.add_sessions(sessions, dates, chunk_enabled=chunk_enabled)
        print(f'  Memory built in {time.time()-t0:.1f}s', flush=True)
        print(f'  Raw docs: {len(mem.raw_docs)}, facts: {len(mem.facts)}, personas: {len(mem.personas)}', flush=True)

        # Answer all questions for this conversation
        for idx, item in enumerate(items):
            q = item['question']
            cor = item['answer']
            qid = item.get('question_id', '?')

            print(f'  [{conv_idx+1}.{idx+1}] {q[:60]}...', flush=True)

            qt = classify_question_type(q)
            registry = PatternRegistry()
            relevant_patterns = registry.get_relevant(q)
            km_instructions = ''
            for p in relevant_patterns[:3]:
                km_instructions += '\n' + p.instruction

            t1 = time.time()
            r = answer_with_memory(q, cor, qt, mem, km_instructions, registry, qid)
            elapsed = time.time() - t1
            pred = r.get('predicted', '')
            is_correct = r.get('correct', False) and not r.get('invalid', False)
            is_invalid = r.get('invalid', False)

            all_predictions.append(pred)
            all_references.append(cor)
            all_results.append({
                'question': q,
                'reference': cor,
                'prediction': pred,
                'judge_correct': is_correct,
                'invalid': is_invalid,
                'type': r.get('type', qt),
            })

            status = '✅' if is_correct else '❌' if not is_invalid else '⚠️'
            print(f'    {status} ({elapsed:.1f}s) pred={pred[:80] if pred else "[empty]"}...', flush=True)

        # Save intermediate results after each conversation
        _save_progress(scorer, all_results, all_predictions, all_references)

    # ── Compute BERTScore ──────────────────────────────────────────
    print(f'\n  Computing BERTScore on {len(all_predictions)} items...', flush=True)
    P_list, R_list, F1_list = compute_bertscore(scorer, all_predictions, all_references)
    _finalize(all_results, P_list, R_list, F1_list, all_predictions, all_references, scorer)


def _save_progress(scorer, results, predictions, references):
    """Save intermediate results."""
    md = _build_md(results, predictions, references, intermediate=True)
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md)
    print(f'  [PROGRESS] Saved to {OUTPUT_MD}', flush=True)


def _finalize(results, P_list, R_list, F1_list, predictions, references, scorer):
    """Compute final metrics and save."""
    total_valid = sum(1 for r in results if not r['invalid'])
    correct_judge = sum(1 for r in results if r['judge_correct'] and not r['invalid'])

    avg_f1 = float(np.mean(F1_list)) if F1_list else 0.0
    avg_p = float(np.mean(P_list)) if P_list else 0.0
    avg_r = float(np.mean(R_list)) if R_list else 0.0

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

    judge_acc = round(correct_judge / total_valid * 100, 1) if total_valid else 0.0

    print(f'\n{"="*60}', flush=True)
    print(f'  {"Metric":<30} {"Value":<10}', flush=True)
    print(f'  {"─"*40}', flush=True)
    print(f'  {"Total items":<30} {len(results):<10}', flush=True)
    print(f'  {"Valid items":<30} {total_valid:<10}', flush=True)
    print(f'  {"Judge Accuracy (%)":<30} {judge_acc:<10}', flush=True)
    print(f'  {"BERTScore Avg P (all)":<30} {avg_p:<10.4f}', flush=True)
    print(f'  {"BERTScore Avg R (all)":<30} {avg_r:<10.4f}', flush=True)
    print(f'  {"BERTScore Avg F1 (all)":<30} {avg_f1:<10.4f}', flush=True)
    print(f'  {"BERTScore Avg P (valid)":<30} {avg_p_valid:<10.4f}', flush=True)
    print(f'  {"BERTScore Avg R (valid)":<30} {avg_r_valid:<10.4f}', flush=True)
    print(f'  {"BERTScore Avg F1 (valid)":<30} {avg_f1_valid:<10.4f}', flush=True)
    print(f'{"="*60}', flush=True)

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

    # Build initial markdown
    md = _build_md(results, predictions, references, intermediate=False)

    # Append BERTScore section
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
"""

    md += """## Per-Item BERT-F1

| # | Question | Type | Judge | BERT-F1 |
|---|----------|------|-------|---------|
"""
    for i, r in enumerate(results):
        q_short = r['question'][:50].replace('|', '/') if r['question'] else ''
        judge_str = '✅' if r['judge_correct'] else ('⚠️' if r['invalid'] else '❌')
        md += f"| {i+1} | {q_short} | {r['type']} | {judge_str} | {F1_list[i]:.4f} |\n"

    md += """
## Per-Type BERT-F1

| Type | Items | Avg BERT-F1 |
|------|-------|-------------|
"""
    by_type = defaultdict(list)
    for i, r in enumerate(results):
        by_type[r['type']].append(i)
    for t, indices in sorted(by_type.items()):
        t_f1 = float(np.mean([F1_list[i] for i in indices])) if indices else 0
        md += f"| {t} | {len(indices)} | {t_f1:.4f} |\n"

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md)
    print(f'\n  Results saved to {OUTPUT_MD}', flush=True)


def _build_md(results, predictions, references, intermediate=False):
    """Build markdown content."""
    total_valid = sum(1 for r in results if not r['invalid'])
    correct_judge = sum(1 for r in results if r['judge_correct'] and not r['invalid'])
    judge_acc = round(correct_judge / total_valid * 100, 1) if total_valid else 0.0

    by_type = defaultdict(list)
    for i, r in enumerate(results):
        by_type[r['type']].append(i)

    title = "hierarchyv2 Evaluation on LoCoMo 50 (IN PROGRESS)" if intermediate else "hierarchyv2 Evaluation on LoCoMo 50"

    md = f"""# {title}

**Date:** 2026-07-27
**Data:** {DATA_PATH}
**API:** DeepSeek Chat
**Status:** {"In progress" if intermediate else "Complete"}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total items | {len(results)} |
| Valid items | {total_valid} |
| **Judge Accuracy** | **{judge_acc}%** |

## Per-Question-Type Breakdown

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

| # | Question | Type | Judge | Prediction (truncated) | Reference (truncated) |
|---|----------|------|-------|------------------------|----------------------|
"""
    for i, r in enumerate(results):
        q_short = r['question'][:50].replace('|', '/') if r['question'] else ''
        pred_short = r['prediction'][:60].replace('|', '/') if r['prediction'] else '[empty]'
        ref_short = r['reference'][:60].replace('|', '/') if r['reference'] else '[empty]'
        judge_str = '✅' if r['judge_correct'] else ('⚠️' if r['invalid'] else '❌')
        md += f"| {i+1} | {q_short} | {r['type']} | {judge_str} | {pred_short} | {ref_short} |\n"

    return md

if __name__ == '__main__':
    main()
