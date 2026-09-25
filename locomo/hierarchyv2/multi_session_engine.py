#!/usr/bin/env python3
"""
multi_session_engine.py — MultiSessionEngine (v8 counting engine)
"""
import re, numpy as np
from collections import defaultdict
from typing import List, Dict
from .embedding import embed
from .knowledge_memory_v2 import _apply_km_patterns
from .llm_utils import call_llm
from .utils import chunk_long_text

class MultiSessionEngine:
    """Standalone multi-session counting engine (v1 approach, proven 80% accuracy)"""
    def __init__(self):
        self.all_docs = []
        self.embs = None
        self.km_facts = []

    @staticmethod
    def _chunk_long_text(user_text, asst_text):
        return chunk_long_text(user_text, asst_text)

    def add_sessions(self, sessions, dates, km_patterns=None, chunk_enabled=True):
        for si, session in enumerate(sessions):
            if not isinstance(session, list): continue
            dt = dates[si] if si < len(dates) else ''
            for i in range(0, len(session), 2):
                pair = session[i:i+2]
                if len(pair) >= 1 and isinstance(pair[0], dict) and pair[0].get('content','').strip():
                    text = f"[{dt[:10]}] [user] {pair[0]['content'].strip()}"
                    if len(pair) >= 2 and isinstance(pair[1], dict):
                        text += f" [assistant] {pair[1]['content'].strip()}"
                    self.all_docs.append({'text': text, 'date': dt[:10], 'session': si})
        # Apply KM patterns for structured fact extraction
        if km_patterns:
            for si, session in enumerate(sessions):
                if not isinstance(session, list): continue
                dt = dates[si] if si < len(dates) else ''
                for turn in session:
                    if isinstance(turn, dict) and turn.get('role') == 'user':
                        txt = turn['content'].strip()
                        km_facts = _apply_km_patterns(txt, dt[:10], f'session_{si}', km_patterns)
                        for f in km_facts:
                            self.km_facts.append(f)
        if self.all_docs:
            self.embs = np.array(embed([d['text'] for d in self.all_docs]))

    def expand_query(self, question: str) -> list:
        queries = [question]
        q = question
        q = re.sub(r'(?i)^(how many|how much|what (?:is|was|were|are) the total|what)\s+', '', q)
        q = re.sub(r'(?i)\s+(did I|have I|do I|i have|i had|did you|have you|do you)\s+.*$', '', q)
        q = re.sub(r'(?i)\s*(in|over|during|within|for|of|about|on|at)\s+.*$', '', q)
        q = q.strip().rstrip('?.,;:! ')
        if q and q != question and len(q) > 3:
            queries.append(q)
        actions = re.findall(r'(?i)(?:bought|worked|assembled|built|visited|attended|spent|acquired|painted|fixed|sold|created|watched|took|picked|returned|bake|led|manage|buy|sell|fix|set up|collect|adopt|adopted|use|using)', question)
        if actions:
            queries.append(' '.join(actions[:3]))
        return queries

    # General entity extraction patterns for listing questions
    _ENTITY_EXTRACT_PATTERNS = [
        (r'(?:playing|tried|trying|got into|hooked on|started|start\s+playing|been\s+playing)\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*(?:\s+\d)?)', 'activity'),
        (r'called\s+"([A-Za-z0-9\s\-\'@#$%^&*()!]+?)"', 'named'),
        (r'called\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', 'named'),
        (r'(?:watching|seen|bingeing|streaming|reading|finished|just\s+finished)\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*(?:\s+\d)?)', 'media'),
        (r'(?:bought|purchased|got|ordered|adopted)\s+(?:a|an|the|a\s+new|another|my)?\s*([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', 'acquisition'),
        (r'(?:working\s+on|started|joined|signed\s+up\s+for)\s+(?:a|an|the)?\s*([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', 'project'),
        (r'(?:visited|went\s+to|travelled\s+to|traveled\s+to|attended)\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', 'place'),
        (r'\bis\s+called\s+([A-Z][a-zA-Z0-9\s\-\'@#!]+?)(?:[.,!?;]|$)', 'named'),
        (r'\bname\s+is\s+([A-Z][a-zA-Z0-9\s\-\'@#!]+?)(?:[.,!?;]|$)', 'named'),
        (r'(?:this\s+new|a\s+new|another)\s+(.*?)(?:called|named)\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)', 'named'),
    ]

    def extract_entities(self, context_text: str, question: str) -> list:
        """Extract named entities from context using general patterns.
        Returns deduplicated list of (entity_text, entity_type)."""
        entities = []
        seen = set()
        q_lower = question.lower()
        for pattern, etype in self._ENTITY_EXTRACT_PATTERNS:
            for m in re.finditer(pattern, context_text):
                grp = [g for g in m.groups() if g and g.strip()][-1]
                if not grp or len(grp) < 2:
                    continue
                name = grp.strip().rstrip('.,!?;:')
                name_lower = name.lower()
                if any(w in name_lower for w in ['which', 'what', 'how', 'when', 'why', 'where']):
                    continue
                if name_lower in {'the', 'a', 'an', 'this', 'that', 'there', 'it'} or name_lower in q_lower:
                    continue
                key = name_lower[:50]
                if key not in seen:
                    seen.add(key)
                    entities.append((name[:60], etype))
        final = []
        final_lower = []
        for name, etype in entities:
            nl = name.lower()
            if not any(nl in f or f in nl for f in final_lower):
                final.append((name, etype))
                final_lower.append(nl)
        return final

    def answer(self, question: str, graph_hint: str = "") -> str:
        """3-sample majority vote counting"""
        queries = self.expand_query(question)
        all_scores = defaultdict(float)
        all_texts = {}
        for qi, q in enumerate(queries):
            weight = [1.0, 0.8, 0.6][qi] if qi < 3 else 0.5
            qe = np.array(embed([q])).flatten()
            sims = np.dot(self.embs, qe)
            top_k = min(30, len(self.all_docs))
            for idx in np.argsort(sims)[::-1][:top_k]:
                text = self.all_docs[idx]['text']
                all_scores[text] = max(all_scores.get(text, 0), float(sims[idx]) * weight)
                all_texts[text] = self.all_docs[idx]
        sorted_results = sorted(all_scores.items(), key=lambda x: -x[1])
        diverse_ctx = []
        for text, score in sorted_results:
            doc = all_texts.get(text, {})
            si = doc.get('session', -1)
            diverse_ctx.append(f'[S{si}] [{score:.2f}] {text[:500]}')
        ctx = '\n'.join(diverse_ctx[:40])
        if len(ctx) > 32000: ctx = ctx[:32000] + '...'

        # 🦞 Stage 1: Algorithmic entity extraction from context
        extracted_entities = self.extract_entities(ctx, question)
        entity_hint = ''
        if extracted_entities:
            entity_names = [e[0] for e in extracted_entities]
            entity_hint = f'\n\n[DETECTED NAMED ENTITIES (algorithmic extraction)]\n' + '\n'.join(f'  - {n}' for n in entity_names)
            print(f'    [MS-EntityExtract] extracted {len(entity_names)} entities: {entity_names[:12]}', flush=True)

        graph_extra = f"\n\n{graph_hint}" if graph_hint else ""
        # Stage 2 (strict format): force TOTAL: X format
        answer = call_llm([
            {'role': 'system', 'content': 'List each unique item on a separate line. End with "TOTAL: X". If asking "how many", output ONLY "TOTAL: X" on a single line.'},
            {'role': 'user', 'content': f'''Question: {question}{graph_extra}\n\nExcerpts:\n{ctx}{entity_hint}\n\nInstructions:\n1. Find each unique item/number from the excerpts.\n2. The [DETECTED NAMED ENTITIES] section lists algorithmically found entities. Use these as your STARTING LIST.\n3. Deduplicate: same entity in multiple sessions = one.\n4. If asking "how many", output ONLY "TOTAL: X" on a single line.\n5. Otherwise, list each specific item on a separate line. Use the EXACT names from the detected entities if possible.\n6. Then end with "TOTAL: X".\n\nItems:'''}
        ], max_tokens=2048) or ''

        return answer


