#!/usr/bin/env python3
"""
hierarchy_graph.py — Lightweight System-2 hierarchy retrieval (Mnemis-inspired)

Builds a 3-level semantic hierarchy from session data:
  Level 0: Entities (person names, places, objects)
  Level 1: Event Types (action categories)
  Level 2: Time Context (year/month buckets)

At retrieval time, System-2 traverses top-down: time → event → entity → sessions.
Results merge with System-1 (BGE) for dual-route retrieval.
"""
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Set, Tuple

# ─── Event type classification ─────────────────────────────────
_EVENT_KEYWORDS = {
    'outdoor_activity': {
        'camping', 'hiking', 'fishing', 'kayaking', 'swimming', 'biking',
        'walk', 'run', 'running', 'jogging', 'climbing', 'surfing',
        'beach', 'park', 'travel', 'trip', 'road trip', 'camp',
    },
    'social_event': {
        'meetup', 'party', 'club', 'convention', 'parade', 'festival',
        'concert', 'gathering', 'celebration', 'pride', 'community',
        'meeting', 'conference', 'workshop', 'seminar',
    },
    'family_life': {
        'adoption', 'child', 'daughter', 'birthday', 'baby', 'kids',
        'family', 'parent', 'mother', 'father', 'sister', 'brother',
        'marry', 'married', 'wedding', 'move in', 'together',
    },
    'health_fitness': {
        'yoga', 'meditation', 'gym', 'exercise', 'fitness', 'diet',
        'weight', 'health', 'heart', 'hospital', 'doctor', 'therapy',
        'palpitation', 'pain', 'hurt', 'injury',
    },
    'work_business': {
        'job', 'IT', 'project', 'store', 'business', 'work', 'office',
        'career', 'promotion', 'interview', 'startup', 'company',
        'networking', 'client', 'meeting', 'convention', 'conference',
    },
    'creative_arts': {
        'paint', 'painting', 'drawing', 'sculpture', 'music', 'sing',
        'dance', 'writing', 'poetry', 'photography', 'art', 'design',
        'pottery', 'craft',
    },
    'food_dining': {
        'cook', 'cooking', 'bake', 'baking', 'recipe', 'restaurant',
        'dinner', 'lunch', 'breakfast', 'coffee', 'ice cream',
        'vegan', 'meal', 'food',
    },
    'entertainment': {
        'game', 'gaming', 'video game', 'movie', 'film', 'tv', 'show',
        'concert', 'tournament', 'valorant', 'music', 'performance',
        'book', 'reading', 'podcast',
    },
    'education_learning': {
        'class', 'course', 'lesson', 'workshop', 'learn', 'study',
        'training', 'school', 'college', 'university', 'tutoring',
    },
    'travel_places': {
        'travel', 'trip', 'vacation', 'visit', 'tour', 'flight',
        'hotel', 'airport', 'road trip', 'canada', 'tokyo', 'rome',
        'banff', 'toronto', 'vancouver', 'new york', 'chicago',
    },
}

# ─── Month/season extractor ────────────────────────────────────
_MONTH_NAMES = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
_SEASON_MONTHS = {'spring': [3,4,5], 'summer': [6,7,8], 'fall': [9,10,11], 'winter': [12,1,2]}


class HierarchyGraph:
    """Lightweight 3-level semantic hierarchy for System-2 retrieval."""

    def __init__(self):
        # Level 0: Entity → session indices
        self.entity_sessions: Dict[str, Set[int]] = defaultdict(set)
        # Level 1: Event type → session indices  
        self.event_sessions: Dict[str, Set[int]] = defaultdict(set)
        # Level 2: Time bucket → session indices
        self.time_sessions: Dict[str, Set[int]] = defaultdict(set)
        # Reverse: session → event types
        self.session_events: Dict[int, Set[str]] = defaultdict(set)
        # Session → year
        self.session_year: Dict[int, int] = {}
        # Session → speakers
        self.session_speakers: Dict[int, Set[str]] = defaultdict(set)

    def build(self, sessions: list, dates: list):
        """Build hierarchy from conversation sessions."""
        for si, (session, date) in enumerate(zip(sessions, dates)):
            if not isinstance(session, list):
                continue
            full_text = ' '.join([
                t.get('content', '') for t in session if isinstance(t, dict)
            ])
            lower_text = full_text.lower()

            # Extract speakers (Level 0: entities)
            speakers = set(re.findall(r'\[([A-Za-z]+)\]', full_text))
            self.session_speakers[si] = {s.lower() for s in speakers}
            for sp in speakers:
                self.entity_sessions[sp.lower()].add(si)

            # Extract named entities (capitalized words that aren't common)
            names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', full_text))
            _common = {'How','What','When','Where','Why','Which','Who','The','This',
                       'That','These','Those','My','Your','His','Her','Its','Our',
                       'Their','Me','You','He','She','It','We','They','Hey','Hi',
                       'Hello','Hi','Oh','No','Yes','Ok','Okay','Sure','Great',
                       'Nice','Good','Well','Right','Now','Just','Also','Very',
                       'Really','One','Two','Three','Four','Five','Six','Seven',
                       'Eight','Nine','Ten','Today','Yesterday','Tomorrow'}
            for name in names:
                if name not in _common and len(name) >= 3:
                    self.entity_sessions[name.lower()].add(si)

            # Classify event type (Level 1)
            for event_type, keywords in _EVENT_KEYWORDS.items():
                for kw in keywords:
                    if kw in lower_text:
                        self.event_sessions[event_type].add(si)
                        self.session_events[si].add(event_type)
                        break

            # Time context (Level 2)
            d_str = str(date)[:10] if date else ''
            if len(d_str) >= 4:
                yr = d_str[:4]
                self.time_sessions[f'year_{yr}'].add(si)
                self.session_year[si] = int(yr)
                if len(d_str) >= 7:
                    self.time_sessions[f'month_{yr}_{int(d_str[5:7]):02d}'].add(si)

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """System-2 retrieval: traverse hierarchy top-down."""
        q_lower = query.lower()
        session_scores = defaultdict(float)

        # ─── Level 2: Time matching ───
        yr_match = re.search(r'\b(202\d)\b', q_lower)
        if yr_match:
            target_yr = yr_match.group(1)
            for si in self.time_sessions.get(f'year_{target_yr}', set()):
                session_scores[si] += 3.0

            # Check month
            for mn, mv in _MONTH_NAMES.items():
                if mn in q_lower:
                    for si in self.time_sessions.get(f'month_{target_yr}_{mv:02d}', set()):
                        session_scores[si] += 5.0
                    break

            # Season
            for season, months in _SEASON_MONTHS.items():
                if season in q_lower:
                    for m in months:
                        for si in self.time_sessions.get(f'month_{target_yr}_{m:02d}', set()):
                            session_scores[si] += 4.0
                    break

        # ─── Level 1: Event type matching ───
        matched_events = set()
        for event_type, keywords in _EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    matched_events.add(event_type)
                    for si in self.event_sessions.get(event_type, set()):
                        session_scores[si] += 4.0
                    break

        # ─── Level 0: Entity matching ───
        names = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
        _common_ents = {'How','What','When','Where','Why','Which','Who','Whom',
            'The','This','That','These','Those','My','Your','His','Her','Its',
            'Our','Their','Me','You','He','She','It','We','They','No','Yes',
            'Can','Will','May','Not','For','With','Was','Were','Have','Has','Had'}
        for name in names:
            if name not in _common_ents:
                n_lower = name.lower()
                for si in self.entity_sessions.get(n_lower, set()):
                    session_scores[si] += 3.0

        # ─── Also check query keywords against event types ───
        q_words = set(re.findall(r'[a-zA-Z]{3,}', q_lower))
        _stop = {'how','what','when','where','why','which','who','whom','whose',
            'the','a','an','this','that','these','those','my','your','his','her',
            'its','our','their','me','you','he','she','it','we','they','no','yes',
            'on','in','at','to','for','with','by','from','of','do','does','did',
            'was','were','has','have','had','be','been','being','am','is','are',
            'can','will','may','not','but','and','or','if','else','into','about',
            'up','out','then','than','now','just','also','very','many','much',
            'all','any','some','every','each','big','new','take','took','doing',
            'spend','spent','plan','day','days','activity','moment','trip','travel',
            'back','going','go','went','come','came','get','got','know','like',
            'time','thing','things','year','month','week','said','tell','told',
            'make','made','see','look','first','last','ago',
            'january','february','march','april','may','june','july','august',
            'september','october','november','december'}
        q_words -= _stop

        # Score by keyword overlap with session text
        # (This is already handled by BGE in System-1, 
        #  so we keep it light here)

        # Sort by score
        ranked = sorted(session_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def get_hierarchy_debug(self, query: str) -> str:
        """Return debug info about what the hierarchy found."""
        q_lower = query.lower()
        parts = []

        yr_match = re.search(r'\b(202\d)\b', q_lower)
        if yr_match:
            yr = yr_match.group(1)
            count = len(self.time_sessions.get(f'year_{yr}', set()))
            parts.append(f'year={yr}({count}sessions)')
            for mn, mv in _MONTH_NAMES.items():
                if mn in q_lower:
                    count = len(self.time_sessions.get(f'month_{yr}_{mv:02d}', set()))
                    parts.append(f'month={mn}({count})')
                    break

        for event_type, keywords in _EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    count = len(self.event_sessions.get(event_type, set()))
                    parts.append(f'event={event_type}({count})')
                    break

        names = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
        _common_ents = {'How','What','When','Where','Why','Which','Who','Whom',
            'The','This','That','These','Those','My','Your','His','Her','Its',
            'Our','Their','Me','You','He','She','It','We','They','No','Yes',
            'Can','Will','May','Not','For','With','Was','Were'}
        for name in names:
            if name not in _common_ents:
                count = len(self.entity_sessions.get(name.lower(), set()))
                parts.append(f'entity={name}({count})')

        return ' | '.join(parts) if parts else 'no hierarchy matches'
