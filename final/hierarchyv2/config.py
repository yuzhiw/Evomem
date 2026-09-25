#!/usr/bin/env python3
"""
config.py — Configuration and global variables for TriMem-AR
"""
import sys, json, os, re, time, hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_BASE = 'https://api.deepseek.com/v1'
EMBED_PATH = os.environ.get('EMBED_PATH', 'BAAI/bge-small-en-v1.5')
DATA_PATH = os.environ.get('DATA_PATH', 'data/locomo10_input_50.json')
KM_PATH = os.environ.get('KM_PATH', 'knowledge_memory.json')

if API_KEY:
    os.environ['OPENAI_API_KEY'] = API_KEY
from openai import OpenAI; _oai = OpenAI(api_key=API_KEY, base_url=API_BASE)

_EMB = None; _TOK = None

_STOP = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our', 'you', 'your', 'i', 'me', 'my', 'what', 'where', 'when', 'why', 'how', 'who', 'which', 'not', 'no', 'nor', 'just', 'only', 'also', 'very', 'really', 'about', 'into', 'over', 'after', 'before', 'between',
    # 2026-08-27: EDPL 注册词质量——泛时序/泛疑问词不进 trigger（曾导致 now/current/value/job 错乱模式）
    'now', 'current', 'currently', 'value', 'values', 'changed', 'change', 'changes',
    'observed', 'list', 'different', 'time', 'state', 'since', 'any', 'some', 'new', 'old', 'first', 'last'}
_MS_COMMON_NAMES = {'How','What','When','Where','Why','Which','Who','Whom','Whose','The','A','An','This','That','These','Those','My','Your','His','Her','Its','Our','Their','Me','You','He','She','It','We','They','No','Yes','True','False','January','February','March','April','May','June','July','August','September','October','November','December','Summer','Winter','Spring','Autumn'}

_COMPUTE_SPECIFICITY_STOP = {
    'the','a','an','is','was','were','are','be','been','being','have','has','had',
    'do','does','did','done','get','got','gotten','make','made','go','went','gone',
    'say','said','see','saw','seen','come','came','take','took','taken','give',
    'gave','given','tell','told','ask','asked','use','used','using','want','wanted',
    'will','would','can','could','shall','should','may','might','must','need',
    'like','just','also','very','really','quite','too','much','many','some','any',
    'all','both','each','few','more','most','other','such','only','even','still',
    'already','yet','now','then','here','there','this','that','these','those',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your',
    'his','its','our','their','mine','yours','hers','its','ours','theirs',
    'not','no','nor','and','or','but','if','because','so','than','as','for',
    'with','about','into','over','after','before','between','through','during',
    'of','to','in','on','at','by','from','up','out','off','down','under','again',
    'further','once','well','back','very','dear','hi','hey','hello','bye','oh',
    'yeah','yes','ok','okay','sure','great','nice','good','bad'
}
