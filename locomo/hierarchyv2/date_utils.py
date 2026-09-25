#!/usr/bin/env python3
"""
date_utils.py — Date parsing and formatting functions
"""
import re
from datetime import datetime, timedelta

# ─── Date parsing ─────────────────────────────────────────────
def parse_date(s):
    if not s: return None
    c = s.split('(')[0].strip() if '(' in s else s
    for f in ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d']:
        try: return datetime.strptime(c, f)
        except: pass
    # LoCoMo: "25 May, 2023" → datetime
    m = re.match(r'(\d{1,2})\s+(\w+),?\s*(\d{4})', c)
    if m:
        d_str, mm_str, y_str = m.groups()
        month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
                     'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
        mo = month_map.get(mm_str.capitalize())
        if mo:
            return datetime(int(y_str), mo, int(d_str))
    return None

def fmt_date(dt):
    return dt.strftime('%Y-%m-%d') if dt else ''
