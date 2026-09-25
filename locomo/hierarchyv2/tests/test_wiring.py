#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_wiring.py — 验证 2026-09-14 接线：写好的代码块是否真的在主路径被调用。

覆盖：
 1. _inject_vke 去重
 2. _vke_context_block 的四个调用条件（开关 / 题型 / 版本链存在 / 去重）
 3. _hbs_weights_for：EDPL error_type → adapt_weights → km.json 持久化
 4. _apply_postproc：只在 temporal-reasoning 或 when 类 SSU/SSA 触发
 5. _ms_counting_fallback：只在计数题且主答案无数字时调用 MultiSessionEngine

全部用 stub/monkeypatch，不发任何 API 请求。
运行（在 code/ 目录下）：python3 -m pytest tests/test_wiring.py -q
"""
import os, sys, json, tempfile

# 官方代码目录被 import 为包名 hierarchyv2（与代码内部绝对导入一致）
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.environ.get('EVOMEM_PKG_PARENT') or tempfile.mkdtemp(prefix='evomem_pkg_')
_LINK = os.path.join(_PKG_PARENT, 'hierarchyv2')
if not os.path.exists(_LINK):
    os.symlink(_CODE_DIR, _LINK)
sys.path.insert(0, _PKG_PARENT)

_TMP = tempfile.mkdtemp(prefix='wiretest_')
os.environ['KM_PATH'] = os.path.join(_TMP, 'km.json')

from hierarchyv2 import run_item_v2 as riv2                      # noqa: E402
from hierarchyv2.trimem_ar_v2 import TriMemAR_v2                 # noqa: E402
from hierarchyv2.knowledge_memory_v2 import DEFAULT_WEIGHTS      # noqa: E402


class _Pat:
    def __init__(self, et):
        self.error_type = et
        self.id = 'p_' + et
        self.confidence = 0.9
        self.triggers = ['job']
        self.is_active = True


def _mem_with_chain():
    mem = TriMemAR_v2()
    for t, o in [('2023-01-05', 'barista'), ('2023-06-01', 'dance studio owner'),
                 ('2024-02-01', 'dance studio owner')]:
        mem.efg.add_fact({'subject': 'caroline', 'predicate': 'person_role', 'object': o,
                          'time': t, 'session_id': 'session_0', 'confidence': 'high'})
    mem.qdate_dt = None
    return mem


def test_inject_vke_dedup():
    assert riv2._inject_vke('CTX', 'BLOCK') == 'BLOCK\n\nCTX'
    assert riv2._inject_vke('CTX', '') == 'CTX'
    assert riv2._inject_vke('[ENTITY-FACT GRAPH (VKE)]\nold', 'BLOCK') == '[ENTITY-FACT GRAPH (VKE)]\nold'


def test_vke_block_conditions(monkeypatch):
    mem = _mem_with_chain()
    q = "What is Caroline's job now?"
    called = {}

    def _fake_retrieve(query, vtype, efg, weights=None, now=None):
        called['n'] = called.get('n', 0) + 1
        return [{'predicate': 'person_role', 'object': 'dance studio owner', 'time': '2024-02-01'}]

    monkeypatch.setattr(riv2, 'vke_retrieve', _fake_retrieve)
    monkeypatch.setattr(riv2, 'vke_format_context', lambda nodes, t: '[VERSIONED KNOWLEDGE (VKE)]\n  fake')

    assert riv2._vke_context_block(q, 'single-session-user', mem, []) == ''
    assert 'n' not in called
    assert riv2._vke_context_block('What is Zebulon doing?', 'knowledge-update', mem, []) == ''
    assert 'n' not in called
    blk = riv2._vke_context_block(q, 'knowledge-update', mem, [])
    assert called.get('n') == 1 and blk.startswith('[VERSIONED KNOWLEDGE (VKE)]')
    monkeypatch.setattr(riv2, 'WIRE_VKE', False)
    for _ in range(2):
        assert riv2._vke_context_block(q, 'knowledge-update', mem, []) == ''
    assert called.get('n') == 1


def test_hbs_weights_update_and_persist():
    w = riv2._hbs_weights_for('KU', [_Pat('reasoning_error')], 'q1')
    assert tuple(w['KU']) == (0.25, 0.35, 0.25, 0.20), w['KU']
    saved = json.load(open(os.environ['KM_PATH']))
    assert 'hbs_weights' in saved and list(saved['hbs_weights']['KU']) == [0.25, 0.35, 0.25, 0.2]
    w2 = riv2._hbs_weights_for('TR', [], 'q2')
    assert tuple(w2['TR']) == tuple(DEFAULT_WEIGHTS['TR'])
    w3 = riv2._hbs_weights_for('MS', [_Pat('unknown_type')], 'q3')
    assert tuple(w3['MS']) == tuple(DEFAULT_WEIGHTS['MS'])


def test_postproc_conditions(monkeypatch):
    hits = []

    class _PP:
        def apply(self, ans, q, qt, sessions, dates, ctx_str):
            hits.append(qt)
            return ans + ' [PP]'

    monkeypatch.setattr(riv2, '_postproc_engine', lambda: _PP())
    a = riv2._apply_postproc('May 3, 2023', 'When did X?', 'temporal-reasoning', [], [], 'ctx')
    assert a.endswith('[PP]') and hits == ['temporal-reasoning']
    a = riv2._apply_postproc('May 3, 2023', 'When did X?', 'single-session-user', [], [], 'ctx')
    assert a.endswith('[PP]') and hits[-1] == 'temporal-reasoning'
    before = len(hits)
    riv2._apply_postproc('x', 'What is X?', 'single-session-user', [], [], 'ctx')
    riv2._apply_postproc('x', 'What is X now?', 'knowledge-update', [], [], 'ctx')
    assert len(hits) == before
    monkeypatch.setattr(riv2, 'WIRE_POSTPROC', False)
    assert riv2._apply_postproc('a', 'When did X?', 'temporal-reasoning', [], [], 'ctx') == 'a'


def test_ms_counting_fallback(monkeypatch):
    calls = []

    class _MSE:
        def add_sessions(self, sessions, dates, km_patterns=None, chunk_enabled=True):
            calls.append('add')

        def answer(self, q, graph_hint=''):
            return 'TOTAL: 4'

    monkeypatch.setattr(riv2, 'MultiSessionEngine', _MSE)
    assert riv2._ms_counting_fallback('3', 'How many X did I buy?', [], []) == '3'
    assert calls == []
    assert riv2._ms_counting_fallback('I am not sure', 'How many X did I buy?', [], []) == 'TOTAL: 4'
    assert calls == ['add']
    assert riv2._ms_counting_fallback('', 'What is X?', [], []) == ''
    monkeypatch.setattr(riv2, 'WIRE_MS', False)
    assert riv2._ms_counting_fallback('', 'How many X did I buy?', [], []) == ''


if __name__ == '__main__':
    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)
            self._undo = []

    test_inject_vke_dedup(); print('PASS test_inject_vke_dedup')
    for fn in [test_vke_block_conditions, test_postproc_conditions, test_ms_counting_fallback]:
        mp = _MP(); fn(mp); mp.undo(); print('PASS', fn.__name__)
    test_hbs_weights_update_and_persist(); print('PASS test_hbs_weights_update_and_persist')
    print('ALL WIRING TESTS PASSED')
