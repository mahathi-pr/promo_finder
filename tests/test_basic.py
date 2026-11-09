import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from promo_finder.parse import load_all
from promo_finder.pipeline import run
from promo_finder.validate import v2_constraints
from promo_finder.llm import mock, parse_json_list
from promo_finder.score import HAS, REVEAL
SCR = os.environ.get("SCRIPTS", "data/scripts_ignored_do_not_publish")

def test_word_boundaries():
    assert HAS(REVEAL, "I determine the reason") == 0 and HAS(REVEAL, "my mother") == 1

def test_parse_counts():
    eps = load_all(SCR); assert len(eps) == 50 and all(e.events for e in eps)

def test_heuristic_constraints():
    eps, picks, _ = run(SCR); assert picks and not v2_constraints(eps, picks)

def test_llm_plumbing_with_mock_and_failure_fallback():
    eps, picks, _ = run(SCR, llm=mock, log=lambda s: None); assert picks and not v2_constraints(eps, picks)
    def boom(*a, **k): raise RuntimeError("down")
    eps, picks, _ = run(SCR, llm=boom, log=lambda s: None); assert picks      # falls back to heuristic

def test_json_fence_tolerance():
    assert parse_json_list('```json\n[{"id":"1-0"}]\n```')[0]["id"] == "1-0"
