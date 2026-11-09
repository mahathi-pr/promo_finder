#!/usr/bin/env python3
"""Find promos in microdrama screenplays.
  python run.py --scripts <folder of episode-*.txt> --out outputs
  PROMO_LLM=anthropic ANTHROPIC_API_KEY=... python run.py --scripts <folder> --out outputs_llm --llm anthropic
"""
import argparse, json, sys, time
from promo_finder.pipeline import run
from promo_finder.report import write_all
from promo_finder.validate import v1_cliffhanger_loo, v2_constraints
from promo_finder.llm import get_backend

ap = argparse.ArgumentParser()
ap.add_argument("--scripts", required=True)
ap.add_argument("--out", default="outputs")
ap.add_argument("--llm", default="none", help="none | anthropic | openai | gemini | mock")
ap.add_argument("--shortlist", type=int, default=6, help="windows per episode sent to the LLM")
a = ap.parse_args()

t0 = time.time()
log_lines = []
def log(s): print(s); log_lines.append(s)

backend = None if a.llm == "none" else get_backend(a.llm)
eps, picks, ncand = run(a.scripts, llm=backend, shortlist_k=a.shortlist, log=log)
recs = write_all(picks, a.out)
viol = v2_constraints(eps, picks)
v1 = v1_cliffhanger_loo(eps)
llm_failed = sum(1 for l in log_lines if "LLM failed" in l)
if a.llm != "none":
    print(f"\nLLM stage: {len(eps) - llm_failed}/{len(eps)} episodes scored by the model, {llm_failed} fell back to heuristic.")
    if llm_failed: print("  -> see run_log.txt for the reasons (rate limit? wrong PROMO_MODEL? bad key?)")
summary = dict(episodes=len(eps), llm_episodes_failed=llm_failed, windows_scored=ncand, promos=len(recs),
               episodes_with_a_promo=len({r["episode"] for r in recs}), llm=a.llm,
               hard_constraint_violations=viol, cliffhanger_leave_one_out=v1, seconds=round(time.time() - t0, 1))
json.dump(summary, open(f"{a.out}/run_summary.json", "w"), indent=2)
open(f"{a.out}/run_log.txt", "w").write("\n".join(log_lines))
print(json.dumps({k: v for k, v in summary.items() if k != "cliffhanger_leave_one_out"}, indent=2))
sys.exit(1 if viol else 0)
