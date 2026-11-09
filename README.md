# Promo Finder — Kuku AI Generalist Assignment, Problem 1

Reads microdrama screenplays (timestamped scene/dialogue/action format) and proposes 20–60s promo clips:
a hook in the first 3s, an unresolved cliffhanger at the end, and (as far as text can tell) readable cold.

## Run it (no API key, ~1 second, Python 3.9+, standard library only)
    python run.py --scripts "<folder with episode-*.txt>" --out outputs   # quote the path if it has spaces
    python -m pytest -q tests          # needs: pip install pytest ; set SCRIPTS=<folder> first

Outputs: `promos.json`, `promos.csv`, `report.md` (ranked, with hook / cliffhanger / why / timestamps),
`run_summary.json` (counts + validation), `run_log.txt`.

## Turn on the LLM stage (swappable model)
    export PROMO_LLM=anthropic ANTHROPIC_API_KEY=...        # default model claude-sonnet-5-5; override with PROMO_MODEL
    python run.py --scripts <folder> --out outputs_llm --llm anthropic
    # other backends: --llm openai (OPENAI_API_KEY, PROMO_MODEL, optional OPENAI_BASE_URL)
    #                 --llm gemini (GEMINI_API_KEY, PROMO_MODEL)   |  --llm mock (offline plumbing test only)
The LLM only scores and explains windows that the code already cut on valid shot boundaries. It never
outputs timestamps, so it cannot invent a cut point. If a call or its JSON fails, that episode keeps the
heuristic result. **Status: this stage was tested only with the mock backend and a forced-failure backend;
no real LLM was available when I built it, so its prompt is unvalidated.**

## How it works
1. `parse.py` — screenplay -> events (camera / action / dialogue / sound) with exact times.
2. `score.py` — every (shot-start, shot-end) window of 20–60s that slices no spoken line (≈12k windows for
   50 episodes). Scores hook (first 3s), cliffhanger (last ~6s), standalone, pace with transparent rules.
3. `pipeline.py` — best non-overlapping windows per episode (max 2), gate: hook >= 0.40 AND cliff >= 0.45,
   rank with an early-episode weight (new viewers: less spoiler/context) and a cast-diversity discount.
4. `rerank.py` / `llm.py` — optional LLM judgement (see above). 5. `validate.py` — checks that need no labels.

## Assumptions
- A promo may start/end only on camera-shot boundaries (clean cuts for an editor), never mid-line.
- Early episodes make better promos for brand-new viewers (episode weight 1.0 for Ep 1–10, down to 0.85).
- The lexicons (insults, reveal words, tones) are hand-tuned to this show and genre; other genres need retuning
  or the LLM stage.

## Known limits (details in the solution doc)
Misses quiet-menace hooks (Ep 1 cathedral), underrates crowded rescue scenes, cannot tell a line needs
earlier context, multi-segment and cross-episode promos are barely supported, no ground-truth labels.

The screenplay pack is Kuku's confidential content and is NOT in this repo.

## Free-tier tips (Gemini)
    export GEMINI_API_KEY=AIza...  PROMO_MODEL=gemini-2.5-flash  PROMO_DELAY=7
    python run.py --scripts "<folder>" --out outputs_llm --llm gemini
`PROMO_DELAY` (seconds before each call) keeps you under free-tier rate limits; 429/5xx errors are retried with backoff.
The run prints how many episodes the model actually scored; any failure falls back to the heuristic and is logged in `run_log.txt`.
