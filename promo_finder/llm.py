"""Swappable LLM backends (stdlib only). Pick with PROMO_LLM=anthropic|openai|gemini|mock.
Model is set with PROMO_MODEL (anthropic defaults to claude-sonnet-5-5; set it explicitly for others).
Keys: ANTHROPIC_API_KEY / OPENAI_API_KEY (+ optional OPENAI_BASE_URL) / GEMINI_API_KEY.
"""
import json, os, re, urllib.request, urllib.error


def _post(url, headers, body, timeout=120, retries=5):
    """POST with retry/backoff on 429/5xx (free tiers rate-limit hard). PROMO_DELAY = seconds to sleep
    before every call (e.g. PROMO_DELAY=7 keeps you under ~10 requests/min)."""
    import time
    time.sleep(float(os.environ.get("PROMO_DELAY", "0")))
    req_data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = min(60, 5 * 2 ** attempt)        # 5, 10, 20, 40, 60 s
                print(f"  ...model busy (HTTP {e.code}), retry {attempt + 1}/{retries} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                print(f"  ...network problem ({e}), retry {attempt + 1}/{retries}", flush=True)
                time.sleep(5 * (attempt + 1)); continue
            raise RuntimeError(f"LLM network error: {e}")


def anthropic(system, user, model=None):
    model = model or os.environ.get("PROMO_MODEL", "claude-sonnet-5-5")
    r = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
               "content-type": "application/json"},
              {"model": model, "max_tokens": 2000, "system": system,
               "messages": [{"role": "user", "content": user}]})
    return "".join(b.get("text", "") for b in r["content"])


def openai(system, user, model=None):
    model = model or os.environ["PROMO_MODEL"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = _post(f"{base}/chat/completions",
              {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "content-type": "application/json"},
              {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    return r["choices"][0]["message"]["content"]


def gemini(system, user, model=None):
    model = model or os.environ["PROMO_MODEL"]
    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=" + os.environ["GEMINI_API_KEY"],
              {"content-type": "application/json"},
              {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}]})
    return r["candidates"][0]["content"]["parts"][0]["text"]


def mock(system, user, model=None):
    """Offline stand-in used ONLY to test the plumbing. Its scores are meaningless."""
    ids = re.findall(r"^### WINDOW (\S+)", user, re.M)
    return json.dumps([{"id": i, "hook_strength": 7, "cliffhanger_strength": 7, "standalone": 7,
                        "hook": "(mock)", "cliffhanger": "(mock)", "why": "(mock)"} for i in ids])


BACKENDS = dict(anthropic=anthropic, openai=openai, gemini=gemini, mock=mock)


def get_backend(name):
    if name not in BACKENDS:
        raise SystemExit(f"unknown --llm {name!r}; choose from {list(BACKENDS)} or 'none'")
    return BACKENDS[name]


def parse_json_list(text):
    """Tolerate ```json fences and chatter around the array."""
    t = re.sub(r"```(?:json)?", "", text)
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < 0: raise ValueError("no JSON array in model output")
    return json.loads(t[i:j + 1])
