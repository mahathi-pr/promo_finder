"""Parse Kuku screenplay .txt files into structured events.

Format (per episode):
  header: EPISODE NUMBER / LOGLINE / CONTINUITY NOTES
  SCENE NN, 'Scene heading', 'Scene start/end'
  blocks:  **LABEL (emotion):**  then  Start: (mm:ss.xx) | End: (mm:ss.xx)  then text
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

TS = r"(\d+):(\d+(?:\.\d+)?)"
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(rf"Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")


def to_sec(m, s):
    return int(m) * 60 + float(s)


def fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


@dataclass
class Event:
    kind: str          # camera | action | dialogue | sfx
    label: str         # CAMERA / ACTION / SOUND FX / speaker name
    tone: str          # emotion tag in parentheses for dialogue ('' otherwise)
    start: float
    end: float
    text: str
    scene: int = 0


@dataclass
class Episode:
    num: int
    logline: str
    continuity: str
    events: list = field(default_factory=list)
    scenes: dict = field(default_factory=dict)   # scene no -> (heading, start, end)
    path: str = ""

    @property
    def duration(self):
        return max((e.end for e in self.events), default=0.0)


def parse_episode(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    num = int(re.search(r"episode-(\d+)", path.name).group(1))
    logline = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("EPISODE LOGLINE:")), "")
    cont = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("CONTINUITY NOTES:")), "")
    ep = Episode(num, logline, cont, path=str(path))
    scene = 0
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        m = re.match(r"SCENE (\d+)", l)
        if m:
            scene = int(m.group(1))
            heading, s0, s1 = "", 0.0, 0.0
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].startswith("Scene heading:"):
                    heading = lines[j].split(":", 1)[1].strip()
                mm = re.match(rf"Scene (start|end):\s*{TS}", lines[j])
                if mm:
                    t = to_sec(mm.group(2), mm.group(3))
                    if mm.group(1) == "start":
                        s0 = t
                    else:
                        s1 = t
            ep.scenes[scene] = (heading, s0, s1)
        b = BLOCK_RE.match(l)
        if b and i + 1 < len(lines):
            t = TIME_RE.search(lines[i + 1])
            if t:
                label_full = b.group(1)
                tone_m = re.match(r"(.+?)\s*\((.*)\)\s*$", label_full)
                label, tone = (tone_m.group(1), tone_m.group(2)) if tone_m else (label_full, "")
                label = label.strip()
                # text = following non-empty lines until next block / scene
                j = i + 2
                buf = []
                while j < len(lines) and not BLOCK_RE.match(lines[j].strip()) \
                        and not lines[j].startswith("SCENE ") and not lines[j].startswith("--- SCRIPT END"):
                    if lines[j].strip():
                        buf.append(lines[j].strip())
                    j += 1
                up = label.upper()
                kind = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx"}.get(up, "dialogue")
                ep.events.append(Event(kind, label, tone,
                                       to_sec(t.group(1), t.group(2)),
                                       to_sec(t.group(3), t.group(4)),
                                       " ".join(buf).strip().strip('"'), scene))
                i = j
                continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.end))
    return ep


def load_all(folder):
    files = sorted(Path(folder).glob("episode-*.txt"))
    return [parse_episode(f) for f in files]
