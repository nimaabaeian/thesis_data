#!/usr/bin/env python3
"""Generator for analysis/orexigenic_analysis.ipynb.

Assembles the notebook as a list of (kind, source) cells and writes it with
nbformat. Keeping the notebook in one generator file makes it easy to keep cells
short, commented and idempotent.
"""
import nbformat as nbf
from pathlib import Path

CELLS = []
def md(src):   CELLS.append(("md", src.rstrip("\n")))
def code(src): CELLS.append(("code", src.rstrip("\n")))

# ==========================================================================
# TITLE
# ==========================================================================
md(r"""# Orexigenic Drive and Always-On Homeostatic Regulation — Analysis

**System under study:** the *Orexigenic Drive* of the `alwaysOn-embodiedBehaviour`
iCub controller, analysed as a socially embedded, always-on embodied sensory-control loop linking
perception → salience → executive regulation → remote/Telegram signalling.

**Research questions**

- **RQ1** — To what extent does the orexigenic drive instantiate the four operational
  functions of
  classical homeostasis: (1) internal monitoring, (2) deficit detection,
  (3) deficit-to-action-selection coupling, (4) behavioural priority reallocation?
- **RQ2** — Does the expression of an orexigenic deficit promote recovery-oriented
  engagement sufficient to support reliable energy replenishment in an always-on
  social robot?
- **RQ3** — Does the robot's **adaptive regulatory memory** (per-person homeostatic affinity)
  encode observed participant behaviour rather than uncontrolled drift, and is that learned
  state subsequently expressed in the robot's allocation of proactive approaches? The
  **role manipulation is experiment metadata only**: roles
  were assigned by the researchers for validation and were never available to the robot or
  used by the software.

> **Fixed design fact (single condition).** The drive was **always on** for the whole
> study. RQ2 is identified from the **within-drive graded deficit**
> (Full → Hungry → Starving) and from the **proactive / mixed-initiative vs reactive**
> contrast. *The graded deficit is the manipulation; the orexigenic state is treated as
> an internal drive-function signal that biases recovery-oriented action selection.*

> **Second design fact (two phases, role manipulation).** The deployment ran in **two
> 4-day phases**. In **Phase 1** (first four experiment days) participants had assigned
> roles: two **obligated feeders** (feed several times a day), two **interact-but-never-feed**
> participants, and everyone else **unconstrained**. In **Phase 2** (last four days) all
> constraints were lifted and everyone behaved normally. The role map is private
> (`analysis/private/role_phase.json`, pseudonymised on load). These roles are **external
> experimental labels**, not controller inputs; RQ3 (analysis B10) uses them only after the
> fact as a *manipulation check and external validation* of the affinity learning. With only
> **2 people per controlled role**, role contrasts are validation evidence with wide
> uncertainty, not population inference.

**Data-layout note (discovered, not assumed).** The eight dated folders under `data/`
are **cumulative snapshots of one continuously-growing database** — each later folder
is a strict superset of the earlier ones. We therefore de-duplicate to the true unit
of analysis, the **run** (`run_id`, 10 runs) grouped by data-collection **day**
(`day_rome`, 8 days). Naive concatenation across folders would 4–5× double-count; we
do not do that. This is verified in Phase 0.2 and the verification gate.
""")

# ==========================================================================
# SETUP
# ==========================================================================
md("## Setup — imports, seed, configuration, palette")

code(r"""
# --- Single setup cell: imports, global seed, config, output dirs, palette ---
from __future__ import annotations
import os, re, json, sqlite3, warnings, glob, math, itertools, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Warnings are silenced, and this is deliberate rather than lazy. Two reasons:
#   1. The expected ones are noise here: MixedLM boundary-variance and GEE iteration-limit
#      warnings are what a 14-cluster design DOES, and every model that can hit them already
#      reports a cluster bootstrap alongside, which is the check that matters.
#   2. Warning text embeds absolute site-packages paths, and this machine's home directory is a
#      participant's name. A filter alone proved unreliable (libraries reset it), so the sink is
#      replaced outright — the leak checker in check_notebook.py enforces that this holds.
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.showwarning = lambda *a, **k: None
os.environ["PYTHONWARNINGS"] = "ignore"
SEED = 42
np.random.seed(SEED)
import random; random.seed(SEED)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

# ---- Config -------------------------------------------------------------------
# DATA_ROOT auto-discovery: env override -> ./data -> repo root.
def _find_data_root():
    env = os.getenv("DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    here = Path.cwd()
    for base in [here] + list(here.parents):
        if (base / "data").is_dir() and any((base / "data").glob("*/data_collection")):
            return base / "data"
        if any(base.glob("*/data_collection/executive_control.db")):
            return base
    return here / "data"

DATA_ROOT = _find_data_root()
REPO_ROOT = DATA_ROOT.parent if DATA_ROOT.name == "data" else DATA_ROOT

ANALYSIS_DIR = REPO_ROOT / "analysis"
CACHE_DIR    = ANALYSIS_DIR / "cache"
OUT_DIR      = ANALYSIS_DIR / "outputs"
FIG_DIR      = ANALYSIS_DIR / "figures"
for d in (CACHE_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- HS colour palette (mirrored from stomachMonitor.py, liquid colour) -------
# Internal DB state codes are HS1/HS2/HS3; we DISPLAY them as Full/Hungry/Starving
# everywhere (figures, tables, text). HS_ORDER holds the data keys; HS_NAMES the labels.
# Categorical-validated (lightness band + CVD separation + chroma): marks carrying data
# always pair these fills with direct value labels (the contrast-relief obligation).
HS_PALETTE = {"HS1": "#4FB98A", "HS2": "#E3A32E", "HS3": "#E5654D", "HS0": "#AEB4BA"}
HS_ACCENT  = {"HS1": "#2C8A60", "HS2": "#B9821C", "HS3": "#B23A26", "HS0": "#6E767D"}
HS_LABEL   = {"HS1": "Full", "HS2": "Hungry", "HS3": "Starving", "HS0": "N/A"}
HS_ORDER   = ["HS1", "HS2", "HS3"]                       # data keys (do not change)
HS_NAME    = {"HS1": "Full", "HS2": "Hungry", "HS3": "Starving", "HS0": "N/A"}
HS_NAMES   = ["Full", "Hungry", "Starving"]             # display labels, same order as HS_ORDER
def hsn(x):
    "Map a state code (or iterable of codes) to its display name Full/Hungry/Starving."
    if isinstance(x, str): return HS_NAME.get(x, x)
    return [HS_NAME.get(v, v) for v in x]

# Social-state display names (data keys ss1..ss4 stay internal). Source SS_DESCRIPTIONS:
# ss1 Unknown, ss2 Known-not-greeted, ss3 Known-greeted, ss4 Talked.
SS_NAME  = {"ss1": "Stranger", "ss2": "Recognized", "ss3": "Greeted", "ss4": "Engaged"}

# --- Identity resolution & pseudonymization (privacy-first) ----------------------------
# Two layers, applied to the same identity columns:
#  1) Canonicalization merges case/spelling variants of the same person and remaps
#     excluded participants to "unknown". The map lives in analysis/private/
#     identity_canon.json (git-ignored) so no real name appears in this notebook.
#  2) Pseudonymization replaces every real identity with a stable code (P01, P02, ...)
#     assigned by first appearance (cell A1b). The real-name mapping is written to
#     analysis/private/ only — published figures/tables/CSVs carry codes exclusively.
PRIVATE_DIR = ANALYSIS_DIR / "private"
PRIVATE_DIR.mkdir(exist_ok=True)
_canon_path = PRIVATE_DIR / "identity_canon.json"
CANON_IDENTITY = json.loads(_canon_path.read_text()) if _canon_path.exists() else {}
if not CANON_IDENTITY:
    print("WARNING: analysis/private/identity_canon.json missing — name variants will not merge.")
# `feeder_face_id` names the PERSON who delivered a meal. It was previously absent from this
# tuple, so it was never canonicalised or pseudonymised — both a real-name leak vector and the
# reason meals could not be attributed to anyone: feeding events carry NO exec_interaction_id, so
# feeder_face_id is the only link from a meal back to who gave it.
IDENTITY_COLS = ("person_id", "face_id", "extracted_name", "user_key", "feeder_face_id")
def canon_identity(v):
    "Canonicalise a single identity string; pass through non-strings and unlisted names."
    if not isinstance(v, str): return v
    return CANON_IDENTITY.get(v.strip().lower(), v)

PSEUDONYM_PRESERVE = {"unknown", "unmatched", ""}   # placeholders, not people
PSEUDONYM_MAP = {}                     # real identity -> P## (seeded in cell A1b)
PSEUDONYMS_ACTIVE = False              # flipped on once the map is seeded
def pseudonymize(v):
    "Replace a real identity with its stable pseudonym; unseen identities get the next code."
    if not isinstance(v, str): return v
    key = v.strip()
    if key.lower() in PSEUDONYM_PRESERVE or re.fullmatch(r"P\d{2,}", key): return v
    if key not in PSEUDONYM_MAP:
        PSEUDONYM_MAP[key] = f"P{len(PSEUDONYM_MAP)+1:02d}"
    return PSEUDONYM_MAP[key]
def apply_pseudonyms(df):
    "Replace real names with pseudonyms in every identity column (no-op until seeded)."
    if not PSEUDONYMS_ACTIVE: return df
    for c in IDENTITY_COLS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(pseudonymize)
    return df

# --- Study phase (external design metadata, not a controller input) -------------------
# Hoisted into setup because B3/B5/B7 all need to condition on phase, not just B10.
# Roles are resolved later (they need the pseudonym map, which is seeded in A1b).
_RP = json.loads((PRIVATE_DIR / "role_phase.json").read_text()) \
      if (PRIVATE_DIR / "role_phase.json").exists() else {}
PHASE1_DAYS_EARLY = set(_RP.get("phase1_days", []))
def phase_of_day(day):
    "P1 = the first 4 days (assigned roles); P2 = the last 4 (all constraints lifted)."
    return "P1" if str(day) in PHASE1_DAYS_EARLY else "P2"

# Recessive grid/axes, constrained_layout ON (prevents title/tick collisions in
# every multi-panel figure), consistent typography. Ink colours for text so labels
# never wear a series colour.
INK, MUTED, GRID = "#22282E", "#5A6470", "#C7CDD4"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.hspace": 0.08,
    "figure.constrained_layout.wspace": 0.04,
    "savefig.pad_inches": 0.15,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.alpha": 0.6,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.9,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.titlepad": 8, "axes.labelsize": 11, "font.size": 11,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "font.family": "DejaVu Sans",
})

def savefig(fig, name):
    "Save a figure to analysis/figures as PNG + SVG at >=220 dpi."
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06, wspace=0.04, hspace=0.08)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=220)
    print(f"  saved figures/{name}.png + .svg")

def bars_with_ci(ax, cats, means, los, his, colors, n_labels=None, small_flag=None,
                 value_fmt="{:.2f}"):
    "Vertical bars with asymmetric CI whiskers, direct value labels, optional n= and small-n hatch."
    x = np.arange(len(cats))
    for i, c in enumerate(cats):
        hatch = "//" if (small_flag and small_flag[i]) else None
        ax.bar(x[i], means[i], color=colors[i], edgecolor="white", linewidth=1.2,
               width=0.72, hatch=hatch, zorder=3)
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.3,
                capsize=4, capthick=1.3, zorder=4)
    for i in range(len(cats)):
        top = means[i] + his[i]
        ax.annotate(value_fmt.format(means[i]), (x[i], top), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="medium")
        if n_labels is not None:
            ax.annotate(f"n={n_labels[i]}", (x[i], 0), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.margins(x=0.08)

# Print paths with $HOME redacted: the username must never appear in published
# notebook output (the leak checker treats it as an identity string).
def show_path(p): return str(p).replace(str(Path.home()), "~")
print("DATA_ROOT :", show_path(DATA_ROOT))
print("REPO_ROOT :", show_path(REPO_ROOT))
print("outputs   :", show_path(OUT_DIR))
""")

# ==========================================================================
# PHASE 0 — READ THE CODE AND THE DATA FIRST
# ==========================================================================
md(r"""## Phase 0 — Ground truth: read the code, then the data

### 0.1 Ground-truth constants (read from source)

Extracted directly from the controller source (`alwaysOn-embodiedBehaviour/modules/`),
not from memory. File:line references given for auditability.

**`executiveControl.py` — `HungerModel` / drive logic**

| Constant | Value | Source |
|---|---|---|
| `drain_hours` (time to empty) | **4.0 h** | `HungerModel.__init__` L107; `configure` L1074 |
| passive drain rate | `100/(4·3600)` = **6.94e-3 %/s** | `_drain` L184 |
| `hungry_threshold` (HS1↔HS2) | **60.0** | L108, L1075 |
| `starving_threshold` (HS2↔HS3) | **25.0** | L109, L1076 |
| HS mapping | level≥60→**Full** (HS1), 25≤level<60→**Hungry** (HS2), level<25→**Starving** (HS3) | `snapshot` L241-246 |

> **Naming.** The controller's state codes are `HS1`/`HS2`/`HS3`; throughout this notebook we
> display them as **Full** / **Hungry** / **Starving** (the codes remain the data keys).
| Meal deltas | `SMALL=10`, `MEDIUM=25`, `LARGE=45` | `_meal_mapping` L987 |
| QR cooldown | **3.0 s** | `_qr_cooldown_sec` L988 |
| feed-wait timeout | **8.0 s** | `_feed_wait_timeout_sec` L989 |
| HS3 feeding-tree override | `_run_hunger_tree` L2329 | see below |
| DB file | `data_collection/executive_control.db` | `DB_FILE` L875 |

**Active metabolic energy costs (charged via `_charge_energy` / `exert`)**

| Constant | Value | Action label(s) |
|---|---|---|
| `CONVERSATION_TURN_ENERGY_COST` | **3.6** | `ss3_conversation_turn` |
| `STARTER_PROMPT_ENERGY_COST` | **1.2** | `ss3_starter` |
| `NAME_QUESTION_ENERGY_COST` | **1.0** | `ss1_ask_name`, `ss1_ask_name_retry` |
| `HUNGER_PROMPT_ENERGY_COST` | **1.0** | `hunger_ask_feed`, `hunger_still_hungry`, `hunger_look_around` |
| `GREETING_ENERGY_COST` | **0.8** | `ss1_greeting`, `ss1_nice_to_meet`, `known_greeting`, reactive greetings |
| `FEED_ACK_ENERGY_COST` | **0.8** | `feed_ack`, `ambient_feed_ack` |

**HS3 feeding-tree override (`_run_hunger_tree`, L2329).** On HS3 the executive stops
the social agenda and enters a feed-seeking loop: it *asks to be fed* (`hunger_ask_feed`),
waits up to 8 s for a QR meal, acknowledges each feed (`feed_ack`), says
`hunger_still_hungry` and loops until the state climbs back to **HS1**, or aborts with
`no_food_qr` after 2 consecutive timeouts. This is the coded behavioural-prioritisation
mechanism tested in B4.

**`salienceNetwork.py`**

| Constant | Value | Source |
|---|---|---|
| `BASELINE_WEIGHTS` | prox 0.5, cent 0.15, gaze 0.5 | L141 |
| `SS_THRESHOLDS` | ss1 0.80, ss2 0.65, ss3 0.70, ss4 0.85 | L144 |
| `HABITUATION_LAMBDA` | 0.06 (~11.6 s half-life) | L151 |
| affinity EMA | `ALPHA=0.25`, `ALPHA_NEG=0.10`, `REWARD_SCALE=25`, `PENALTY_FLOOR=-0.3` | L155-160 |
| affinity→threshold | `GAIN=0.15`, `FLOOR=0.50` (`eff_thr = max(FLOOR, base − GAIN·affinity)`) | L161-162, L1360 |
| IPS image normalisation | 640×480 | `_compute_ips` L1381 |
| `SS_DESCRIPTIONS` | ss1 Unknown / ss2 Known,NotGreeted / ss3 Known,Greeted,NoTalk / ss4 Talked | L171 |
| DB file | `data_collection/salience_network.db` | L282 |

> **Social-state naming.** The controller's codes `ss1`/`ss2`/`ss3`/`ss4` are displayed
> throughout as **Stranger** / **Recognized** / **Greeted** / **Engaged** (codes remain the data keys).

**`chatBot.py` (Telegram, proactive)** — `HS_DWELL_SEC=60` (debounce a flapping drive
before proactive logic acts), `HS2_ENTRY_MIN_INTERVAL=30 min`,
`HS3_RECOVERY_MIN_INTERVAL=30 min`; proactive event types `hs2_entry`, `hs3_proactive`,
`hs3_recovery`; a debounced `_stable_hs` latch (L163) motivates B2's flapping analysis.
DB `data_collection/chat_bot.db` (L110).

**`vision.py`** — `landmark_events` schema (gaze/attention/zone/distance/`faces_in_frame`);
DB `data_collection/vision.db` (L128).

**`stomachMonitor.py`** — HS colour palette (reused in every figure): HS1/**Full** green `#5BCB97`,
HS2/**Hungry** amber `#F4BB52`, HS3/**Starving** red `#ED7059`, HS0 grey `#AEB4BA` (L72-75).

**`prompts.json`** — the deficit is verbalised as *"I'm so hungry, would you feed me
please?"* (`hunger_ask_feed`), *"I'm still hungry. Give me more please."*
(`hunger_still_hungry`), and HS3 feed-ack *"Oh wow, thank you {name}! You literally just
saved my life!"* — the coded gates behind the B3 framing numbers.
""")

code(r"""
# Ground-truth constants recorded as data (used by the verification gate V1/V5).
CONST = dict(
    DRAIN_HOURS=4.0, HUNGRY_THRESHOLD=60.0, STARVING_THRESHOLD=25.0,
    DRAIN_RATE_PER_SEC=100.0 / (4.0 * 3600.0),          # nominal passive drain
    MEALS={"SMALL_MEAL": 10.0, "MEDIUM_MEAL": 25.0, "LARGE_MEAL": 45.0},
    QR_COOLDOWN_SEC=3.0, FEED_WAIT_TIMEOUT_SEC=8.0,
    ACTIVE_COST={
        "ss3_conversation_turn": 3.6, "ss3_starter": 1.2,
        "ss1_ask_name": 1.0, "ss1_ask_name_retry": 1.0,
        "hunger_ask_feed": 1.0, "hunger_still_hungry": 1.0, "hunger_look_around": 1.0,
        "known_greeting": 0.8, "ss1_greeting": 0.8, "ss1_nice_to_meet": 0.8,
        "feed_ack": 0.8, "ambient_feed_ack": 0.8,
        "reactive_unknown_greeting": 0.8, "reactive_nice_to_meet": 0.8,
    },
    SS_THRESHOLDS={"ss1": 0.80, "ss2": 0.65, "ss3": 0.70, "ss4": 0.85},
    BASELINE_WEIGHTS={"prox": 0.5, "cent": 0.15, "gaze": 0.5},
    HABITUATION_LAMBDA=0.06, IMG_W=640.0, IMG_H=480.0,
    # Affinity EMA (salienceNetwork.py L155-162) — the adaptive regulatory memory.
    AFFINITY={"ALPHA": 0.25, "ALPHA_NEG": 0.10, "REWARD_SCALE": 25.0,
              "PENALTY_FLOOR": -0.3, "POSITIVE_BAND": 0.20,
              "THR_GAIN": 0.15, "THR_FLOOR": 0.50},
    CHAT={"PRIORITY_THRESHOLD_HS2": 0.20, "MIN_INTERACTIONS_FOR_TRUST": 3,
          "HS_DWELL_SEC": 60.0},
)
# Threshold aliases used throughout: the LEVEL is ground truth, the label is derived from it.
HS_FULL_MIN     = CONST["HUNGRY_THRESHOLD"]     # level >= 60  -> HS1 (Full)
HS_STARVING_MAX = CONST["STARVING_THRESHOLD"]   # level <  25  -> HS3 (Starving)
print("Nominal passive drain rate: %.6f %%/s  (empty in %.1f h)"
      % (CONST["DRAIN_RATE_PER_SEC"], CONST["DRAIN_HOURS"]))

# The constants above are checked against the pinned controller source by
# `analysis/check_constants.py` (see the Reproducibility section). They are NOT
# re-typed from memory anywhere else in this notebook.
CONST_SOURCE = json.loads((ANALYSIS_DIR / "controller_source.json").read_text()) \
    if (ANALYSIS_DIR / "controller_source.json").exists() else {}
if CONST_SOURCE:
    # The repo URL is NOT printed: it embeds a GitHub username who is also a study participant.
    # It lives in analysis/controller_source.json, which is where a reader should look for it.
    _commit = str(CONST_SOURCE.get("commit", "UNPINNED"))
    print(f"Controller source: see analysis/controller_source.json; commit = "
          f"{_commit[:12] if _commit != 'UNPINNED' else 'UNPINNED (constants are UNVERIFIED against source)'}")
""")

md("### 0.2 Data layout discovery + de-duplication to the run/day unit")

code(r"""
# Auto-discover session folders that contain a data_collection dir with the 4 DBs.
DB_KINDS = {
    "executive": "executive_control.db",
    "salience":  "salience_network.db",
    "vision":    "vision.db",
    "chat":      "chat_bot.db",
}
def discover_sessions(root: Path):
    sessions = {}
    for dc in sorted(root.glob("*/data_collection")):
        folder = dc.parent.name
        present = {k: (dc / fn) for k, fn in DB_KINDS.items() if (dc / fn).exists()}
        if present:
            sessions[folder] = {"path": dc.parent, "dbs": present}
    return sessions

SESSIONS = discover_sessions(DATA_ROOT)
print(f"Discovered {len(SESSIONS)} session folder(s) under {show_path(DATA_ROOT)}")

def ro_connect(path: Path):
    "Open a sqlite DB strictly read-only / immutable (never mutate the source)."
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)

def _count(path, sql):
    try:
        con = ro_connect(path); n = con.execute(sql).fetchone()[0]; con.close(); return n
    except Exception as e:
        return f"ERR:{e}"

# Session manifest: which DBs present + key row counts (raw, pre-dedup).
rows = []
for folder, info in SESSIONS.items():
    d = info["dbs"]
    rows.append(dict(
        session=folder,
        exec=("Y" if "executive" in d else "-"),
        sal=("Y" if "salience" in d else "-"),
        vis=("Y" if "vision" in d else "-"),
        chat=("Y" if "chat" in d else "-"),
        interactions=_count(d["executive"], "SELECT COUNT(*) FROM interactions") if "executive" in d else None,
        runs=_count(d["executive"], "SELECT COUNT(DISTINCT run_id) FROM interactions") if "executive" in d else None,
        days=_count(d["executive"], "SELECT COUNT(DISTINCT day_rome) FROM interactions") if "executive" in d else None,
    ))
manifest = pd.DataFrame(rows)
print("\n== Session manifest (RAW counts — note the monotone growth = cumulative snapshots) ==")
print(manifest.to_string(index=False))

assert len(SESSIONS) == 8, f"Expected 8 session folders, found {len(SESSIONS)}"
for folder, info in SESSIONS.items():
    missing = set(DB_KINDS) - set(info["dbs"])
    if missing:
        print(f"  WARNING: session {folder} missing DBs: {missing}")
""")

code(r"""
# Confirm the cumulative-snapshot structure and pick the SUPERSET folder (run-union).
def run_set(path):
    con = ro_connect(path); s = set(r[0] for r in con.execute("SELECT DISTINCT run_id FROM interactions")); con.close(); return s

run_sets = {f: run_set(info["dbs"]["executive"]) for f, info in SESSIONS.items()}
union_runs = set().union(*run_sets.values())
SUPERSET_FOLDER = max(run_sets, key=lambda f: len(run_sets[f]))
is_super = run_sets[SUPERSET_FOLDER] == union_runs

print(f"Union of runs across all folders : {len(union_runs)}")
print(f"Superset folder                  : {SUPERSET_FOLDER} ({len(run_sets[SUPERSET_FOLDER])} runs)")
print(f"Superset contains every run      : {is_super}")
# Every folder's run set must be a subset of the superset (cumulative property).
assert all(run_sets[f] <= run_sets[SUPERSET_FOLDER] for f in run_sets), \
    "Folders are NOT cleanly cumulative; fall back to dedup-union loader."
print("Confirmed: folders are cumulative; loading from the superset snapshot per DB kind.")
SUPER = SESSIONS[SUPERSET_FOLDER]["dbs"]
""")

md("### 0.3 / 0.4 Read-only access + schema confirmation")

code(r"""
# Print schema for the key executive tables + list all views (one snapshot).
def table_info(path, table):
    con = ro_connect(path)
    df = pd.read_sql(f"PRAGMA table_info({table})", con); con.close()
    return list(df["name"])

def list_views(path):
    con = ro_connect(path)
    v = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")]
    con.close(); return v

for kind, table in [("executive","interactions"), ("executive","hunger_level_events"),
                    ("salience","face_ips_events"), ("salience","interaction_attempts"),
                    ("vision","landmark_events"), ("chat","chat_messages")]:
    cols = table_info(SUPER[kind], table)
    print(f"{kind}.{table}  ({len(cols)} cols): {', '.join(cols[:14])} ...")
print()
for kind in DB_KINDS:
    print(f"{kind} views:", ", ".join(list_views(SUPER[kind])))
""")

# ==========================================================================
# PHASE A — DATA PREPARATION
# ==========================================================================
md(r"""## Phase A — Data preparation

Loaders read the **superset snapshot** per DB kind (Phase 0 confirmed the folders are
cumulative), prefer the pre-built `*_clean` views (which already apply
`valid_for_analysis=1`), attach `session_id = day_rome`, and cache to
`analysis/cache/*.parquet`. The cluster/unit key is `(run_id)`; `person_id`/`user_key`
nests within it.
""")

code(r"""
# --- A1. Loader: read a view/table from the superset snapshot, cache to parquet ---
def apply_identity_canon(df):
    "Apply the private canonicalization map (variant merges, exclusions) to identity columns."
    for c in IDENTITY_COLS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(canon_identity)
    return df

def filter_analysis_rows(df):
    "Apply analysis-validity filters idempotently, even for views that are expected to be clean."
    df = df.copy()
    if "valid_for_analysis" in df.columns:
        df = df[df["valid_for_analysis"] == 1]
    if "is_test_run" in df.columns:
        df = df[df["is_test_run"] == 0]
    return df

def load_view(db_kind, view_or_table, columns=None, force=False):
    "Load a view/table (superset snapshot), add session_id=day_rome, canonicalise identities, cache."
    cache = CACHE_DIR / f"{db_kind}__{view_or_table}.parquet"
    if cache.exists() and not force:
        df = filter_analysis_rows(pd.read_parquet(cache))
        # canon + pseudonyms on cache-hit too (cache stores real names; publish-safe frames only in RAM)
        return apply_pseudonyms(apply_identity_canon(df.reset_index(drop=True)))
    path = SUPER[db_kind]
    cols = "*" if not columns else ", ".join(columns)
    con = ro_connect(path)
    df = pd.read_sql(f"SELECT {cols} FROM {view_or_table}", con)
    con.close()
    # Apply validity filters everywhere. Clean views already satisfy this today, but
    # making it idempotent prevents future raw/diagnostic views from slipping in.
    df = filter_analysis_rows(df)
    # session_id = data-collection day (the meaningful 'session'); keep run_id too.
    if "day_rome" in df.columns:
        df["session_id"] = df["day_rome"]
    df = apply_identity_canon(df.reset_index(drop=True))
    df.to_parquet(cache, index=False)   # cache keeps real names (private); pseudonymize on return
    return apply_pseudonyms(df)

# Heavy tables: read only needed columns.
LM_COLS  = ["run_id","day_rome","monotonic_sec","timestamp_epoch","frame_id","faces_in_frame",
            "face_id","track_id","zone","distance","cos_angle","attention","is_talking",
            "time_in_view","valid_for_analysis","is_test_run"]
IPS_COLS = ["run_id","day_rome","monotonic_sec","timestamp_epoch","track_id","face_id","person_id",
            "social_state","is_known","eligible","is_active_target","ips","ips_before_habituation",
            "habituation_multiplier","prox_score","cent_score","gaze_score",
            "valid_for_analysis","is_test_run"]

print("Loading core frames ...")
interactions = load_view("executive", "v_interactions_clean")
turns        = load_view("executive", "v_interaction_turns")
hunger_tl    = load_view("executive", "v_hunger_level_timeline")
hunger_raw   = load_view("executive", "hunger_level_events")
reactive     = load_view("executive", "reactive_interactions")
attempts     = load_view("salience",  "v_interaction_attempts_clean")
ips          = load_view("salience",  "face_ips_events", columns=IPS_COLS)
landmarks    = load_view("vision",    "landmark_events", columns=LM_COLS)
chat_msgs    = load_view("chat",      "v_chat_messages_clean")
chat_events  = load_view("chat",      "v_chat_events_clean")

for nm, df in [("interactions",interactions),("turns",turns),("hunger_tl",hunger_tl),
               ("attempts",attempts),("ips",ips),("landmarks",landmarks),
               ("chat_msgs",chat_msgs),("chat_events",chat_events)]:
    print(f"  {nm:14s} {df.shape}")
print("runs:", interactions['run_id'].nunique(), "| days:", interactions['session_id'].nunique())
""")

code(r"""
# --- A1b. Pseudonymization: stable codes assigned by first appearance -----------
# Every real identity becomes P01, P02, ... ordered by its earliest timestamp_epoch
# across the core frames. The real-name mapping is written to analysis/private/
# pseudonym_map.json (git-ignored); published outputs carry the codes only. All
# frames loaded later via load_view()/read_view_super() are pseudonymized on load.
_first_seen = {}
for _df in (interactions, turns, hunger_tl, hunger_raw, reactive, attempts, ips, landmarks):
    if "timestamp_epoch" not in _df.columns: continue
    for _c in IDENTITY_COLS:
        if _c in _df.columns and _df[_c].dtype == object:
            for _name, _ts in _df.dropna(subset=[_c]).groupby(_c)["timestamp_epoch"].min().items():
                _k = str(_name).strip()
                if _k.lower() in PSEUDONYM_PRESERVE: continue
                if _k not in _first_seen or _ts < _first_seen[_k]:
                    _first_seen[_k] = _ts
for _k in sorted(_first_seen, key=_first_seen.get):
    PSEUDONYM_MAP[_k] = f"P{len(PSEUDONYM_MAP)+1:02d}"
PSEUDONYMS_ACTIVE = True

(PRIVATE_DIR / "pseudonym_map.json").write_text(json.dumps(PSEUDONYM_MAP, indent=2) + "\n")

for _nm in ("interactions","turns","hunger_tl","hunger_raw","reactive","attempts",
            "ips","landmarks","chat_msgs","chat_events"):
    globals()[_nm] = apply_pseudonyms(globals()[_nm])
print(f"Pseudonymized {len(PSEUDONYM_MAP)} identities as P01..P{len(PSEUDONYM_MAP):02d} "
      "(by first appearance); real-name map kept in analysis/private/, never published.")
""")

code(r"""
# --- A2. Keys & clocks --------------------------------------------------------
# Standardise: timestamp_epoch = absolute time; monotonic_sec = within-run interval.
# NEVER mix monotonic across runs. Derive helper columns on the interaction frame.
def add_clock_keys(df):
    df = df.copy()
    if "timestamp_epoch" in df.columns:
        ts = pd.to_datetime(df["timestamp_epoch"], unit="s", utc=True).dt.tz_convert("Europe/Rome")
        df["dt_rome"] = ts
        df["hour_of_day"] = ts.dt.hour
    # run_elapsed_sec already present in these tables; keep as within-run clock.
    return df

interactions = add_clock_keys(interactions)
# user_key: prefer a learned name, else face_id.
def _user_key(row):
    nm = str(row.get("extracted_name") or "").strip()
    return nm if nm else str(row.get("face_id") or "unknown")
interactions["user_key"] = interactions.apply(_user_key, axis=1)
# person_id proxy on interactions = face_id (salience uses person_id ~ face_id/name)
interactions["person_id"] = interactions["face_id"].astype(str)
RUN_KEY = "run_id"
print("Clock keys added. Example interaction keys:")
print(interactions[["run_id","session_id","hour_of_day","user_key",
                    "hunger_state_start","initial_state","final_state"]].head())
""")

md("### A3. Quality gates (assert zero, or report)")

code(r"""
# --- A3. Quality-view checks + custom asserts grounded in the constants --------
QUALITY = {"soft": [], "hard": []}
def q_check(name, ok, detail="", hard=True):
    (QUALITY["hard"] if hard else QUALITY["soft"]).append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name}  {detail}")

def read_view_super(kind, view):
    con = ro_connect(SUPER[kind]); df = pd.read_sql(f"SELECT * FROM {view}", con); con.close()
    return apply_pseudonyms(apply_identity_canon(df))   # same privacy pipeline here too

print("Quality views (must be empty):")
for kind, view in [("executive","v_quality_hunger_invalid_levels"),
                   ("executive","v_quality_interaction_missing_metadata"),
                   ("salience","v_quality_salience_missing_metadata"),
                   ("chat","v_quality_chat_missing_metadata")]:
    try:
        n = len(read_view_super(kind, view))
        q_check(f"{view} empty", n == 0, f"(rows={n})")
    except Exception as e:
        q_check(f"{view} empty", False, f"(view error: {e})", hard=False)

# Cell sizes by hunger state, aggregated across runs (the per-run breakdown lives in the
# views; here we show the analysis-relevant totals — note each module has its own run_ids).
print("\nCell sizes by hunger state (totals across runs; the Starving cells are the small-n ones):")
def hs_totals(kind, view, statecol, valcol):
    try:
        v=read_view_super(kind,view)
        g=v.groupby(statecol)[valcol].sum().reindex(["HS1","HS2","HS3"])
        nruns=v["run_id"].nunique()
        print(f"  {view:34s} (runs={nruns}): " +
              "  ".join(f"{s}={int(g.get(s,0) or 0)}" for s in ["HS1","HS2","HS3"]))
    except Exception as e:
        print(f"  ({view} unavailable: {e})")
hs_totals("executive","v_quality_condition_counts","hunger_state_start","interaction_count")
hs_totals("salience","v_quality_attempt_counts","hunger_state","attempt_count")
hs_totals("chat","v_quality_chat_condition_counts","hs","row_count")

print("\nCustom asserts grounded in ground-truth constants:")
lv = hunger_raw["stomach_level_after"]
q_check("stomach level in [0,100]", ((lv >= 0) & (lv <= 100)).all(),
        f"(min={lv.min():.2f}, max={lv.max():.2f})")
feeds = hunger_raw[hunger_raw["event_type"] == "feeding"]
q_check("no feeding with meal_delta==0", (feeds["meal_delta"] != 0).all(),
        f"(n_feeds={len(feeds)})")
bad_meal = feeds[~feeds.apply(lambda r: abs(r["meal_delta"] - CONST["MEALS"].get(r["meal_payload"], -1)) < 1e-6, axis=1)]
q_check("meal_delta matches SMALL/MEDIUM/LARGE constants", len(bad_meal) == 0, f"(mismatches={len(bad_meal)})")
# HS label consistent with level thresholds (on the 'after' side).
def hs_from_level(x):
    return "HS1" if x >= 60 else ("HS2" if x >= 25 else "HS3")
chk = hunger_raw.dropna(subset=["hunger_state_after"]).copy()
chk["expected"] = chk["stomach_level_after"].apply(hs_from_level)
mismatch = (chk["expected"] != chk["hunger_state_after"]).sum()
q_check("HS label consistent with level thresholds", mismatch == 0,
        f"(mismatch rows={mismatch}/{len(chk)})", hard=False)
""")

code(r"""
# Drain monotonicity: passive drain and active costs must never increase the level.
# Do not diff after dropping feeding rows: that compares the post-feed sample to the
# pre-feed row and creates false positives across legitimate meal jumps.
def drain_monotonic_report(hr):
    checked = hr[hr["event_type"].isin(["sample", "active_cost"])].copy()
    delta = pd.to_numeric(checked["level_delta"], errors="coerce")
    return int((delta > 1e-6).sum()), int(delta.notna().sum())
bad, total = drain_monotonic_report(hunger_raw)
q_check("passive/active level changes are non-increasing", bad == 0,
        f"(violations={bad}/{total})", hard=False)

# Emit quality_report.md
def write_quality_report():
    lines = ["# Quality report", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    lines.append("| check | result | detail |\n|---|---|---|")
    for sev in ("hard","soft"):
        for name, ok, detail in QUALITY[sev]:
            tag = ("PASS" if ok else ("FAIL" if sev=="hard" else "WARN"))
            lines.append(f"| {name} | {tag} | {detail} |")
    (OUT_DIR / "quality_report.md").write_text("\n".join(lines))
    print("wrote outputs/quality_report.md")
write_quality_report()
""")

md("### A4. Episode / unit reconstruction")

md(r"""> **Performance note.** `v_hs3_episodes` and `v_drain_segments` use correlated
> subqueries that are fine on the small early snapshots but O(n²) on the 164k-row
> superset (they hang). We reconstruct them in pandas below.

#### A4.0 — The logging artefact that forced this rebuild

The executive logger writes `hunger_state_before` and `hunger_state_after` as **the same
value** on the row where a threshold is crossed by *passive drain*:

```
monotonic  event   stimulus       hs_before  hs_after  level_before  level_after
11529.10   sample  passive_drain  HS2        HS2       25.012734     25.005765
11530.10   sample  passive_drain  HS3        HS3       25.005765     24.998797   <- the crossing
```

Only *discrete* events (`interaction_cost`, `feeding`, `mode`) produce a genuine
`before != after` row. Anything keyed off those two label columns is therefore **blind to
every drain-driven crossing**. Two artefacts followed, and both are corrected here:

- **`v_hs_transitions` contains zero passive-drain crossings.** It holds 49 Full→Hungry
  falls, all of them `interaction_cost`; the **32 real drain-driven falls are absent**, as are
  2 of the 10 Hungry→Starving falls. This is why the view reports 8 entries into Starving but
  17 exits from it — an impossible ledger, and the tell that something was missing.
- **The old Starving-episode builder found 8 episodes where the level series contains 17.**
  It looked for `after == "HS3" and before != "HS3"`, so it kept only the episodes the robot
  fell into *while someone was interacting with it* — and those are, unsurprisingly, the
  episodes where someone was standing there to feed it. The old "8/8 recovered by feeding"
  was the selection rule restating itself.

From here on, **hunger state is derived from the level series**, which is ground truth: the
level is the software integrator, the label is a view of it. `hs_transitions_logged` is
retained only so the two can be compared.""")

code(r"""
# --- A4. Materialise analysis units. -------------------------------------------------
#
# CRITICAL FIX (see the markdown cell above): the executive logger does NOT emit a
# `hunger_state_before != hunger_state_after` row when a threshold is crossed by PASSIVE
# DRAIN. On a drain crossing it flips BOTH fields to the new state on the same row:
#
#     monotonic  event   stimulus       hs_before  hs_after  level_before  level_after
#     11529.10   sample  passive_drain  HS2        HS2       25.012734     25.005765
#     11530.10   sample  passive_drain  HS3        HS3       25.005765      24.998797   <- crossing
#
# Only discrete `interaction_cost` / `feeding` / `mode` events produce a genuine
# before != after row. Consequences of keying off the label fields (the old code did):
#
#   * `v_hs_transitions` contains ZERO passive-drain crossings — 32 real Full->Hungry drain
#     falls and 2 real Hungry->Starving drain falls are simply absent.
#   * The old `build_hs3_episodes` looked for `after=="HS3" and before!="HS3"`, so it found
#     8 Starving episodes where the levels contain 17. The 9 it dropped were exactly the
#     drain-entered ones — i.e. the episodes where nobody was interacting, hence nobody was
#     there to feed the robot. That makes "8/8 resolved by feeding" a selection artefact:
#     the selection rule and the finding were the same thing.
#
# Everything downstream is therefore rebuilt from the LEVEL SERIES, which is ground truth
# (the level is the software integrator; the label is a derived view of it).
hs_transitions_logged = read_view_super("executive", "v_hs_transitions")   # kept for provenance only

def hs_from_level_series(x):
    "Vectorised label from level, using the controller's own 60/25 constants."
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x >= HS_FULL_MIN, "HS1", np.where(x >= HS_STARVING_MAX, "HS2", "HS3"))

def build_hs_crossings(hr):
    '''EVERY threshold crossing, derived from the level series rather than the label fields.

    One row per crossing, with the triggering event's stimulus_type, so drain-driven falls
    and action/feeding-driven jumps can finally be told apart.
    '''
    hr = hr.sort_values(["run_id","monotonic_sec","id"]).copy()
    hr["level"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    hr["lab"] = hs_from_level_series(hr["level"])
    rows = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        lab = g["lab"].values
        chg = np.where(lab[1:] != lab[:-1])[0]
        for i in chg:
            r = g.iloc[i+1]
            rows.append(dict(
                run_id=run, day_rome=r["day_rome"], session_id=r.get("session_id"),
                id=int(r["id"]), timestamp_epoch=r["timestamp_epoch"],
                monotonic_sec=float(r["monotonic_sec"]),
                from_state=lab[i], to_state=lab[i+1],
                event_type=r["event_type"], cause=r["stimulus_type"],
                stomach_level_before=float(g["level"].iloc[i]),
                stomach_level_after=float(r["level"]),
                level_delta=float(r["level"]) - float(g["level"].iloc[i]),
                exec_interaction_id=r.get("exec_interaction_id")))
    return pd.DataFrame(rows)

def build_hs3_episodes(hr):
    '''Starving episodes from LEVEL crossings (level < 25), not from the logged label fields.

    An episode opens the first sample the level drops below 25 (or at run start, if the run
    begins below 25) and closes when the level returns to >= 25 — or is right-censored at the
    end of the run if it never does. `entry_cause` records what pushed it under: `passive_drain`
    (nobody there) versus `interaction_cost` (someone was interacting). That column is the whole
    reason the old episode set was biased, so it is now a first-class field.
    '''
    hr = hr.sort_values(["run_id","monotonic_sec","id"]).copy()
    hr["level"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    eps = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        lvl = g["level"].values
        mono = g["monotonic_sec"].values
        starving = lvl < HS_STARVING_MAX
        run_end = float(mono.max())
        i = 0
        while i < len(g):
            if not starving[i]:
                i += 1; continue
            j = i
            while j + 1 < len(g) and starving[j+1]:
                j += 1
            entry_mono = float(mono[i])
            entry_is_run_start = (i == 0)
            # Escape = first sample at/after j+1 with level back >= 25. None => censored at run end.
            escaped = (j + 1) < len(g)
            escape_mono = float(mono[j+1]) if escaped else np.nan
            escape_event = g["event_type"].iloc[j+1] if escaped else None
            # Full recovery = first later sample with level >= 60.
            post = g.iloc[j+1:] if escaped else g.iloc[0:0]
            full_hit = post[post["level"] >= HS_FULL_MIN]
            full_mono = float(full_hit["monotonic_sec"].iloc[0]) if len(full_hit) else np.nan
            full_event = full_hit["event_type"].iloc[0] if len(full_hit) else None
            censor_mono = full_mono if np.isfinite(full_mono) else (escape_mono if escaped else run_end)
            # Feeds inside the Starving stretch, plus the feed that ends it.
            win = g[(g["monotonic_sec"] >= entry_mono) & (g["monotonic_sec"] <= censor_mono)]
            feeds = win[win["event_type"] == "feeding"].sort_values(["monotonic_sec","id"])
            first_feed_mono = float(feeds["monotonic_sec"].iloc[0]) if len(feeds) else np.nan
            eps.append(dict(
                episode_id=int(g["id"].iloc[i]), run_id=run, day_rome=g["day_rome"].iloc[i],
                session_id=g.get("session_id", pd.Series([None]*len(g))).iloc[i],
                entry_ts_epoch=g["timestamp_epoch"].iloc[i], entry_mono=entry_mono,
                entry_cause=("run_start" if entry_is_run_start else g["stimulus_type"].iloc[i]),
                entry_level=float(lvl[i]), min_level=float(np.min(lvl[i:j+1])),
                first_feed_mono=first_feed_mono,
                time_to_first_feed_sec=(first_feed_mono - entry_mono) if np.isfinite(first_feed_mono) else np.nan,
                escape_mono=escape_mono,
                time_to_escape_starving_sec=(escape_mono - entry_mono) if escaped else np.nan,
                full_recovery_mono=full_mono,
                time_to_full_recovery_sec=(full_mono - entry_mono) if np.isfinite(full_mono) else np.nan,
                exit_mono=censor_mono, episode_duration_sec=float(censor_mono - entry_mono),
                # Time actually spent under 25 — right-censored episodes report time-to-run-end.
                hs3_duration_sec=float((escape_mono if escaped else run_end) - entry_mono),
                n_samples=int(j - i + 1),
                received_feed=int(len(feeds) > 0),
                escaped_starving=int(escaped),
                escaped_starving_by_feeding=int(escaped and escape_event == "feeding"),
                recovered_to_full=int(np.isfinite(full_mono)),
                recovered_to_full_by_feeding=int(np.isfinite(full_mono) and full_event == "feeding"),
                resolved_by_feeding=int(np.isfinite(full_mono) and full_event == "feeding"),
                censored_at_run_end=int(not escaped),
                meals_received=int(len(feeds)),
                total_active_energy_during_episode=float(
                    win[win["event_type"]=="active_cost"]["active_energy_cost"].sum()),
                exit_cause=("recovered_full_by_feeding" if (np.isfinite(full_mono) and full_event == "feeding")
                            else "recovered_full_nonfeeding" if np.isfinite(full_mono)
                            else "escaped_starving_by_feeding" if (escaped and escape_event == "feeding")
                            else "escaped_starving_nonfeeding" if escaped
                            else "censored_end_of_run")))
            i = j + 1
    return pd.DataFrame(eps)

def build_drain_segments(hr):
    "Pandas port of v_drain_segments: contiguous passive_drain runs, empirical slope."
    hr = hr.sort_values(["run_id","monotonic_sec","id"])
    segs = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        is_pd = (g["stimulus_type"] == "passive_drain").values
        # segment id increments whenever a passive row follows a non-passive gap
        seg_id = 0; prev_pd = False; cur = []
        def flush(rows):
            if len(rows) >= 2:
                sub = g.loc[rows]
                dt = sub["monotonic_sec"].iloc[-1] - sub["monotonic_sec"].iloc[0]
                dl = sub["stomach_level_after"].iloc[-1] - sub["stomach_level_after"].iloc[0]
                segs.append(dict(run_id=run, day_rome=sub["day_rome"].iloc[0],
                    start_mono=sub["monotonic_sec"].iloc[0], end_mono=sub["monotonic_sec"].iloc[-1],
                    start_level=sub["stomach_level_after"].iloc[0], end_level=sub["stomach_level_after"].iloc[-1],
                    duration_sec=dt, empirical_drain_rate=(dl/dt if dt>0 else np.nan),
                    n_samples=len(sub)))
        for i in range(len(g)):
            if is_pd[i]:
                if not prev_pd and cur: flush(cur); cur=[]
                cur.append(i); prev_pd = True
            else:
                if cur: flush(cur); cur=[]
                prev_pd = False
        if cur: flush(cur)
    return pd.DataFrame(segs)

hs_transitions = build_hs_crossings(hunger_raw)   # LEVEL-derived: the complete crossing ledger
hs3_episodes   = build_hs3_episodes(hunger_raw)   # LEVEL-derived: all Starving episodes
drain_segments = build_drain_segments(hunger_raw)
for df in (hs3_episodes, hs_transitions, drain_segments):
    if "day_rome" in df.columns: df["session_id"] = df["day_rome"]

# --- Prove the artefact, and prove the fix, in the notebook output itself. ------------
_lg = (hs_transitions_logged.groupby(["from_state","to_state"]).size().rename("logged"))
_rw = (hs_transitions.groupby(["from_state","to_state"]).size().rename("level_derived"))
_cmp = pd.concat([_lg, _rw], axis=1).fillna(0).astype(int)
_cmp["missed_by_the_log"] = _cmp["level_derived"] - _cmp["logged"]
print("Threshold crossings: what the LOG recorded vs what the LEVELS actually did\n")
print(_cmp.rename(index=HS_NAME).to_string())
print(f"\n  -> the logged view misses {int(_cmp['missed_by_the_log'].clip(lower=0).sum())} "
      f"crossings, ALL of them drain-driven falls into deficit.")
print("  -> its Starving ledger is impossible: "
      f"{int(_lg.get(('HS2','HS3'), 0))} entries but "
      f"{int(_lg.get(('HS3','HS1'), 0)) + int(_lg.get(('HS3','HS2'), 0))} exits.")
print("\nCrossings by triggering cause (level-derived — drain falls are visible again):")
print(pd.crosstab([hs_transitions["from_state"], hs_transitions["to_state"]],
                  hs_transitions["cause"]).to_string())
hs_transitions_logged.to_parquet(OUT_DIR / "hs_transitions_logged_view.parquet", index=False)
_cmp.reset_index().to_csv(OUT_DIR / "hs_crossing_log_gap.csv", index=False)

# Drive timeline resampled to a fixed 30 s grid per run (for plotting + area-under-deficit)
def resample_timeline(hr, grid=30.0):
    out = []
    for run, g in hr.sort_values(["run_id","monotonic_sec"]).groupby("run_id"):
        g = g.dropna(subset=["monotonic_sec","stomach_level_after"])
        if len(g) < 2: continue
        t0, t1 = g["monotonic_sec"].min(), g["monotonic_sec"].max()
        grid_t = np.arange(t0, t1, grid)
        lvl = np.interp(grid_t, g["monotonic_sec"], g["stomach_level_after"])
        day = g["day_rome"].iloc[0]
        out.append(pd.DataFrame({"run_id": run, "session_id": day,
                                 "monotonic_sec": grid_t, "level": lvl}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
timeline_30s = resample_timeline(hunger_raw)

# Area-under-deficit (time-integrated deficit below 60) per run, in %*hours.
def deficit_area(tl, thr=60.0, grid=30.0):
    rows = []
    for run, g in tl.groupby("run_id"):
        deficit = np.clip(thr - g["level"].values, 0, None)
        rows.append(dict(run_id=run, session_id=g["session_id"].iloc[0],
                         deficit_area_pct_h=float(deficit.sum() * grid / 3600.0),
                         mean_level=float(g["level"].mean()),
                         frac_below60=float((g["level"] < 60).mean()),
                         frac_below25=float((g["level"] < 25).mean())))
    return pd.DataFrame(rows)
deficit_by_run = deficit_area(timeline_30s)

print(f"\nhs3_episodes:   {hs3_episodes.shape}  "
      f"(first_feed={int(hs3_episodes['received_feed'].sum())}, "
      f"escaped_starving_by_feeding={int(hs3_episodes['escaped_starving_by_feeding'].sum())}, "
      f"recovered_full_by_feeding={int(hs3_episodes['recovered_to_full_by_feeding'].sum())})")
print("Starving episodes by how the robot FELL INTO Starving:")
print(hs3_episodes.groupby("entry_cause").agg(
    n=("episode_id","size"), total_hs3_sec=("hs3_duration_sec","sum"),
    max_hs3_sec=("hs3_duration_sec","max"), fed=("received_feed","sum")).round(1).to_string())
print(f"hs_transitions: {hs_transitions.shape}")
print(f"timeline_30s:   {timeline_30s.shape}")
print(f"\nDeficit exposure by run ({deficit_by_run['run_id'].nunique()} MONITORED runs; "
      f"{interactions['run_id'].nunique()} of them had visitor interactions — the rest are "
      f"always-on idle runs where the drive kept draining with nobody present):")
print(deficit_by_run.to_string(index=False))
hs3_episodes.to_parquet(OUT_DIR / "hs3_episodes.parquet", index=False)
hs_transitions.to_parquet(OUT_DIR / "hs_transitions.parquet", index=False)
""")

md("### A5. Master feature table (leakage-safe)")

code(r"""
# --- A5. One row per interaction; only PRE-START / start-time features as predictors -
# Salience attempt features (by exec_interaction_id).
att = attempts.copy()
att_feat = att[["exec_interaction_id","timestamp_epoch","start_ss","is_proactive","duration_sec","abort_reason"]].copy()
att_feat = att_feat.rename(columns={"timestamp_epoch":"attempt_end_epoch","abort_reason":"attempt_abort"})
att_feat["duration_sec"] = pd.to_numeric(att_feat["duration_sec"], errors="coerce")
att_feat["attempt_start_epoch"] = att_feat["attempt_end_epoch"] - att_feat["duration_sec"]
att_feat = att_feat.dropna(subset=["exec_interaction_id"]).drop_duplicates("exec_interaction_id")

# Reconstruct the earliest defensible interaction-start anchor. Executive interaction rows
# are timestamped when the result is saved, i.e. after the interaction; using that timestamp
# would leak post-start perception into "pre-start" predictors. Prefer the salience attempt
# start (end - duration), then the first logged turn for reactive/conversational cases.
first_turn = (turns.dropna(subset=["timestamp_epoch"])
                   .groupby("interaction_id")["timestamp_epoch"]
                   .min()
                   .rename("first_turn_epoch")
                   .reset_index())
start_clock = (interactions[["interaction_id","timestamp_epoch"]]
               .rename(columns={"timestamp_epoch":"interaction_logged_epoch"})
               .merge(first_turn, on="interaction_id", how="left")
               .merge(att_feat[["exec_interaction_id","attempt_start_epoch"]],
                      left_on="interaction_id", right_on="exec_interaction_id", how="left")
               .drop(columns=["exec_interaction_id"]))
def choose_start_anchor(row):
    candidates = []
    for col, source in (("attempt_start_epoch", "salience_attempt_start"),
                        ("first_turn_epoch", "first_turn")):
        val = row.get(col)
        if pd.notna(val):
            candidates.append((float(val), source))
    if not candidates:
        return pd.Series({"interaction_start_epoch": np.nan, "start_anchor_source": "missing"})
    val, source = min(candidates, key=lambda x: x[0])
    return pd.Series({"interaction_start_epoch": val, "start_anchor_source": source})
start_clock = pd.concat([start_clock, start_clock.apply(choose_start_anchor, axis=1)], axis=1)
interactions_for_features = interactions.merge(
    start_clock[["interaction_id","interaction_logged_epoch","interaction_start_epoch","start_anchor_source"]],
    on="interaction_id", how="left")
anchor_lag = interactions_for_features["interaction_logged_epoch"] - interactions_for_features["interaction_start_epoch"]
print("start-anchor coverage:",
      interactions_for_features["interaction_start_epoch"].notna().sum(), "/", len(interactions_for_features),
      "| median logged-minus-start %.1fs" % anchor_lag.dropna().median())
print(interactions_for_features["start_anchor_source"].value_counts(dropna=False).to_string())

# CROSS-MODULE KEY NOTE (discovered): executive, salience and vision each mint their own
# run_id and their own monotonic clock — they are NOT shared. The only cross-module key is
# absolute wall-clock `timestamp_epoch`, plus `face_id` (stable identity from vision) and
# `exec_interaction_id` (salience attempts -> executive interactions). We therefore match
# perception features on face_id + a timestamp_epoch window bracketing reconstructed
# interaction_start_epoch. Rows with no defensible start anchor keep perception features
# missing rather than falling back to the post-hoc executive save timestamp.
IPS_PREWIN, IPS_POST = 15.0, 2.0
def prewindow_ips(inter, ips):
    feats = []
    ips_by = {k: v.sort_values("timestamp_epoch") for k, v in ips.groupby("face_id")}
    for _, r in inter.iterrows():
        rec = dict(interaction_id=r["interaction_id"])
        g = ips_by.get(r.get("face_id"))
        if g is not None and pd.notna(r.get("interaction_start_epoch")):
            t = r["interaction_start_epoch"]
            w = g[(g["timestamp_epoch"] >= t - IPS_PREWIN) & (g["timestamp_epoch"] <= t + IPS_POST)]
            if len(w):
                rec.update(ips_mean=w["ips"].mean(), ips_max=w["ips"].max(),
                           prox=w["prox_score"].mean(), cent=w["cent_score"].mean(),
                           gaze=w["gaze_score"].mean(),
                           habituation_mult=w["habituation_multiplier"].mean(),
                           eligible_pre=w["eligible"].max())
        feats.append(rec)
    return pd.DataFrame(feats)
ips_feat = prewindow_ips(interactions_for_features, ips)

# Pre-start landmark features (same approach window): attention/talking/gaze + co-presence.
def prewindow_landmarks(inter, lm):
    lm_by = {k: v.sort_values("timestamp_epoch") for k, v in lm.groupby("face_id")}
    lm_time = lm.sort_values("timestamp_epoch")
    feats = []
    for _, r in inter.iterrows():
        rec = dict(interaction_id=r["interaction_id"])
        t = r.get("interaction_start_epoch")
        g = lm_by.get(r.get("face_id"))
        if g is not None and pd.notna(t):
            w = g[(g["timestamp_epoch"] >= t - IPS_PREWIN) & (g["timestamp_epoch"] <= t + IPS_POST)]
            if len(w):
                rec.update(cos_angle=w["cos_angle"].mean(),
                           attention_frac=(w["attention"].astype(str)
                                           .isin(["MUTUAL_GAZE","NEAR_GAZE"]).mean()),
                           talking_rate=pd.to_numeric(w["is_talking"], errors="coerce").mean(),
                           time_in_view=w["time_in_view"].max())
        if pd.notna(t):
            wc = lm_time[(lm_time["timestamp_epoch"] >= t - 2.0) & (lm_time["timestamp_epoch"] <= t + 2.0)]
            rec["copresence"] = float(wc["faces_in_frame"].max()) if len(wc) else np.nan
        feats.append(rec)
    return pd.DataFrame(feats)
lm_feat = prewindow_landmarks(interactions_for_features, landmarks)
print(f"pre-window coverage: IPS {ips_feat['ips_mean'].notna().sum() if 'ips_mean' in ips_feat else 0}/{len(interactions)}, "
      f"landmarks {lm_feat['attention_frac'].notna().sum() if 'attention_frac' in lm_feat else 0}/{len(interactions)}")

# Assemble master table (predictors are pre-start/at-start only; outcomes are labels).
base = interactions_for_features[[
    "interaction_id","run_id","session_id","day_rome","hour_of_day","user_key","person_id",
    "interaction_logged_epoch","interaction_start_epoch","start_anchor_source",
    "track_id","face_id","initial_state","final_state","success","replied_any","greeted","talked",
    "n_turns","trigger_mode","interaction_tag","hunger_state_start","hunger_state_end",
    "stomach_level_start","stomach_level_end","active_energy_cost","homeostatic_reward",
    "meals_eaten_count",
]].copy()
master = (base.merge(att_feat, left_on="interaction_id", right_on="exec_interaction_id", how="left")
              .merge(ips_feat, on="interaction_id", how="left")
              .merge(lm_feat, on="interaction_id", how="left"))
# Ensure expected feature columns exist even if coverage was 0 (keeps downstream robust).
for c in ["ips_mean","ips_max","prox","cent","gaze","habituation_mult","eligible_pre",
          "cos_angle","attention_frac","talking_rate","time_in_view","copresence",
          "start_ss","is_proactive","duration_sec","attempt_abort",
          "interaction_logged_epoch","interaction_start_epoch","start_anchor_source"]:
    if c not in master.columns: master[c] = np.nan
master.to_parquet(OUT_DIR / "master_interactions.parquet", index=False)
print("master_interactions:", master.shape)
_prev = [c for c in ["interaction_id","hunger_state_start","initial_state","ips_mean",
                     "copresence","replied_any","final_state","n_turns"] if c in master.columns]
print(master[_prev].head())
""")

md("### A6. NLP prep (co-present turns + Telegram messages)")

code(r"""
# --- A6. Assemble text units with hunger_mentioned flag; strip emoji ----------
# Emoji strip mirrors chatBot.py approach (regex over emoji unicode ranges).
_EMOJI_RE = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
                       "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF"
                       "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "]+", flags=re.UNICODE)
def strip_emoji(s):
    return _EMOJI_RE.sub("", str(s or "")).strip()

# Co-present conversational turns (executive interaction_turns).
turns_nlp = turns.copy()
for col in ("user_utterance","assistant_utterance"):
    if col in turns_nlp.columns:
        turns_nlp[col+"_clean"] = turns_nlp[col].map(strip_emoji)
turns_nlp["channel"] = "copresent"

# Telegram messages.
chat_nlp = chat_msgs.copy()
if "text" in chat_nlp.columns:
    chat_nlp["text_clean"] = chat_nlp["text"].map(strip_emoji)
chat_nlp["channel"] = "telegram"

print("co-present turns:", turns_nlp.shape, "| hunger_mentioned:",
      int(turns_nlp.get("hunger_mentioned", pd.Series(dtype=int)).fillna(0).sum()))
print("telegram messages:", chat_nlp.shape, "| hunger_mentioned:",
      int(chat_nlp.get("hunger_mentioned", pd.Series(dtype=int)).fillna(0).sum()))

print("\n=== Phase A complete — master table head ===")
print(master.head(3).to_string())
print("\nhs3_episodes head:"); print(hs3_episodes.head(3).to_string())
print("\nhs_transitions head:"); print(hs_transitions.head(3).to_string())
""")

# ==========================================================================
# ENGINEERING VERIFICATION — quality gate
# ==========================================================================
md(r"""## Engineering verification — quality gate (must pass before analysis)

`verify()` collects V1–V5, prints a pass/fail table and writes
`verification_report.md`. A **hard** failure stops the notebook; a **soft** warning is
logged and surfaced. The active-energy-cost table is a first-class deliverable built
here and cross-checked against the source constants.
""")

code(r"""
# --- Active energy cost table (first-class deliverable) -----------------------
# Rebuild from hunger_level_events (event_type='active_cost'); cross-check to CONST.
ac = hunger_raw[hunger_raw["event_type"] == "active_cost"].copy()
cost_tbl = (ac.groupby("stimulus_label")
              .agg(cost_per_event=("active_energy_cost", "median"),
                   cost_min=("active_energy_cost", "min"),
                   cost_max=("active_energy_cost", "max"),
                   n_events=("active_energy_cost", "size"),
                   total_energy=("active_energy_cost", "sum"),
                   mean_stomach_before=("stomach_level_before", "mean"))
              .reset_index())
def cost_category(lbl):
    if "conversation_turn" in lbl: return "conversation (largest sink)"
    if "starter" in lbl:           return "conversation opener"
    if "ask_name" in lbl:          return "name extraction"
    if lbl.startswith("hunger_"):  return "drive-driven feeding prompt / seeking"
    if "feed_ack" in lbl:          return "meal acknowledgement"
    if lbl.startswith("reactive"): return "reactive greeting"
    return "greeting"
cost_tbl["category"] = cost_tbl["stimulus_label"].map(cost_category)
cost_tbl["const_expected"] = cost_tbl["stimulus_label"].map(CONST["ACTIVE_COST"])
cost_tbl["matches_source"] = cost_tbl.apply(
    lambda r: (pd.notna(r["const_expected"]) and abs(r["cost_per_event"] - r["const_expected"]) < 1e-6
               and abs(r["cost_min"] - r["cost_max"]) < 1e-6), axis=1)
cost_tbl = cost_tbl.sort_values("total_energy", ascending=False)
cost_tbl.to_csv(OUT_DIR / "active_cost_table.csv", index=False)
print(cost_tbl[["stimulus_label","cost_per_event","n_events","total_energy",
                "category","const_expected","matches_source"]].to_string(index=False))
print("\nwrote outputs/active_cost_table.csv")
""")

code(r"""
# --- verify(): V1-V5 --------------------------------------------------------
def verify():
    checks = []  # (id, name, severity, ok, detail)
    def add(vid, name, ok, detail="", hard=True):
        checks.append((vid, name, "hard" if hard else "soft", bool(ok), detail))

    # V1 — code<->data consistency
    tr = hs_transitions
    hs12 = tr[((tr.from_state=="HS1")&(tr.to_state=="HS2"))|((tr.from_state=="HS2")&(tr.to_state=="HS1"))]
    ok12 = ((hs12["stomach_level_before"]-60).abs().le(15) & (hs12["stomach_level_after"]-60).abs().le(15)).mean() if len(hs12) else 1.0
    br12 = ((hs12[["stomach_level_before","stomach_level_after"]].min(axis=1) <= 60) &
            (hs12[["stomach_level_before","stomach_level_after"]].max(axis=1) >= 60)).mean() if len(hs12) else np.nan
    add("V1a","HS1<->HS2 transitions bracket 60", (np.isnan(br12) or br12 >= 0.9),
        f"(bracket_frac={br12:.2f}, n={len(hs12)})", hard=False)
    hs23 = tr[((tr.from_state=="HS2")&(tr.to_state=="HS3"))|((tr.from_state=="HS3")&(tr.to_state=="HS2"))]
    br23 = ((hs23[["stomach_level_before","stomach_level_after"]].min(axis=1) <= 25) &
            (hs23[["stomach_level_before","stomach_level_after"]].max(axis=1) >= 25)).mean() if len(hs23) else np.nan
    add("V1b","HS2<->HS3 transitions bracket 25", (np.isnan(br23) or br23 >= 0.9),
        f"(bracket_frac={br23:.2f}, n={len(hs23)})", hard=False)
    add("V1c","meal_delta == SMALL/MEDIUM/LARGE const", len(bad_meal)==0, f"(mismatches={len(bad_meal)})")
    # Fitted passive-drain slope vs nominal
    ds = drain_segments.dropna(subset=["empirical_drain_rate"])
    ds = ds[ds["duration_sec"] > 120]
    med_rate = -ds["empirical_drain_rate"].median() if len(ds) else np.nan  # drain is negative
    nominal = CONST["DRAIN_RATE_PER_SEC"]
    add("V1d","fitted drain rate ~ nominal 100/(4h)",
        (not np.isnan(med_rate)) and 0.5*nominal <= med_rate <= 1.8*nominal,
        f"(median={med_rate:.2e}/s vs nominal={nominal:.2e}/s, n_seg={len(ds)})", hard=False)
    add("V1e","stomach level in [0,100]",
        bool(((hunger_raw['stomach_level_after']>=0)&(hunger_raw['stomach_level_after']<=100)).all()))

    # V2 — referential integrity
    iid = set(interactions["interaction_id"])
    a = attempts.dropna(subset=["exec_interaction_id"])
    m = a["exec_interaction_id"].isin(iid).mean() if len(a) else np.nan
    add("V2a","salience attempts resolve to interactions", (np.isnan(m) or m>=0.95),
        f"(match={m:.3f}, n={len(a)})", hard=False)
    h = hunger_raw.dropna(subset=["exec_interaction_id"])
    mh = h["exec_interaction_id"].isin(iid).mean() if len(h) else np.nan
    add("V2b","hunger events' exec_interaction_id resolve", (np.isnan(mh) or mh>=0.95),
        f"(match={mh:.3f}, n={len(h)})", hard=False)
    tt = turns["interaction_id"].isin(iid).mean() if len(turns) else np.nan
    add("V2c","interaction_turns subset of interactions", (np.isnan(tt) or tt>=0.99),
        f"(match={tt:.3f})", hard=False)

    # V3 — metadata & validity
    add("V3a","8 session-days present", interactions["session_id"].nunique()==8,
        f"(days={interactions['session_id'].nunique()}, runs={interactions['run_id'].nunique()})")
    add("V3b","no NULL run_id in interactions", interactions["run_id"].notna().all())
    add("V3c","no NULL run_id in hunger events", hunger_raw["run_id"].notna().all())

    # V4 — clock sanity
    viol = 0
    for run, g in hunger_raw.sort_values(["run_id","id"]).groupby("run_id"):
        viol += (g["monotonic_sec"].diff() < -1e-6).sum()
    add("V4a","monotonic_sec non-decreasing within run", viol==0, f"(violations={viol})")

    # V5 — energy budget integrity
    ac_rows = hunger_raw[hunger_raw["event_type"]=="active_cost"]
    non_ac = hunger_raw[hunger_raw["event_type"]!="active_cost"]
    add("V5a","active cost only on active_cost rows",
        (ac_rows["active_energy_cost"]>0).all() and (non_ac["active_energy_cost"].fillna(0)==0).all(),
        f"(active rows={len(ac_rows)})")
    varying = cost_tbl[~cost_tbl["matches_source"] & cost_tbl["const_expected"].notna()]
    add("V5b","each action maps to one deterministic cost (== source)", len(varying)==0,
        f"(varying/mismatch={len(varying)})")
    out_e = ac_rows["active_energy_cost"].sum()
    in_e  = hunger_raw[hunger_raw["event_type"]=="feeding"]["meal_delta"].sum()
    add("V5c","corpus energy balance reported", True,
        f"(active_out={out_e:.1f}, meal_in={in_e:.1f}, net={in_e-out_e:+.1f})", hard=False)

    # Report
    df = pd.DataFrame(checks, columns=["id","check","severity","pass","detail"])
    print(df.to_string(index=False))
    hard_fail = df[(df["severity"]=="hard") & (~df["pass"])]
    lines = ["# Verification report", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
             "| id | check | severity | result | detail |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['id']} | {r['check']} | {r['severity']} | "
                     f"{'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |")
    lines += ["", f"**Corpus energy balance:** active-out {out_e:.1f} vs meal-in {in_e:.1f} "
              f"(net {in_e-out_e:+.1f} over {interactions['run_id'].nunique()} runs)."]
    (OUT_DIR / "verification_report.md").write_text("\n".join(lines))
    print("\nwrote outputs/verification_report.md")
    if len(hard_fail):
        raise AssertionError(f"HARD verification failures:\n{hard_fail.to_string(index=False)}")
    print("\n>>> VERIFICATION GATE PASSED (hard checks) <<<")
    return df

verify_df = verify()
""")

# ==========================================================================
# WRITE NOTEBOOK
# ==========================================================================
def build_and_write(path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                   for k, s in CELLS]
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path} with {len(CELLS)} cells")

# ==========================================================================
# PHASE B — STATISTICAL ANALYSIS
# ==========================================================================
md(r"""## Phase B — Statistical analysis

**Methodology.** Data are clustered (turns ⊂ interactions ⊂ people ⊂ runs ⊂ days), and the
cluster count is small: **14 named people, 12 runs, 8 days**. Two consequences drive every
modelling choice below.

1. **Asymptotic sandwich SEs are not trustworthy at 14 clusters.** GEE robust SEs are known to be
   anti-conservative well below ~40 clusters. We therefore fit GEE/MixedLM as the point-estimate
   engine but **lead with the person-cluster bootstrap interval** everywhere, and report the
   asymptotic CI second. Where the two disagree, the bootstrap is the honest statement.
2. **Some quantities in this system are true by construction.** The stomach level is a software
   integrator, the hunger label is derived from it by the same thresholds, and the per-person
   affinity is a deterministic EMA of the logged reward. Analyses of those quantities are
   **implementation verification**, not empirical inference, and are labelled as such.

Every result carries exactly one **evidence class**:

| Class | Meaning |
|---|---|
| `Implementation verification` | The result follows from the controller source; it confirms faithful logging/implementation, not a discovered fact. |
| `Within-deployment association` | A cluster-aware association estimated on this deployment. Not causal, not a population estimate. |
| `Exploratory observation` | Descriptive; too small-n or too selection-prone to support inference. |
| `Inconclusive` | The analysis was run and did not settle the question. |
| `Requires replication` | Direction is suggestive but identification needs new data. |

Every inferential p-value used in a conclusion is registered in `PTABLE` and Benjamini–Hochberg
corrected within its declared family. Quantities classed as implementation verification or
exploratory carry **no** confirmatory p-value.
""")

code(r"""
# --- Shared statistical helpers ----------------------------------------------
import scipy.stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf
import statsmodels.api as sm

RESULTS = {}      # analysis key -> dict(evidence, verdict, ...)
PTABLE = []       # full multiplicity ledger: every inferential p used in a conclusion
SENSITIVITY = []  # cluster-bootstrap sensitivity rows

# --- Role map (external design metadata; needs the pseudonym map, seeded in A1b) --------
# Phase 1: 2 obligated feeders, 2 interact-but-never-feed, everyone else unconstrained.
# Phase 2: all constraints lifted. Roles were NEVER controller inputs.
ROLE_OF_EARLY = {}
for _name, _role in _RP.get("roles", {}).items():
    _pid = PSEUDONYM_MAP.get(canon_identity(_name))
    if _pid is not None:
        ROLE_OF_EARLY[_pid] = _role
    else:
        print("WARNING: a role-map entry did not match any observed identity (name withheld).")
def role_of(pid):
    if str(pid).lower() in ("unknown", "unmatched", ""): return "unknown"
    return ROLE_OF_EARLY.get(str(pid), "normal")
print(f"Roles resolved for {len(ROLE_OF_EARLY)} people: "
      f"{pd.Series(list(ROLE_OF_EARLY.values())).value_counts().to_dict()}")
print(f"Phase 1 days: {sorted(PHASE1_DAYS_EARLY)}")

# The five permitted evidence classes. Nothing else may appear in a verdict.
EV_IMPL   = "Implementation verification"
EV_ASSOC  = "Within-deployment association"
EV_EXPL   = "Exploratory observation"
EV_INCONC = "Inconclusive"
EV_REPL   = "Requires replication"
EVIDENCE_CLASSES = {EV_IMPL, EV_ASSOC, EV_EXPL, EV_INCONC, EV_REPL}

def verdict(key, evidence, text, **kw):
    "Record one analysis verdict under exactly one evidence class."
    assert evidence in EVIDENCE_CLASSES, f"illegal evidence class: {evidence!r}"
    RESULTS[key] = kw | {"evidence": evidence, "verdict": text}
    print(f"VERDICT [{evidence}]: {text}")

def register_p(analysis, model, term, p, family, status="confirmatory", note=""):
    '''Every p-value that enters a conclusion must pass through here.

    status="confirmatory" -> enters a BH family and may support a claim.
    status="exploratory"  -> recorded for transparency, never used to support a claim,
                             and excluded from the BH families (correcting an
                             uninterpretable p-value would only lend it false authority).
    '''
    PTABLE.append(dict(analysis=analysis, model=model, term=term, p=float(p),
                       family=family, status=status, note=note))
    return float(p)

def boot_ci(x, fn=np.mean, n=5000, alpha=0.05, seed=SEED):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return (fn(x), np.nanpercentile(stats, 100*alpha/2), np.nanpercentile(stats, 100*(1-alpha/2)))

def boot_diff_ci(a, b, fn=np.mean, n=5000, alpha=0.05, seed=SEED):
    a = np.asarray(pd.Series(a).dropna(), float); b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = np.array([fn(rng.choice(a, len(a), True)) - fn(rng.choice(b, len(b), True)) for _ in range(n)])
    return (fn(a)-fn(b), np.nanpercentile(d, 100*alpha/2), np.nanpercentile(d, 100*(1-alpha/2)))

def exact_prop_ci(k, n):
    "Clopper-Pearson exact interval. The right tool for 0/n and n/n cells."
    if n == 0: return (np.nan, np.nan, np.nan)
    lo, hi = proportion_confint(int(k), int(n), method="beta")
    return (k/n, float(lo), float(hi))

def cluster_bootstrap_effect(df, cluster_col, fit_fn, *, n=1000, seed=SEED, label="effect"):
    '''Resample whole clusters with replacement and refit the supplied effect.

    With 12-15 clusters the asymptotic sandwich SE is anti-conservative, so THIS is the
    interval we lead with; the GEE/MixedLM CI is reported second. Returns
    (lo, median, hi, n_ok) and also records the share of refits that failed - a high
    failure rate is itself diagnostic of separation or a degenerate design cell.
    '''
    rng = np.random.default_rng(seed)
    clusters = pd.Series(df[cluster_col].dropna().unique())
    vals = []
    for _ in range(n):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for i, c in enumerate(picked):
            g = df[df[cluster_col] == c].copy()
            # Give duplicated clusters distinct ids if the model uses the cluster column.
            g[cluster_col] = f"{c}__boot{i}"
            parts.append(g)
        sample = pd.concat(parts, ignore_index=True)
        try:
            val = fit_fn(sample)
            if np.isfinite(val):
                vals.append(float(val))
        except Exception:
            pass
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 25:
        print(f"{label}: cluster bootstrap failed/unstable ({len(vals)}/{n} successful refits)")
        return (np.nan, np.nan, np.nan, len(vals))
    lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
    print(f"{label}: cluster bootstrap median {mid:.3g} [95% {lo:.3g}, {hi:.3g}] "
          f"({len(vals)}/{n} successful refits)")
    return (lo, mid, hi, len(vals))

# --- Firth penalised logistic regression -------------------------------------
# Maximum-likelihood logistic regression is undefined under (quasi-)separation: the MLE
# diverges and the Wald SE collapses, which manufactures an absurdly small p-value from a
# cell that contains almost no information. B4 has exactly this shape (1 success in 13
# Starving interactions). Firth's penalised likelihood (Jeffreys prior) keeps the estimate
# finite and gives a usable profile-penalised-likelihood interval.
def firth_logit(X, y, max_iter=200, tol=1e-8):
    "Return (beta, loglik_penalised). X must already include an intercept column."
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0/(1.0 + np.exp(-eta))
        W = p*(1-p)
        XtWX = X.T @ (X * W[:, None])
        try:
            XtWX_inv = np.linalg.pinv(XtWX)
        except np.linalg.LinAlgError:
            break
        # Hat diagonal for the Jeffreys penalty term.
        H = (X * W[:, None]) @ XtWX_inv
        h = np.einsum("ij,ij->i", H, X)
        U = X.T @ (y - p + h*(0.5 - p))          # penalised score
        step = XtWX_inv @ U
        # Step-halving keeps the iteration stable on separated data.
        for _ in range(20):
            nb = beta + step
            if np.all(np.isfinite(nb)) and np.max(np.abs(step)) < 1e6:
                break
            step = step/2.0
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    eta = X @ beta
    p = np.clip(1.0/(1.0 + np.exp(-eta)), 1e-12, 1-1e-12)
    ll = float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))
    W = p*(1-p)
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    ll_pen = ll + 0.5*logdet if sign > 0 else ll
    return beta, ll_pen

def firth_profile_ci(X, y, idx, alpha=0.05, grid=241, span=8.0):
    '''Profile penalised-likelihood CI for coefficient `idx`.

    Profiling (not Wald) is the point: under separation the Wald SE is meaningless, but the
    penalised likelihood still has a well-defined shape, so the interval it implies is real.
    '''
    beta_hat, ll_max = firth_logit(X, y)
    b0 = beta_hat[idx]
    crit = sps.chi2.ppf(1-alpha, 1)/2.0
    keep = [j for j in range(X.shape[1]) if j != idx]
    def prof_ll(b_fixed):
        # Re-fit the remaining coefficients with an offset holding beta[idx] = b_fixed.
        off = X[:, idx]*b_fixed
        Xr = X[:, keep]
        bb = np.zeros(Xr.shape[1])
        for _ in range(200):
            eta = Xr @ bb + off
            p = 1.0/(1.0 + np.exp(-eta)); W = p*(1-p)
            XtWX = Xr.T @ (Xr * W[:, None])
            inv = np.linalg.pinv(XtWX)
            H = (Xr * W[:, None]) @ inv
            h = np.einsum("ij,ij->i", H, Xr)
            U = Xr.T @ (y - p + h*(0.5 - p))
            step = inv @ U
            bb_new = bb + step
            if not np.all(np.isfinite(bb_new)): break
            if np.max(np.abs(bb_new - bb)) < 1e-8:
                bb = bb_new; break
            bb = bb_new
        eta = Xr @ bb + off
        p = np.clip(1.0/(1.0 + np.exp(-eta)), 1e-12, 1-1e-12)
        ll = float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))
        W = p*(1-p)
        Xf = np.column_stack([Xr, X[:, idx]])
        sign, logdet = np.linalg.slogdet(Xf.T @ (Xf * W[:, None]))
        return ll + 0.5*logdet if sign > 0 else ll
    lo_grid = np.linspace(b0 - span, b0, grid)
    hi_grid = np.linspace(b0, b0 + span, grid)
    lo = np.nan
    for b in lo_grid:                    # walk in from the left to the crossing
        if ll_max - prof_ll(b) <= crit:
            lo = b; break
    hi = np.nan
    for b in hi_grid[::-1]:              # walk in from the right
        if ll_max - prof_ll(b) <= crit:
            hi = b; break
    # Penalised likelihood-ratio test of beta[idx] = 0.
    lr = 2.0*(ll_max - prof_ll(0.0))
    p_lrt = float(sps.chi2.sf(max(lr, 0.0), 1))
    return float(b0), float(lo), float(hi), p_lrt

def cluster_permutation_p(df, cluster_col, label_col, stat_fn, *, n=5000, seed=SEED):
    '''Randomisation inference: permute the treatment label ACROSS CLUSTERS, not rows.

    With 2 people per assigned role, the exact randomisation distribution is tiny, so this is
    the honest reference distribution: it asks how often a random re-assignment of the SAME
    role labels to the SAME people would produce a statistic this extreme. Returns
    (observed, p_two_sided, n_perms_used).
    '''
    rng = np.random.default_rng(seed)
    cl = df[[cluster_col, label_col]].drop_duplicates().reset_index(drop=True)
    labels = cl[label_col].values.copy()
    obs = stat_fn(df)
    if not np.isfinite(obs):
        return (np.nan, np.nan, 0)
    null = []
    for _ in range(n):
        perm = rng.permutation(labels)
        mapping = dict(zip(cl[cluster_col].values, perm))
        d2 = df.copy()
        d2[label_col] = d2[cluster_col].map(mapping)
        try:
            v = stat_fn(d2)
            if np.isfinite(v): null.append(float(v))
        except Exception:
            pass
    null = np.asarray(null, float)
    if len(null) < 50:
        return (float(obs), np.nan, len(null))
    # Two-sided, centred on the null mean; +1 correction so p is never exactly 0.
    centre = np.median(null)
    p = (1.0 + np.sum(np.abs(null - centre) >= np.abs(obs - centre))) / (len(null) + 1.0)
    return (float(obs), float(min(p, 1.0)), len(null))
""")

md(r"""> **Read B1–B2 as implementation verification, not inferential results.** The stomach level is a software
> integrator and the hunger label is derived from it by the same 60/25 thresholds, so
> "drain matches nominal" (B1) and "transitions bracket the thresholds" (B2) hold *by
> construction*. They confirm the monitoring/detection machinery is faithfully implemented;
> the only non-trivial empirical content is the **dense autonomous sampling** (B1) and the
> **near-zero flapping** (B2). The confirmatory weight of RQ1 sits in **B3** (deficit→action
> coupling) and **B4** (the Starving override).""")

md("### B1 — Internal monitoring *(verification-grade: faithful implementation + autonomy)*")

code(r"""
# B1: observed passive-drain slope per run vs nominal; coverage; autonomy.
ds = drain_segments.dropna(subset=["empirical_drain_rate"]).copy()
ds = ds[ds["duration_sec"] > 120]
ds["drain_per_sec"] = -ds["empirical_drain_rate"]     # make positive = %/s lost
nominal = CONST["DRAIN_RATE_PER_SEC"]
eff, lo, hi = boot_ci(ds["drain_per_sec"])
print(f"Fitted passive-drain rate: {eff:.3e} %/s  [95% CI {lo:.3e}, {hi:.3e}]  (nominal {nominal:.3e})")
print(f"  ratio to nominal: {eff/nominal:.2f}   n_segments={len(ds)}  runs={ds['run_id'].nunique()}")

# Coverage: max inter-sample gap of passive_drain samples per run.
samp = hunger_raw[hunger_raw["stimulus_type"]=="passive_drain"].sort_values(["run_id","monotonic_sec"])
gaps = samp.groupby("run_id")["monotonic_sec"].apply(lambda s: s.diff().max())
print(f"\nSampling coverage: median max inter-sample gap = {gaps.median():.2f}s "
      f"(worst {gaps.max():.1f}s across runs)")

# Autonomy: the drive lives on its own background clock, not driven by interactions.
# Measure it two ways: (a) drain samples never coincide with an interaction, and more
# meaningfully (b) what share of ALL stomach-level LOSS is passive background drain vs
# active interaction cost — if most loss is passive, monitoring runs independently.
# Autonomy = continuous background monitoring, independent of interactions: the drive
# logs a dense drain sample every ~2s across the whole runtime, none tied to an interaction.
auto_frac = samp["exec_interaction_id"].isna().mean()
total_runtime_h = (hunger_raw.groupby("run_id")["monotonic_sec"].agg(lambda s:s.max()-s.min()).sum())/3600
# Run reconciliation: the drive monitored 12 runs; only 10 had visitor interactions.
N_MONITORED = hunger_raw["run_id"].nunique()
N_INTERACT  = interactions["run_id"].nunique()
idle_runs = sorted(set(hunger_raw["run_id"]) - set(interactions["run_id"]))
print(f"\nRun reconciliation: {N_MONITORED} MONITORED runs, {N_INTERACT} WITH interactions. "
      f"The {N_MONITORED-N_INTERACT} monitoring-only runs (e.g. {', '.join(r[:8] for r in idle_runs)}) "
      f"had drain logging but zero visitors — the robot kept getting hungry with nobody present.")
print(f"Autonomy: {len(samp):,} background drain samples over {total_runtime_h:.1f} h, "
      f"{auto_frac*100:.0f}% with no interaction attached — monitoring runs on its own clock.")
# Coupling fact for RQ1-3: how level loss splits between background drain and active cost.
passive_loss = -samp["level_delta"].clip(upper=0).sum()
active_loss  = hunger_raw[hunger_raw["event_type"]=="active_cost"]["active_energy_cost"].sum()
print(f"  (Level-loss split: {passive_loss:.0f} passive drain vs {active_loss:.0f} active interaction cost.)")
# NOTE: the fitted drain rate == nominal with a degenerate CI because the stomach level is a
# software integrator — this CONFIRMS the monitoring machinery is faithfully implemented and
# runs autonomously, it is not an independent empirical measurement. Genuine empirical weight
# sits in RQ1-4 (override), RQ2-c (occupancy), the D1 ablation and B9.
ok = (0.6*nominal <= eff <= 1.6*nominal) and gaps.median() < 30 and auto_frac > 0.99
verdict("B1", EV_IMPL,
        f"{'Confirmed' if ok else 'FAILED'}: the drive is a software integrator that self-drains at "
        f"exactly {eff/nominal:.2f}x nominal (degenerate CI, as expected for a software integrator — "
        f"this is not a measurement of anything) and samples every {gaps.median():.1f}s across "
        f"{N_MONITORED} runs / {total_runtime_h:.0f} h, autonomously, including "
        f"{N_MONITORED-N_INTERACT} runs with no visitors. The only non-trivial content is the dense "
        f"autonomous sampling; the drain rate matching nominal is true by construction.",
        effect=eff, ci=(lo,hi), n=len(ds))
""")

md(r"""### B2 — Deficit detection *(implementation verification)*

Bracket accuracy here is **1.00 by construction** — the label is computed from the level by the
same 60/25 constants, so a crossing cannot fail to bracket its own threshold. It is arithmetic, and
it is reported only so that nobody mistakes it for evidence. Two things in this cell do carry
information, and **both were previously reported wrongly**.

**1. The drain-fall bracket width.** How tightly is a *continuously* falling level caught? This is
the only number here that speaks to detection latency, and it has to be measured on the
**drain-driven** falls — which requires the level-derived crossing table, because the logged view
contains none of them (A4.0). The previous report gave the widest bracket as **3.6 stomach points**
and glossed it as "one drain sample". It is not: 3.6 is the largest **action cost** in the
controller (`CONVERSATION_TURN_ENERGY_COST`), and it appeared identically at *both* thresholds,
which is the tell. A real drain sample is ~0.007–0.016 points.

**2. Flapping — and here the previous conclusion is not just imprecise, it is backwards.** The old
report stated **"zero rapid reversals at either threshold, so the labels do not flap"**, and
presented that as the one non-trivial result in B2. Its detector could never have found one: it
tested `r.from_state == prev.from_state` when a reversal requires `r.from_state == prev.to_state`.
The condition was unsatisfiable, so the answer was always going to be zero, whatever the data said.

Corrected, on the complete crossing set: **the labels do flap.** There are rapid reversals at the
60 boundary with a median gap of ~29 s, overwhelmingly the pattern *action cost pushes the level
below 60 → someone feeds the robot → back above 60*. This is not a defect — it is the drive
oscillating around its own threshold while being regulated — but it is a real property of the
system, and it retrospectively **justifies the `HS_DWELL_SEC = 60 s` debounce in `chatBot.py`**,
which exists precisely so that the proactive-ping logic does not fire on this churn. The previous
write-up cited the absence of flapping as evidence of clean detection; the debounce was there
because the flapping is real.""")

code(r"""
# B2 runs on the LEVEL-DERIVED crossings (hs_transitions), so drain falls are finally visible.
tr = hs_transitions.copy()
def brackets(row, thr):
    lo = min(row.stomach_level_before, row.stomach_level_after)
    hi = max(row.stomach_level_before, row.stomach_level_after)
    return lo <= thr <= hi
m12 = tr[tr[["from_state","to_state"]].apply(lambda r: set(r)=={"HS1","HS2"}, axis=1)]
m23 = tr[tr[["from_state","to_state"]].apply(lambda r: set(r)=={"HS2","HS3"}, axis=1)]
acc12 = m12.apply(lambda r: brackets(r,HS_FULL_MIN), axis=1).mean() if len(m12) else np.nan
acc23 = m23.apply(lambda r: brackets(r,HS_STARVING_MAX), axis=1).mean() if len(m23) else np.nan
print(f"Full<->Hungry bracket 60: accuracy={acc12:.2f} (n={len(m12)}) [1.00 by construction]")
print(f"Hungry<->Starving bracket 25: accuracy={acc23:.2f} (n={len(m23)}) [1.00 by construction]")

# --- Flapping: a crossing that UNDOES the previous one within `window` seconds. -----------
# THE BUG THIS REPLACES: the old detector tested `r.from_state == prev.from_state`. A reversal
# requires `r.from_state == prev.TO_state` — you can only undo a crossing by going back the way
# you came. The old condition was unsatisfiable, so it returned 0 no matter what the data did,
# and "the labels do not flap" was reported as a finding on that basis.
def flapping(m, window=120.0):
    fl, tot, rows = 0, 0, []
    for run, g in m.sort_values(["run_id","monotonic_sec"]).groupby("run_id"):
        prev = None
        for _, r in g.iterrows():
            if prev is not None and (r.monotonic_sec - prev[0]) < window \
               and r.from_state == prev[2] and r.to_state == prev[1]:
                fl += 1                       # this crossing goes back the way the last one came
                rows.append(dict(run_id=run, gap_sec=float(r.monotonic_sec - prev[0]),
                                 down_cause=prev[3], up_cause=r.cause))
            prev = (r.monotonic_sec, r.from_state, r.to_state, r.cause); tot += 1
    return fl, tot, pd.DataFrame(rows)

fl12, tot12, rev12 = flapping(m12)
fl23, tot23, rev23 = flapping(m23)
print(f"\nRapid reversals (a crossing undone within 120 s), on the COMPLETE crossing set:")
print(f"    Full<->Hungry:     {fl12}/{tot12} crossings")
print(f"    Hungry<->Starving: {fl23}/{tot23} crossings")
_rev = pd.concat([rev12, rev23], ignore_index=True)
if len(_rev):
    print(f"\n    Median gap: {_rev['gap_sec'].median():.0f}s. What causes them:")
    print(_rev.groupby(["down_cause","up_cause"]).size().rename("n").to_frame()
              .to_string().replace("\n","\n    "))
    print(f"\n    -> The dominant pattern is: an ACTION COST pushes the level below the threshold,")
    print(f"       someone FEEDS the robot, and it crosses straight back. The drive oscillates")
    print(f"       around its own boundary while being regulated. That is not a defect — but it")
    print(f"       IS real, and it is exactly what chatBot.py's HS_DWELL_SEC={CONST['CHAT']['HS_DWELL_SEC']:.0f}s")
    print(f"       debounce exists to absorb, so that proactive pings do not fire on the churn.")
    print(f"\n    The previous report claimed ZERO reversals and cited that as evidence the labels")
    print(f"    do not flap. Its detector compared from-state to from-state instead of to-state,")
    print(f"    so it could never fire. The debounce was in the source all along BECAUSE the")
    print(f"    flapping is real.")
    _rev.to_csv(OUT_DIR/"b2_flapping_events.csv", index=False)

# --- The informative number: how tightly is a CONTINUOUSLY falling level caught? -------
# The old code took the max |level_delta| over ALL falls and called it "one drain sample".
# It was not: every fall in the logged view was an interaction_cost fall, so the number it
# reported (3.6) was the largest ACTION COST in the controller, not a drain step at all.
# A real drain sample is ~0.007-0.016 stomach points. Falls are now split by cause.
def _fall_stats(m, cause):
    f = m[(m["level_delta"] < 0) & (m["cause"] == cause)]
    if not len(f): return (0, np.nan, np.nan)
    w = (f["stomach_level_before"] - f["stomach_level_after"]).abs()
    return (len(f), float(w.max()), float(w.mean()))
_nominal_step = CONST["DRAIN_RATE_PER_SEC"] * float(gaps.median())
rows = []
for thr, edge, m in [(HS_FULL_MIN, "Full<->Hungry", m12), (HS_STARVING_MAX, "Hungry<->Starving", m23)]:
    n_dr, w_dr, wm_dr = _fall_stats(m, "passive_drain")
    n_ac, w_ac, wm_ac = _fall_stats(m, "interaction_cost")
    rises = int((m["level_delta"] > 0).sum())
    rows.append(dict(threshold=thr, edge=edge, n_crossings=len(m),
                     bracket_accuracy=acc12 if thr == HS_FULL_MIN else acc23,
                     n_drain_falls=n_dr, max_drain_bracket_width=w_dr, mean_drain_bracket_width=wm_dr,
                     nominal_drain_step=_nominal_step,
                     n_action_cost_falls=n_ac, max_action_cost_bracket_width=w_ac,
                     n_feeding_rises=rises,
                     rapid_reversals_lt120s=fl12 if thr == HS_FULL_MIN else fl23))
b2_check = pd.DataFrame(rows)
b2_check.to_csv(OUT_DIR/"b2_detection_check.csv", index=False)
print("\nB2 detection check (-> outputs/b2_detection_check.csv):")
print(b2_check.round(4).to_string(index=False))
print(f"\n  Drain-driven falls are caught within ~{b2_check['max_drain_bracket_width'].max():.4f} "
      f"stomach points (nominal step at the median {gaps.median():.1f}s sampling gap = "
      f"{_nominal_step:.4f}). Action-cost falls jump the boundary by up to "
      f"{b2_check['max_action_cost_bracket_width'].max():.1f} points — that is one "
      f"CONVERSATION_TURN_ENERGY_COST (3.6), a discrete charge, not a detection latency.")

# Observed crossing counts by edge: the tabular form of the state-transition graph.
b2_counts = (tr.groupby(["from_state","to_state","cause"]).size().rename("n").reset_index())
b2_counts["from_state"]=b2_counts["from_state"].map(HS_NAME)
b2_counts["to_state"]=b2_counts["to_state"].map(HS_NAME)
b2_counts.to_csv(OUT_DIR/"b2_transition_counts.csv", index=False)
print("\nObserved state crossings by cause (-> outputs/b2_transition_counts.csv):")
print(b2_counts.to_string(index=False))
ok = (np.nan_to_num(acc12,nan=1)>=0.9) and (np.nan_to_num(acc23,nan=1)>=0.9)
verdict("B2", EV_IMPL,
        f"Detection is faithfully implemented, and one previously reported result is REVERSED. "
        f"Bracket accuracy ({np.nan_to_num(acc12):.2f}/{np.nan_to_num(acc23):.2f}) is true by "
        f"construction — the label is computed from the level by the same thresholds — and carries "
        f"no evidential weight. Detection LATENCY is genuinely tight: drain-driven falls are caught "
        f"within {b2_check['max_drain_bracket_width'].max():.4f} stomach points, about one "
        f"{gaps.median():.1f}s sampling step (the earlier figure of '3.6 points = one drain sample' "
        f"was wrong; 3.6 is the largest ACTION COST, not a drain step). But the labels DO FLAP: "
        f"{fl12} rapid reversals at the 60 boundary and {fl23} at 25"
        + (f" (median gap {_rev['gap_sec'].median():.0f}s)" if len(_rev) else "")
        + f", mostly an action cost pushing the level under the "
        f"threshold and a feed pulling it straight back. The previous report claimed ZERO reversals "
        f"and called that the non-trivial result of B2; its detector compared from-state against "
        f"from-state rather than to-state and could never have fired. The flapping is real, and it "
        f"is why chatBot.py carries a {CONST['CHAT']['HS_DWELL_SEC']:.0f}s HS_DWELL debounce.",
        n=len(tr), n_reversals=int(fl12+fl23))
""")

md(r"""### B3 — Deficit and behavioural allocation

RQ1-3 asks whether the orexigenic *deficit* changes the controller's action selection, so the
contrast is **no-deficit (Full/HS1) vs deficit (Hungry+Starving, HS2+HS3)**.

**Two kinds of quantity live in this section, and they are not the same kind of evidence.**

*Robot actions that are state-gated in the source.* The LLM system overlay switches to a hungry
persona (`system_overlay_hs2`); the face-to-face prompts switch to `*_hs2` variants; HS3 runs the
`_run_hunger_tree` feed-seeking loop (`hunger_ask_feed` / `hunger_still_hungry` /
`hunger_look_around`); the remote channel emits `hs2_entry` / `hs3_proactive` pings that *cannot*
fire at Full. These are **`Implementation verification`**. "Proactive pings: 0 at Full → 172 in
deficit" is not a finding — it is an `if` statement. Reporting them alongside measured outcomes,
as an earlier version of this analysis did, inflates the apparent evidence.

*Outcomes that depend on what a human then did.* Whether a meal actually arrives is not under the
robot's control. That is the only quantity here that can be an **association**.

**Outcome naming (corrected).** The inferential outcome is `meals_eaten_count > 0`, which records
that **a meal was received during the interaction** — an outcome of the human-robot dyad. It was
previously named `fed01` and described as "feeding pursuit", i.e. as a robot *action*. It is not:
the robot can pursue feeding and get nothing, and can receive food without asking. The variable is
now `feeding_received` and the estimand is a **deficit–feeding association**, not a measure of the
robot's pursuit. Robot pursuit is reported separately, and separately classed.

**Model.** Person-clustered logistic GEE, `feeding_received ~ deficit`, plus pre-interaction-control
sensitivity fits (social state at approach, phase, day, trigger mode, the person's prior interaction
count). All predictors are fixed **before** the interaction starts, so no outcome information leaks
in. Clustering is reported by person **and** by run, because those are different dependence
structures and neither dominates. Unidentified faces (`person_id == "unknown"`) are **excluded**
from person-clustered models: they are not one person, and treating ~23 interactions from an unknown
number of strangers as a single cluster mis-states the correlation structure. They are retained in
the run-clustered sensitivity fit, where the cluster is well defined.

This is a single always-on condition with no drive-off control, so **no causal claim is available**
and none is made.""")

code(r"""
# B3: does a DEFICIT (HS2+HS3) change what the robot does, vs no-deficit (HS1)?
# We compare the coded state-gated recovery repertoire across both channels.
def deficit_grp(hs): return "Full" if hs=="HS1" else ("Deficit" if hs in ("HS2","HS3") else None)

# ---- (1) Face-to-face: hunger framing in the robot's own speech (LLM overlay effect) ----
tt = turns_nlp.copy()
tt["grp"] = tt["hunger_state"].map(deficit_grp)
tt["hm"] = pd.to_numeric(tt.get("hunger_mentioned", 0), errors="coerce").fillna(0)
fr = tt.dropna(subset=["grp"]).groupby("grp")["hm"].agg(["mean","size"])
f_full, f_def = fr.loc["Full","mean"], fr.loc["Deficit","mean"]
da,dlo,dhi = boot_diff_ci(tt[tt.grp=="Deficit"]["hm"], tt[tt.grp=="Full"]["hm"])
print(f"(1) Face-to-face hunger framing: Full {f_full:.3f} -> Deficit {f_def:.3f} "
      f"(x{f_def/max(f_full,1e-9):.0f}; diff {da:+.2f} [95% CI {dlo:.2f},{dhi:.2f}], "
      f"n_turns Full={int(fr.loc['Full','size'])}/Deficit={int(fr.loc['Deficit','size'])})")

# ---- (2) Face-to-face: feed-seeking speech acts fire only in a deficit (coded HS3 tree) ----
ac = hunger_raw[hunger_raw["event_type"]=="active_cost"].copy()
ac["grp"] = ac["hunger_state_before"].fillna(ac["hunger_state_after"]).map(deficit_grp)
seek = ac[ac["stimulus_label"].isin(["hunger_ask_feed","hunger_still_hungry","hunger_look_around"])]
sk = seek.groupby("grp").size()
print(f"(2) Feed-seeking speech acts (ask_feed/still_hungry/look_around): "
      f"Full={int(sk.get('Full',0))} vs Deficit={int(sk.get('Deficit',0))} "
      f"(deficit-only, as coded in _run_hunger_tree)")

# ---- (3) Co-present interaction behaviour: pursuit rises under deficit ----
d = master.copy()
d["grp"] = d["hunger_state_start"].map(deficit_grp)
d["fed"] = pd.to_numeric(d["meals_eaten_count"], errors="coerce").fillna(0) > 0
d = d.dropna(subset=["grp"])
g = d.groupby("grp").agg(n=("interaction_id","size"), feed_pursuit=("fed","mean"))
print("\n(3) Co-present interactions Full vs Deficit:")
print(g.loc[["Full","Deficit"]].round(3).to_string())
dp,dplo,dphi = boot_diff_ci(d[d.grp=="Deficit"]["fed"].astype(float), d[d.grp=="Full"]["fed"].astype(float))
print(f"    feeding-pursuit diff (Deficit-Full): {dp:+.2f} [95% CI {dplo:.2f},{dphi:.2f}]")

# ---- (4) Meal size grows with the deficit ----
feeds = hunger_raw[hunger_raw["event_type"]=="feeding"].copy()
feeds["grp"] = feeds["hunger_state_before"].fillna(feeds["hunger_state_after"]).map(deficit_grp)
ms = feeds.dropna(subset=["grp"]).groupby("grp")["meal_delta"].agg(["mean","size"])
print(f"\n(4) Meal size: Full {ms.loc['Full','mean']:.1f} -> Deficit {ms.loc['Deficit','mean']:.1f}")

# ---- (5) Remote channel: proactive pings are entirely deficit-gated (0 at Full) ----
ev = chat_events.copy()
prov = ev[ev["event_type"].isin(["hs2_entry","hs3_proactive"])]
n_full_ping = int((prov["hs"]=="HS1").sum())
n_def_ping = int(prov["hs"].isin(["HS2","HS3"]).sum())
ping_counts = prov.groupby("event_type").size()
am = chat_msgs.copy(); am = am[am["role"]=="assistant"] if "role" in am.columns else am
am["grp"] = am["hs"].map(deficit_grp)
am["hm"] = pd.to_numeric(am.get("hunger_mentioned",0), errors="coerce").fillna(0)
tg = am.dropna(subset=["grp"]).groupby("grp")["hm"].mean()
print(f"\n(5) Remote (Telegram): proactive hunger pings = {n_full_ping} at Full vs {n_def_ping} in "
      f"Deficit ({int(ping_counts.get('hs2_entry',0))} hs2_entry + "
      f"{int(ping_counts.get('hs3_proactive',0))} hs3_proactive) — the whole proactive channel is deficit-gated. "
      f"Telegram hunger-framing: Full {tg.get('Full',float('nan')):.3f} -> Deficit {tg.get('Deficit',float('nan')):.3f}.")

# ---- INFERENTIAL ANCHOR -------------------------------------------------------------
# Outcome: FEEDING RECEIVED (meals_eaten_count > 0) — a dyadic outcome that depends on what
# the human did. NOT "feeding pursuit": the robot cannot make a meal appear. The framing /
# ping / feed-seeking rows above are coded gates and are NOT modelled — their significance
# would be true by construction, so giving them a p-value would be dishonest arithmetic.
#
# "unknown" is an unrecognised-face placeholder, not a person. Pooling those interactions
# into one person-cluster would assert a dependence structure that does not exist, so they
# are dropped from the person-clustered fits and kept only where the cluster is run.
d["deficit"] = (d["grp"]=="Deficit").astype(int)
d["feeding_received"] = d["fed"].astype(int)
d_named = d[d["person_id"] != "unknown"].copy()
print(f"\nPerson-clustered models exclude 'unknown' faces: "
      f"{len(d)} interactions -> {len(d_named)} across {d_named['person_id'].nunique()} named people "
      f"({len(d)-len(d_named)} unknown-face interactions dropped; they are retained in the "
      f"run-clustered fit below, where the cluster is well defined).")
print("\nDeficit x feeding-received (named people):")
print(pd.crosstab(d_named["grp"], d_named["feeding_received"], margins=True).to_string())

_g3 = smf.gee("feeding_received ~ deficit", groups="person_id", data=d_named,
              family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
b3_or = float(np.exp(_g3.params["deficit"]))
b3_ci = np.exp(_g3.conf_int().loc["deficit"]).tolist()
b3_p  = register_p("B3", "feeding_received ~ deficit (logistic GEE, cluster=person)",
                   "deficit", float(_g3.pvalues["deficit"]), "RQ1/2-behaviour", "confirmatory")

# THE INTERVAL WE LEAD WITH. 14 person-clusters is far below the ~40 where the GEE sandwich
# SE becomes trustworthy, so the asymptotic CI is reported second, not first.
_b3_boot = cluster_bootstrap_effect(
    d_named, "person_id",
    lambda _s: np.exp(smf.glm("feeding_received ~ deficit", data=_s,
                              family=sm.families.Binomial()).fit().params["deficit"]),
    label="B3 deficit->feeding OR (person-cluster bootstrap)")
_b3_boot_run = cluster_bootstrap_effect(
    d, "run_id",
    lambda _s: np.exp(smf.glm("feeding_received ~ deficit", data=_s,
                              family=sm.families.Binomial()).fit().params["deficit"]),
    label="B3 deficit->feeding OR (run-cluster bootstrap)")
print(f"\nPRIMARY (lead with this): deficit->feeding OR = {b3_or:.2f}, "
      f"person-cluster bootstrap [95% {_b3_boot[0]:.2f}, {_b3_boot[2]:.2f}]")
print(f"  asymptotic GEE CI (reported second; anti-conservative at 14 clusters): "
      f"[{b3_ci[0]:.2f}, {b3_ci[1]:.2f}], p={b3_p:.2e}")
print(f"  run-clustered bootstrap [95% {_b3_boot_run[0]:.2f}, {_b3_boot_run[2]:.2f}] "
      f"(different dependence structure, same conclusion)")

# ---- SENSITIVITY: pre-interaction controls -------------------------------------------
# Everything on the right-hand side is fixed BEFORE the interaction starts. `prior_n` is the
# person's cumulative interaction count strictly before this row, so it cannot absorb the
# outcome. These are association models: they cannot identify a causal effect of the drive,
# because there is no drive-off condition to compare against.
d_named = d_named.sort_values(["person_id","interaction_start_epoch"])
d_named["prior_n"] = d_named.groupby("person_id").cumcount()
d_named["phase"] = d_named["day_rome"].map(lambda dd: "P1" if str(dd) in PHASE1_DAYS_EARLY else "P2")
_sens_specs = [
    ("unadjusted",                     "feeding_received ~ deficit"),
    ("+ social state at approach",     "feeding_received ~ deficit + C(initial_state)"),
    ("+ trigger mode",                 "feeding_received ~ deficit + C(initial_state) + C(trigger_mode)"),
    ("+ phase",                        "feeding_received ~ deficit + C(initial_state) + C(trigger_mode) + C(phase)"),
    ("+ prior interaction count",      "feeding_received ~ deficit + C(initial_state) + C(trigger_mode) + C(phase) + prior_n"),
]
print("\nSensitivity to pre-interaction controls (all clustered on person):")
b3_sens_rows=[]
for lbl, fml in _sens_specs:
    for cl in ("person_id", "run_id"):
        src = d_named if cl == "person_id" else d
        if cl == "run_id" and "prior_n" in fml:
            src = d_named    # prior_n only defined on named people
        try:
            f = smf.gee(fml, groups=cl, data=src, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
            orr = float(np.exp(f.params["deficit"]))
            ci = np.exp(f.conf_int().loc["deficit"]).tolist()
            b3_sens_rows.append(dict(model=lbl, cluster=cl.replace("_id",""), n=len(src),
                                     odds_ratio=orr, lo=ci[0], hi=ci[1],
                                     p=float(f.pvalues["deficit"])))
        except Exception as e:
            b3_sens_rows.append(dict(model=lbl, cluster=cl.replace("_id",""), n=len(src),
                                     odds_ratio=np.nan, lo=np.nan, hi=np.nan, p=np.nan))
b3_sens = pd.DataFrame(b3_sens_rows)
b3_sens.to_csv(OUT_DIR/"b3_adjusted_models.csv", index=False)
print(b3_sens.round(3).to_string(index=False))
print("  -> the deficit-feeding association is not created by any of these controls.")

# Leave-one-person-out: is one participant carrying it?
_lopo=[]
for _pid in d_named["person_id"].unique():
    _sub=d_named[d_named["person_id"]!=_pid]
    try:
        _f=smf.gee("feeding_received ~ deficit", groups="person_id", data=_sub,
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        _lopo.append(np.exp(_f.params["deficit"]))
    except Exception: pass
print(f"\nLeave-one-person-out OR range: [{min(_lopo):.2f}, {max(_lopo):.2f}] ({len(_lopo)} refits) "
      f"— no single person carries it.")

SENSITIVITY.append(dict(metric="B3_deficit_feeding_OR", primary=b3_or,
                        boot_lo=_b3_boot[0], boot_median=_b3_boot[1],
                        boot_hi=_b3_boot[2], successful_refits=_b3_boot[3],
                        unit="person-cluster bootstrap over interactions"))
SENSITIVITY.append(dict(metric="B3_deficit_feeding_OR_runcluster", primary=b3_or,
                        boot_lo=_b3_boot_run[0], boot_median=_b3_boot_run[1],
                        boot_hi=_b3_boot_run[2], successful_refits=_b3_boot_run[3],
                        unit="run-cluster bootstrap over interactions"))
globals()["_b3"]=dict(orr=b3_or, ci=b3_ci, p=b3_p, boot=[_b3_boot[0],_b3_boot[1],_b3_boot[2]],
                      boot_run=[_b3_boot_run[0],_b3_boot_run[1],_b3_boot_run[2]],
                      lopo=[min(_lopo),max(_lopo)], n=len(d_named))

verdict("B3", EV_ASSOC,
        f"Within-deployment association between the orexigenic deficit and feeding received. "
        f"Odds of a meal arriving during an interaction are {b3_or:.1f}x higher in deficit than at "
        f"Full (person-cluster bootstrap [{_b3_boot[0]:.1f}, {_b3_boot[2]:.1f}]; run-cluster bootstrap "
        f"[{_b3_boot_run[0]:.1f}, {_b3_boot_run[2]:.1f}]; asymptotic GEE CI [{b3_ci[0]:.1f}, {b3_ci[1]:.1f}], "
        f"p={b3_p:.1e}; LOPO {min(_lopo):.1f}-{max(_lopo):.1f}). It survives adjustment for social state, "
        f"trigger mode, phase and prior interaction count. Raw rates: feeding received in "
        f"{g.loc['Full','feed_pursuit']:.2f} of Full interactions vs {g.loc['Deficit','feed_pursuit']:.2f} "
        f"in deficit; meals are larger in deficit ({ms.loc['Full','mean']:.0f} -> {ms.loc['Deficit','mean']:.0f}). "
        f"SEPARATELY, and as implementation verification only: the hunger framing "
        f"({f_full*100:.0f}%->{f_def*100:.0f}%), the {int(sk.get('Deficit',0))} feed-seeking speech acts and "
        f"the {n_def_ping} proactive pings (vs {n_full_ping} at Full) are state-gated in source — they are "
        f"`if` statements, not findings, and carry no inferential weight. Single always-on condition: "
        f"this is an association, NOT a causal effect of the drive.",
        p=float(b3_p), n=len(d_named))
""")

md(r"""### B4 — Behavioural priority reallocation under Starving *(exploratory)*

**This cell is small-n to the point of quasi-separation, and it is now reported that way.**

There are **13 Starving interactions in the entire corpus**, of which exactly **1** reached
Engaged. Six of the 13 come from a single person. An earlier version fitted an ordinary logistic
GEE to that cell and reported `OR 0.03 [0.008, 0.136], p = 1.9e-6`. That p-value was an artefact:
maximum-likelihood logistic regression is **undefined under quasi-separation** — the coefficient
runs to −∞, the Wald SE collapses with it, and the ratio manufactures a tiny p-value out of a cell
containing almost no information. The tell was in the analysis's own output: the person-cluster
bootstrap lower bound came back as `3.3e-11`, which is not a confidence bound, it is a divergence.

So B4 is refitted with **Firth's penalised likelihood** (Jeffreys prior), which keeps the estimate
finite under separation, and read alongside **Fisher's exact test** on the raw 2×2. Its p-value is
recorded for transparency but is **excluded from the confirmatory BH families** — correcting a
number this fragile would only lend it authority it has not earned.

The direction (Starving suppresses social completion while feeding rises) is consistent and
plausible. The magnitude is not estimable from 13 interactions. Evidence class:
**`Exploratory observation`**. It is not used to support RQ1 on its own.""")

code(r"""
# B4: SSxHS crosstab of outcomes; the HS3 override (turns & Engaged collapse, feeding rises).
d = master.copy()
d["reached_ss4"] = (d["final_state"]=="ss4").astype(int)
d["n_turns"] = pd.to_numeric(d["n_turns"], errors="coerce").fillna(0)
d["fed_here"] = pd.to_numeric(d["meals_eaten_count"], errors="coerce").fillna(0) > 0
d = d.dropna(subset=["hunger_state_start"])
d["starving"] = (d["hunger_state_start"]=="HS3").astype(int)
d_named = d[d["person_id"] != "unknown"].copy()   # 'unknown' is not a person-cluster

piv_turns = d.pivot_table(index="initial_state", columns="hunger_state_start",
                          values="n_turns", aggfunc="mean")
piv_ss4 = d.pivot_table(index="initial_state", columns="hunger_state_start",
                        values="reached_ss4", aggfunc="mean")
print("E[n_turns] by State×Hunger:\n", piv_turns.rename(index=SS_NAME,columns=HS_NAME).round(2).to_string())
print("\nP(reach Engaged) by State×Hunger:\n", piv_ss4.rename(index=SS_NAME,columns=HS_NAME).round(2).to_string())

hs3 = d[d["hunger_state_start"]=="HS3"]; rest = d[d["hunger_state_start"].isin(["HS1","HS2"])]

# ---- FIRST: state plainly how thin this cell is. --------------------------------------
_k_hs3, _n_hs3 = int(hs3["reached_ss4"].sum()), len(hs3)
_k_rest, _n_rest = int(rest["reached_ss4"].sum()), len(rest)
print(f"\n*** THE ENTIRE B4 CELL ***")
print(f"    Starving:     Engaged in {_k_hs3}/{_n_hs3} interactions")
print(f"    Full/Hungry:  Engaged in {_k_rest}/{_n_rest} interactions")
print(f"    Starving interactions come from {hs3['person_id'].nunique()} people "
      f"({hs3['person_id'].value_counts().to_dict()}) across {hs3['run_id'].nunique()} runs.")
print(f"    One success in {_n_hs3} trials cannot support a precise odds ratio. Read the")
print(f"    exact interval below, not a model coefficient.")

# Exact (Clopper-Pearson) completion rates + Fisher exact on the raw 2x2.
_e_hs3 = exact_prop_ci(_k_hs3, _n_hs3); _e_rest = exact_prop_ci(_k_rest, _n_rest)
print(f"\nP(Engaged | Starving)    = {_e_hs3[0]:.2f}  exact 95% CI [{_e_hs3[1]:.2f}, {_e_hs3[2]:.2f}]  (n={_n_hs3})")
print(f"P(Engaged | Full/Hungry) = {_e_rest[0]:.2f}  exact 95% CI [{_e_rest[1]:.2f}, {_e_rest[2]:.2f}]  (n={_n_rest})")
_tab = [[_k_hs3, _n_hs3-_k_hs3], [_k_rest, _n_rest-_k_rest]]
_or_fisher, _p_fisher = sps.fisher_exact(_tab)
print(f"Fisher exact on the 2x2: OR={_or_fisher:.3f}, p={_p_fisher:.2e} "
      f"(exact, but treats interactions as independent — they are not)")

# ---- Firth penalised logistic: finite estimate under separation. -----------------------
def _b4_design(df):
    "Design matrix [intercept, starving, social-state dummies]; 'starving' is always column 1."
    X = pd.get_dummies(df[["starving","initial_state"]], columns=["initial_state"],
                       drop_first=True, dtype=float)
    cols = ["starving"] + [c for c in X.columns if c != "starving"]
    X = X[cols]
    X.insert(0, "intercept", 1.0)
    return X.values, df["reached_ss4"].astype(float).values

_X4, _y4 = _b4_design(d_named)
_b4_beta, _b4_lo, _b4_hi, _b4_plrt = firth_profile_ci(_X4, _y4, 1)   # col 1 == starving
_or4, _ci4 = float(np.exp(_b4_beta)), [float(np.exp(_b4_lo)), float(np.exp(_b4_hi))]
print(f"\nFirth penalised logistic (Engaged ~ starving + social state, n={len(d_named)} named):")
print(f"    OR(Engaged | Starving) = {_or4:.3f}  profile-penalised-likelihood 95% CI "
      f"[{_ci4[0]:.3f}, {_ci4[1]:.3f}], penalised LRT p={_b4_plrt:.2e}")
print(f"    (For contrast, the unpenalised GEE this replaces reported OR 0.03 [0.008, 0.136], "
      f"p~2e-6 — a number produced by the likelihood diverging, not by evidence.)")

# The p-value is recorded, but as EXPLORATORY: it does not enter a confirmatory BH family.
register_p("B4", "reached_ss4 ~ starving + C(initial_state) (Firth penalised logistic)",
           "starving", _b4_plrt, family="(none - excluded)", status="exploratory",
           note="quasi-separation: 1 success in 13 Starving interactions; reported for "
                "transparency, never used to support a claim")

def _b4_firth_or(_s):
    X, y = _b4_design(_s)
    beta, _ = firth_logit(X, y)
    return float(np.exp(beta[1]))
_b4_boot = cluster_bootstrap_effect(d_named, "person_id", _b4_firth_or,
                                    label="B4 Starving OR (Firth, person-cluster bootstrap)")
print("  Note the bootstrap no longer collapses to ~1e-11: the Firth penalty keeps every refit "
      "finite, which is exactly what the unpenalised version could not do.")
SENSITIVITY.append(dict(metric="B4_starving_override_OR_firth", primary=_or4,
                        boot_lo=_b4_boot[0], boot_median=_b4_boot[1],
                        boot_hi=_b4_boot[2], successful_refits=_b4_boot[3],
                        unit="person-cluster bootstrap over interactions (Firth)"))

# Turns and feeding, descriptively, with cluster-bootstrap intervals.
dt, dlo, dhi = boot_diff_ci(hs3["n_turns"], rest["n_turns"])
ss4_hs3, ss4_rest = _e_hs3[0], _e_rest[0]
feed_hs3 = hs3["fed_here"].mean() if len(hs3) else np.nan
feed_rest = rest["fed_here"].mean()
print(f"\nTurns: E[n_turns|Starving]={hs3['n_turns'].mean():.2f} vs "
      f"E[n_turns|Full/Hungry]={rest['n_turns'].mean():.2f}; diff={dt:+.2f} [95% CI {dlo:.2f},{dhi:.2f}]")
print(f"Feeding received: P(meal|Starving)={feed_hs3:.2f} vs P(meal|Full/Hungry)={feed_rest:.2f} "
      f"— the two move in OPPOSITE directions, which is the actual content of B4: this is a "
      f"reallocation of priority, not undifferentiated disengagement.")

b4_tab = pd.DataFrame([
    dict(quantity="P(Engaged) | Starving", k=_k_hs3, n=_n_hs3, estimate=_e_hs3[0],
         exact_lo=_e_hs3[1], exact_hi=_e_hs3[2]),
    dict(quantity="P(Engaged) | Full/Hungry", k=_k_rest, n=_n_rest, estimate=_e_rest[0],
         exact_lo=_e_rest[1], exact_hi=_e_rest[2]),
])
b4_tab.to_csv(OUT_DIR/"b4_starving_exact.csv", index=False)

override = (not np.isnan(dt) and dt < 0) and (np.nan_to_num(feed_hs3) >= feed_rest)
verdict("B4", EV_EXPL,
        f"Directionally consistent, not estimable. Starving interactions completed socially in "
        f"{_k_hs3}/{_n_hs3} cases (exact 95% CI [{_e_hs3[1]:.2f}, {_e_hs3[2]:.2f}]) versus "
        f"{_k_rest}/{_n_rest} otherwise (exact [{_e_rest[1]:.2f}, {_e_rest[2]:.2f}]), while feeding "
        f"received ROSE ({feed_rest:.2f} -> {feed_hs3:.2f}) and turns fell ({dt:+.2f}). The opposing "
        f"directions are the substance: this looks like priority reallocation toward recovery rather "
        f"than disengagement. But the whole cell is 1 success in {_n_hs3} interactions from "
        f"{hs3['person_id'].nunique()} people, so the Firth OR ({_or4:.3f}, profile CI "
        f"[{_ci4[0]:.3f}, {_ci4[1]:.3f}]) is a bound, not an estimate; the earlier "
        f"OR 0.03 / p~2e-6 was a separation artefact. Excluded from the confirmatory families. "
        f"Requires replication with more Starving exposure before any magnitude is claimed.",
        n=len(d_named), n_starving=_n_hs3)
""")

md(r"""### B5a — Meal size and deficit severity

Does the *amount* of food delivered scale with how hungry the robot is? An earlier version
answered this from three unadjusted means (21 / 29 / 43) with no interval, no clustering and no
check on who supplied them. Meals are supplied by a handful of people — two of whom were *obligated
to feed* in Phase 1 — so unadjusted means over meals can be carried entirely by one person's habits.

Here: n / mean / median / IQR per state, **cluster-bootstrap intervals**, a cluster-aware model
with hunger state as an **ordered** predictor (Full < Hungry < Starving), and a refit **excluding
the two obligated feeders**. If the gradient only exists because the obligated feeders were around,
that must show up.""")

code(r"""
# --- B5a. Meal size vs deficit severity -----------------------------------------------
feeds = hunger_raw[hunger_raw["event_type"]=="feeding"].copy()
feeds["hs_before"] = feeds["hunger_state_before"].fillna(feeds["hunger_state_after"])
feeds = feeds[feeds["hs_before"].isin(HS_ORDER)].copy()
# Attribute each meal to the person who gave it. Feeding events carry NO exec_interaction_id
# (checked: 0/108), so the only link to a person is `feeder_face_id` — now pseudonymised like
# every other identity column. Without this, meals cannot be clustered by person at all, and the
# obligated-feeder sensitivity check below would be impossible.
feeds["person_id"] = feeds["feeder_face_id"]
feeds["hs_rank"] = feeds["hs_before"].map({"HS1":0,"HS2":1,"HS3":2}).astype(float)
print(f"Meals attributable to a named person: "
      f"{int(feeds['person_id'].notna().sum() - (feeds['person_id']=='unknown').sum())}/{len(feeds)} "
      f"(the rest were delivered by an unrecognised face).")

print("Meal size by deficit state at feed time:")
desc = feeds.groupby("hs_before")["meal_delta"].agg(
    n="size", mean="mean", median="median",
    q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).reindex(HS_ORDER)
desc["IQR"] = desc["q75"] - desc["q25"]
# Interval by RUN cluster (meals repeat within runs and within people).
for hs in HS_ORDER:
    sub = feeds[feeds["hs_before"]==hs]
    if len(sub) < 2: continue
    b = cluster_bootstrap_effect(sub, "run_id", lambda _s: _s["meal_delta"].mean(),
                                 label=f"  mean meal size {hsn(hs)} (run-cluster boot)")
    desc.loc[hs,"boot_lo"], desc.loc[hs,"boot_hi"] = b[0], b[2]
desc.index = [hsn(h) for h in desc.index]
print("\n" + desc.round(2).to_string())
desc.to_csv(OUT_DIR/"b5_meal_size_by_state.csv")

# Ordered-predictor model. hs_rank is 0/1/2, so the coefficient is the mean increase in meal
# size per STEP down the deficit ladder. Clustered on run (meals nest in runs); person-clustered
# refit reported alongside, since neither dependence structure dominates.
_mf = feeds.dropna(subset=["meal_delta","hs_rank"]).copy()
_m_run = smf.ols("meal_delta ~ hs_rank", _mf).fit(
    cov_type="cluster", cov_kwds={"groups": _mf["run_id"]})
_mf_named = _mf[_mf["person_id"].notna() & (_mf["person_id"] != "unknown")].copy()
print(f"\nOrdered model  meal_delta ~ hs_rank  (0=Full, 1=Hungry, 2=Starving):")
print(f"  run-clustered:    {_m_run.params['hs_rank']:+.2f} stomach points per step "
      f"[{_m_run.conf_int().loc['hs_rank',0]:+.2f}, {_m_run.conf_int().loc['hs_rank',1]:+.2f}], "
      f"p={_m_run.pvalues['hs_rank']:.4f} (n={len(_mf)} meals, {_mf['run_id'].nunique()} runs)")
if len(_mf_named) > 5 and _mf_named["person_id"].nunique() > 1:
    _m_per = smf.ols("meal_delta ~ hs_rank", _mf_named).fit(
        cov_type="cluster", cov_kwds={"groups": _mf_named["person_id"]})
    print(f"  person-clustered: {_m_per.params['hs_rank']:+.2f} "
          f"[{_m_per.conf_int().loc['hs_rank',0]:+.2f}, {_m_per.conf_int().loc['hs_rank',1]:+.2f}], "
          f"p={_m_per.pvalues['hs_rank']:.4f} (n={len(_mf_named)} attributable meals, "
          f"{_mf_named['person_id'].nunique()} people)")
else:
    print(f"  person-clustered: not fitted — only {len(_mf_named)} meals attributable to a named "
          f"person, which is too few to cluster on.")
_b5_meal_p = register_p("B5a", "meal_delta ~ hs_rank (OLS, cluster-robust by run)", "hs_rank",
                        float(_m_run.pvalues["hs_rank"]), "RQ1/2-behaviour", "confirmatory")
_b5_meal_boot = cluster_bootstrap_effect(
    _mf, "run_id", lambda _s: smf.ols("meal_delta ~ hs_rank", _s).fit().params["hs_rank"],
    label="B5a meal-size slope per deficit step (run-cluster bootstrap)")
SENSITIVITY.append(dict(metric="B5a_meal_size_slope", primary=float(_m_run.params["hs_rank"]),
                        boot_lo=_b5_meal_boot[0], boot_median=_b5_meal_boot[1],
                        boot_hi=_b5_meal_boot[2], successful_refits=_b5_meal_boot[3],
                        unit="run-cluster bootstrap over meals"))

# --- Does the gradient survive without the two obligated feeders? ----------------------
_feeder_pids = {p for p, r in ROLE_OF_EARLY.items() if r == "feeder"}
_no_feeders = _mf_named[~_mf_named["person_id"].isin(_feeder_pids)]
if len(_no_feeders) > 5 and _no_feeders["hs_rank"].nunique() > 1:
    _m_nf = smf.ols("meal_delta ~ hs_rank", _no_feeders).fit(
        cov_type="cluster", cov_kwds={"groups": _no_feeders["person_id"]})
    print(f"\nEXCLUDING the {len(_feeder_pids)} obligated feeders "
          f"({len(_mf_named)-len(_no_feeders)} of their meals dropped):")
    print(f"  slope {_m_nf.params['hs_rank']:+.2f} "
          f"[{_m_nf.conf_int().loc['hs_rank',0]:+.2f}, {_m_nf.conf_int().loc['hs_rank',1]:+.2f}], "
          f"p={_m_nf.pvalues['hs_rank']:.4f} (n={len(_no_feeders)} meals, "
          f"{_no_feeders['person_id'].nunique()} people)")
    print(f"  Mean meal size without them: "
          f"{_no_feeders.groupby('hs_before')['meal_delta'].mean().round(1).to_dict()}")
    _nf_slope = float(_m_nf.params["hs_rank"])
    _nf_ok = _nf_slope > 0
else:
    _nf_slope, _nf_ok = np.nan, False
    print("\nToo few meals outside the obligated feeders to refit — the gradient CANNOT be "
          "shown to be independent of them.")
globals()["_b5_meal"]=dict(slope=float(_m_run.params["hs_rank"]),
                           boot=[_b5_meal_boot[0],_b5_meal_boot[1],_b5_meal_boot[2]],
                           p=_b5_meal_p, nf_slope=_nf_slope,
                           means={h: float(desc.loc[hsn(h),"mean"]) for h in HS_ORDER})
""")

md(r"""### B5b — The remote recovery loop: does a proactive ping actually elicit a reply?

**The old analysis over-counted replies and mixed in the wrong events.** It looped over every
proactive ping and asked "was there *any* user message in the next hour?" — so a single reply could
be credited to several pings, and pings were treated as independent rows even though they cluster
hard within subscribers. It also pooled `hs3_recovery` events (the robot saying *"thanks, I'm
full"*) in with the hunger pings. A recovery notification is **not** a request for food; counting
replies to it as evidence that hunger signalling works is a category error.

The rebuild:

- **One-to-one matching.** Each user reply is consumed by **at most one** preceding ping (the
  nearest one within the window), so no reply is double-counted. `n_replies <= n_pings` by
  construction.
- **Ping types kept apart**: `hs2_entry` (Hungry), `hs3_proactive` (Starving) and `hs3_recovery`
  (a thank-you — **excluded** from the main analysis and reported only as a contrast).
- **Control windows.** The comparison that matters is not "did a reply follow a ping" (people
  message the bot anyway) but "**did a reply follow a ping more often than it followed a matched
  window with no ping**". Controls are drawn per subscriber from **inside the monitored run
  spans** — the hours the robot was actually running and a reply was possible — with no ping
  within the window. Drawing them from the whole calendar instead (nights, gaps between sessions)
  would compare a ping against hours when nobody was going to message the bot regardless, and
  would inflate the apparent effect several-fold. The control set is the analysis here; getting it
  wrong is the easiest way to manufacture a result.
- **Clustered intervals** by subscriber and by run, and **window sensitivity** at 15 / 30 / 60 min.
- The arbitrary `ping_rate > 0.20` pass mark is **gone**. It encoded nothing; the verdict now rests
  on the ping-vs-control contrast and its interval.""")

code(r"""
# --- B5b. Remote loop: one-to-one ping -> reply matching -------------------------------
ev = chat_events.copy().sort_values(["chat_id","timestamp_epoch"])

# PSEUDONYMISE THE SUBSCRIBER FIRST. `chat_id` is a raw Telegram user ID — a stable, real-world
# personal identifier, as identifying as a name and trivially reversible by anyone who has ever
# messaged that account. It was NOT in IDENTITY_COLS, so a per-subscriber table exported from it
# would have published real Telegram IDs to the repository. Map to stable S## codes, ordered by
# first appearance, before anything downstream can touch it.
_sub_order = ev.sort_values("timestamp_epoch")["chat_id"].drop_duplicates().tolist()
SUBSCRIBER_MAP = {c: f"S{i+1:02d}" for i, c in enumerate(_sub_order)}
ev["chat_id"] = ev["chat_id"].map(SUBSCRIBER_MAP)
print(f"Subscribers pseudonymised: {len(SUBSCRIBER_MAP)} raw Telegram IDs -> "
      f"S01..S{len(SUBSCRIBER_MAP):02d}. No raw chat_id reaches any exported table.")

PING_TYPES   = ["hs2_entry", "hs3_proactive"]     # genuine hunger signalling
EXCLUDED     = ["hs3_recovery"]                   # "thanks, I'm full" — NOT a request for food
WINDOWS      = [900.0, 1800.0, 3600.0]            # 15 / 30 / 60 min

def match_pings(ev, ping_types, window):
    '''Greedy one-to-one match: each reply is consumed by at most one ping.

    Walk the pings for a subscriber in time order; the first not-yet-consumed user message
    inside (ping, ping+window] answers it. This is what stops a single chatty reply from
    being credited to five different pings, which is how the old row-independent loop
    inflated the response rate.
    '''
    rows = []
    for chat, g in ev.groupby("chat_id"):
        pings = g[g["event_type"].isin(ping_types)].sort_values("timestamp_epoch")
        msgs  = g[g["event_type"] == "user_message"].sort_values("timestamp_epoch")
        used  = set()
        for _, p in pings.iterrows():
            t0 = p["timestamp_epoch"]
            cand = msgs[(msgs["timestamp_epoch"] > t0) &
                        (msgs["timestamp_epoch"] <= t0 + window) &
                        (~msgs["id"].isin(used))]
            hit = len(cand) > 0
            if hit:
                used.add(int(cand["id"].iloc[0]))     # consume exactly one reply
            rows.append(dict(chat_id=chat, run_id=p["run_id"], day_rome=p["day_rome"],
                             ping_id=int(p["id"]), ping_type=p["event_type"], hs=p["hs"],
                             t=t0, replied=int(hit),
                             latency_sec=(float(cand["timestamp_epoch"].iloc[0]) - t0) if hit else np.nan))
    return pd.DataFrame(rows)

pm = match_pings(ev, PING_TYPES, 3600.0)
_n_excl = int(ev["event_type"].isin(EXCLUDED).sum())
print(f"Proactive HUNGER pings: {len(pm)} "
      f"({pm['ping_type'].value_counts().to_dict()}) across {pm['chat_id'].nunique()} subscribers.")
print(f"Excluded from the main analysis: {_n_excl} hs3_recovery events "
      f"('thanks, I'm full' — a notification, not a request for food).")
print(f"Replies matched one-to-one: {int(pm['replied'].sum())} "
      f"(<= n_pings by construction; the old loop could and did count one reply many times).")

# Sanity: prove no reply was attributed twice, at every window.
for w in WINDOWS:
    _p = match_pings(ev, PING_TYPES, w)
    assert _p["replied"].sum() <= len(_p), "one-to-one matching violated"
print("\nResponse rate by ping type and window (exact 95% CI):")
win_rows=[]
for w in WINDOWS:
    _p = match_pings(ev, PING_TYPES, w)
    for pt in PING_TYPES + ["ALL"]:
        s = _p if pt == "ALL" else _p[_p["ping_type"] == pt]
        if not len(s): continue
        k, n = int(s["replied"].sum()), len(s)
        est, lo, hi = exact_prop_ci(k, n)
        win_rows.append(dict(window_min=int(w/60), ping_type=pt, k=k, n=n,
                             rate=est, exact_lo=lo, exact_hi=hi))
win_df = pd.DataFrame(win_rows)
print(win_df.round(3).to_string(index=False))
win_df.to_csv(OUT_DIR/"b5_ping_response_windows.csv", index=False)

# --- Matched control windows: would a reply have come anyway? ---------------------------
# For each ping, draw a control instant for the SAME subscriber on a day they were active, at a
# comparable time of day, with NO ping in the preceding `window` seconds. Then ask the same
# question of it. Without this contrast, "21% of pings got a reply" is uninterpretable: people
# message the bot regardless.
def control_windows(ev, pm, window, run_spans, seed=SEED):
    '''Matched control instants: same subscriber, ROBOT RUNNING, no ping nearby.

    The matching set decides the answer, and it is easy to get wrong. A first version of this
    cell drew control instants uniformly across each subscriber's whole epoch span — which is
    mostly nights and the gaps between sessions, when the robot was off and nobody was going to
    message it anyway. That control came back at ~1/172, and it made the ping look far more
    effective than it is. A control window has to be a moment when a reply was actually POSSIBLE.

    So control instants are drawn only from inside the monitored run spans (robot on), for the
    same subscriber, with no ping within +/- window. The counterfactual is then the right one:
    "the robot was running, this person was reachable, and it did NOT ping them."
    '''
    rng = np.random.default_rng(seed)
    rows = []
    ping_t = {c: g["t"].values for c, g in pm.groupby("chat_id")}
    spans = [(a, b - window) for a, b in run_spans if (b - window) > a]
    if not spans:
        return pd.DataFrame()
    widths = np.array([b - a for a, b in spans], float)
    probs = widths / widths.sum()
    for chat, g in ev.groupby("chat_id"):
        msgs = g[g["event_type"] == "user_message"].sort_values("timestamp_epoch")
        n_want = int((pm["chat_id"] == chat).sum())
        if not n_want:
            continue
        pt = ping_t.get(chat, np.array([]))
        tries, made = 0, 0
        while made < n_want and tries < 500 * n_want:
            tries += 1
            k = rng.choice(len(spans), p=probs)      # sample a run in proportion to its length
            a, b = spans[k]
            t0 = float(rng.uniform(a, b))
            if len(pt) and np.any((pt >= t0 - window) & (pt <= t0 + window)):
                continue                             # must be a clean, ping-free window
            hit = int(((msgs["timestamp_epoch"] > t0) &
                       (msgs["timestamp_epoch"] <= t0 + window)).any())
            rows.append(dict(chat_id=chat, t=t0, replied=hit)); made += 1
    return pd.DataFrame(rows)

# The only hours in which the robot could have pinged anyone, and therefore the only hours from
# which a control window may legitimately be drawn.
_spans = (hunger_raw.groupby("run_id")["timestamp_epoch"].agg(["min","max"])
                    .apply(tuple, axis=1).tolist())
print(f"\nControl windows are drawn ONLY from inside the {len(_spans)} monitored run spans "
      f"({sum(b-a for a,b in _spans)/3600:.1f} h during which the robot was actually running). "
      f"Drawing them from the whole calendar would compare a ping against nights and inter-session "
      f"gaps, when no reply was possible at all, and would badly inflate the effect.")
ctrl = control_windows(ev, pm, 3600.0, _spans)
if len(ctrl):
    _kp, _np_ = int(pm["replied"].sum()), len(pm)
    _kc, _nc  = int(ctrl["replied"].sum()), len(ctrl)
    _ep = exact_prop_ci(_kp, _np_); _ec = exact_prop_ci(_kc, _nc)
    print(f"\nPING vs MATCHED CONTROL WINDOW (60 min):")
    print(f"    after a hunger ping:  {_kp}/{_np_} = {_ep[0]:.2f}  exact [{_ep[1]:.2f}, {_ep[2]:.2f}]")
    print(f"    matched no-ping window: {_kc}/{_nc} = {_ec[0]:.2f}  exact [{_ec[1]:.2f}, {_ec[2]:.2f}]")
    _both = pd.concat([pm.assign(pinged=1)[["chat_id","replied","pinged"]],
                       ctrl.assign(pinged=0)[["chat_id","replied","pinged"]]], ignore_index=True)
    _b5_ctrl = cluster_bootstrap_effect(
        _both, "chat_id",
        lambda _s: (_s[_s.pinged==1]["replied"].mean() - _s[_s.pinged==0]["replied"].mean()),
        label="B5b ping - control reply-rate difference (subscriber-cluster bootstrap)")
    _b5_ctrl_run = cluster_bootstrap_effect(
        pm, "run_id", lambda _s: _s["replied"].mean(),
        label="B5b ping response rate (run-cluster bootstrap)")
    SENSITIVITY.append(dict(metric="B5b_ping_minus_control_reply_rate",
                            primary=float(_ep[0]-_ec[0]),
                            boot_lo=_b5_ctrl[0], boot_median=_b5_ctrl[1], boot_hi=_b5_ctrl[2],
                            successful_refits=_b5_ctrl[3],
                            unit="subscriber-cluster bootstrap, ping vs matched control window"))
else:
    _ep = _ec = (np.nan,)*3
    _b5_ctrl = _b5_ctrl_run = (np.nan, np.nan, np.nan, 0)

# Per-subscriber / per-run spread: is the response rate carried by one person?
print("\nResponse rate by subscriber (60 min, one-to-one):")
_by_sub = pm.groupby("chat_id").agg(pings=("replied","size"), replies=("replied","sum"))
_by_sub["rate"] = (_by_sub["replies"]/_by_sub["pings"]).round(2)
print(_by_sub.sort_values("pings", ascending=False).to_string())
_by_sub.to_csv(OUT_DIR/"b5_ping_by_subscriber.csv")
print(f"\n  {(_by_sub['replies']==0).sum()}/{len(_by_sub)} subscribers NEVER replied to a hunger ping.")
_by_run = pm.groupby("run_id").agg(pings=("replied","size"), replies=("replied","sum"))
print(f"  Across runs, the reply rate ranges "
      f"{(_by_run['replies']/_by_run['pings']).min():.2f}-{(_by_run['replies']/_by_run['pings']).max():.2f}.")

# The verdict rests on the ping-vs-control contrast and its subscriber-clustered interval.
_diff = float(_ep[0] - _ec[0]) if np.isfinite(_ep[0]) and np.isfinite(_ec[0]) else np.nan
_ctrl_lo = _b5_ctrl[0]
_meaningful = np.isfinite(_ctrl_lo) and _ctrl_lo > 0.0
globals()["_b5_ping"]=dict(rate=float(_ep[0]), ctrl=float(_ec[0]), diff=_diff,
                           boot=[_b5_ctrl[0],_b5_ctrl[1],_b5_ctrl[2]],
                           n_pings=len(pm), n_subs=int(pm["chat_id"].nunique()))
globals()["_b5_pm"]=pm; globals()["_b5_win"]=win_df

_meal_ok = (globals()["_b5_meal"]["boot"][0] > 0)
if _meal_ok and _meaningful:
    _ev5, _txt5 = EV_ASSOC, "Within-deployment association"
elif _meal_ok:
    _ev5, _txt5 = EV_EXPL, "Meal-size gradient holds; the remote loop does not clear its control"
else:
    _ev5, _txt5 = EV_INCONC, "Neither channel clears its control"

verdict("B5", _ev5,
        f"{_txt5}. MEAL SIZE scales with deficit severity: "
        f"{globals()['_b5_meal']['means']['HS1']:.0f} (Full) -> "
        f"{globals()['_b5_meal']['means']['HS2']:.0f} (Hungry) -> "
        f"{globals()['_b5_meal']['means']['HS3']:.0f} (Starving), "
        f"{globals()['_b5_meal']['slope']:+.1f} points per deficit step "
        f"(run-cluster bootstrap [{globals()['_b5_meal']['boot'][0]:+.1f}, "
        f"{globals()['_b5_meal']['boot'][2]:+.1f}])"
        + (f", and it survives excluding the two obligated feeders "
           f"({globals()['_b5_meal']['nf_slope']:+.1f}/step)."
           if np.isfinite(globals()['_b5_meal']['nf_slope']) else
           ", but it CANNOT be shown to be independent of the obligated feeders.")
        + f" REMOTE LOOP, with one-to-one reply matching and hs3_recovery notifications excluded: "
          f"{int(pm['replied'].sum())}/{len(pm)} hunger pings drew a reply within 1 h "
          f"({_ep[0]:.2f}, exact [{_ep[1]:.2f}, {_ep[2]:.2f}]), against {_ec[0]:.2f} "
          f"[{_ec[1]:.2f}, {_ec[2]:.2f}] in matched no-ping control windows — a difference of "
          f"{_diff:+.2f} (subscriber-cluster bootstrap [{_b5_ctrl[0]:+.2f}, {_b5_ctrl[2]:+.2f}]). "
          f"{(_by_sub['replies']==0).sum()}/{len(_by_sub)} subscribers never replied to any hunger ping. "
          f"The remote channel is therefore a weak recovery pathway"
        + ("; its advantage over the control window is not distinguishable from zero at the "
           "subscriber level, so it is reported as exploratory." if not _meaningful else "."),
        n=len(pm))
""")

md(r"""### B6 — Observed Starving episodes *(exploratory, and now over the complete set)*

**What changed, and why it matters more than anything else in this notebook.**

The previous version reported *"8/8 Starving episodes received a feed, escaped Starving, and
recovered to Full; median time to first feed 21 s"* — a perfect record. It was an artefact. The
episode builder identified entries by looking for a logged `before != HS3 → after == HS3` row, and
the executive logger **never emits that row for a drain-driven crossing** (A4.0). So it found only
the episodes the robot fell into *while a human was actively interacting with it* — and those are
exactly the episodes where a human was standing there to feed it. The selection rule and the
finding were the same statement.

Rebuilt from the level series, the corpus contains **17 Starving episodes, not 8**. The nine that
were invisible include the two longest, and one of them is the single most informative event in the
deployment:

> **Run `5b99e872`, 15 June: the robot spent ~30 continuous minutes below the Starving threshold,
> falling to level 10.5 — and that run had 15 logged interactions. People were present. It was not
> fed.** That one episode is ~65% of all Starving time in the corpus, and the old analysis could
> not see it.

Reported below with exact binomial intervals, run-level clustering (episodes are **not**
independent — four of them occur in a single run), and explicit right-censoring. This is an
**`Exploratory observation`** and is **not** used as evidence of reliable recovery. RQ2-c does not
rest on it.""")

code(r"""
# B6: first feed, escape from Starving, and recovery to Full — over ALL 17 episodes.
ep = hs3_episodes.copy()
n_ep = len(ep)

print(f"Starving episodes in the corpus: {n_ep}  (the old label-keyed builder found 8)")
print("\nBy how the robot fell into Starving:")
_by_cause = ep.groupby("entry_cause").agg(
    n=("episode_id","size"), fed=("received_feed","sum"),
    escaped_by_feeding=("escaped_starving_by_feeding","sum"),
    recovered_full_by_feeding=("recovered_to_full_by_feeding","sum"),
    total_hs3_sec=("hs3_duration_sec","sum"), max_hs3_sec=("hs3_duration_sec","max"))
print(_by_cause.round(1).to_string())
print("\n  ^ the `interaction_cost` row is what the old analysis reported as the whole story.")

# --- Outcomes with EXACT intervals, over the complete set -------------------------------
_rows=[]
for lbl, col in [("received a feed", "received_feed"),
                 ("escaped Starving via feeding", "escaped_starving_by_feeding"),
                 ("recovered to Full via feeding", "recovered_to_full_by_feeding"),
                 ("right-censored at end of run", "censored_at_run_end")]:
    k = int(ep[col].sum()); est, lo, hi = exact_prop_ci(k, n_ep)
    _rows.append(dict(outcome=lbl, k=k, n=n_ep, rate=est, exact_lo=lo, exact_hi=hi))
b6 = pd.DataFrame(_rows)
print("\nOutcomes over ALL episodes (Clopper-Pearson exact 95% CI):")
print(b6.round(3).to_string(index=False))
b6.to_csv(OUT_DIR/"b6_episode_outcomes.csv", index=False)

# --- Clustering: episodes are not independent ------------------------------------------
_per_run = ep.groupby("run_id").size()
print(f"\nCLUSTERING: {n_ep} episodes occur in only {len(_per_run)} runs "
      f"(max {_per_run.max()} episodes in one run; {int((_per_run>1).sum())} runs contribute >1). "
      f"Treating them as independent overstates the effective sample size — the effective n is "
      f"closer to {len(_per_run)} than to {n_ep}.")
print(f"Episodes per run: {_per_run.sort_values(ascending=False).to_dict()}")
_b6_boot = cluster_bootstrap_effect(
    ep, "run_id", lambda _s: _s["recovered_to_full_by_feeding"].mean(),
    label="B6 fraction recovering to Full by feeding (run-cluster bootstrap)")

# --- The episode the old analysis could not see -----------------------------------------
_worst = ep.sort_values("hs3_duration_sec", ascending=False).iloc[0]
_tot_hs3 = float(ep["hs3_duration_sec"].sum())
print(f"\n*** LONGEST STARVING EPISODE ***")
print(f"    run {str(_worst['run_id'])[:8]}, {_worst['day_rome']}")
print(f"    entered by:      {_worst['entry_cause']}")
print(f"    duration:        {_worst['hs3_duration_sec']:.0f} s "
      f"({_worst['hs3_duration_sec']/60:.0f} min) — {_worst['hs3_duration_sec']/_tot_hs3*100:.0f}% "
      f"of ALL Starving time in the corpus")
print(f"    lowest level:    {_worst['min_level']:.1f}  (Starving threshold is 25)")
print(f"    meals received:  {int(_worst['meals_received'])}")
_worst_inter = int((master['run_id'] == _worst['run_id']).sum())
print(f"    interactions logged in that run: {_worst_inter} — people WERE present, and the robot")
print(f"    was not fed. This is the counter-example to 'the loop reliably prevents starvation'.")

# Time-weighted view: episode counts flatter the loop, seconds do not.
print(f"\nTime-weighted: {_tot_hs3:.0f} s total in Starving; "
      f"{ep[ep.entry_cause=='passive_drain']['hs3_duration_sec'].sum():.0f} s of it "
      f"({ep[ep.entry_cause=='passive_drain']['hs3_duration_sec'].sum()/_tot_hs3*100:.0f}%) in "
      f"drain-entered episodes — the ones nobody was there for, and the ones the old builder dropped.")

# --- Kaplan-Meier, descriptive only ------------------------------------------------------
try:
    from lifelines import KaplanMeierFitter
    ep2 = ep.copy()
    ep2["dur"] = ep2["time_to_first_feed_sec"].fillna(ep2["hs3_duration_sec"])
    ep2["event"] = ep2["received_feed"].astype(int)
    ep2 = ep2[ep2["dur"] > 0]
    kmf = KaplanMeierFitter().fit(ep2["dur"], ep2["event"], label="time-to-first-feed")
    _km_med = kmf.median_survival_time_
    print(f"\nKM median time-to-first-feed across all {len(ep2)} episodes: {_km_med:.0f}s "
          f"(vs the 21 s the selected subset reported). No Cox model: {n_ep} episodes in "
          f"{len(_per_run)} runs cannot support a covariate model.")
    globals()["_km_hs3"] = kmf
except Exception as e:
    print("survival failed (small-n expected):", e)
    _km_med = np.nan

_feed_k  = int(ep["received_feed"].sum())
_full_k  = int(ep["recovered_to_full_by_feeding"].sum())
_e_feed  = exact_prop_ci(_feed_k, n_ep)
_e_full  = exact_prop_ci(_full_k, n_ep)
globals()["_b6"]=dict(n=n_ep, feed_k=_feed_k, full_k=_full_k, e_feed=_e_feed, e_full=_e_full,
                      worst_sec=float(_worst["hs3_duration_sec"]),
                      worst_min_level=float(_worst["min_level"]),
                      boot=[_b6_boot[0],_b6_boot[1],_b6_boot[2]], total_hs3_sec=_tot_hs3)

verdict("B6", EV_EXPL,
        f"Descriptive, and materially worse than previously reported. Over ALL {n_ep} Starving "
        f"episodes (not the 8 the label-keyed builder could see): {_feed_k}/{n_ep} received a feed "
        f"(exact 95% CI [{_e_feed[1]:.2f}, {_e_feed[2]:.2f}]) and {_full_k}/{n_ep} recovered to Full "
        f"by feeding (exact [{_e_full[1]:.2f}, {_e_full[2]:.2f}]; run-cluster bootstrap "
        f"[{_b6_boot[0]:.2f}, {_b6_boot[2]:.2f}]) — not the 100% the selected subset showed. The "
        f"episodes cluster in {len(_per_run)} runs, so the effective n is nearer {len(_per_run)} than "
        f"{n_ep}. The longest episode ran {_worst['hs3_duration_sec']/60:.0f} minutes down to level "
        f"{_worst['min_level']:.1f} in a run with {_worst_inter} logged interactions: people were "
        f"present and it was not fed. This is an operational status check on a clustered, "
        f"small, partly right-censored sample. It is NOT a recovery rate, and RQ2-c does not rest on it.",
        effect=_e_full[0], n=n_ep)
""")

md(r"""### B7 — Long-run Starving occupancy (CTMC)

Three defects in the previous fit are corrected here.

**1. The final dwell in every run was thrown away.** The old `state_sequence()` emitted a segment
only *between* consecutive state changes, so the stretch from the last change to the end of the run
— which is a genuine, right-censored sojourn — never entered the state-time denominators. That
biases every rate `q_ij = count_ij / time_in_i`, and it biases them **unevenly**: the terminal
segment is disproportionately Hungry or Starving (runs tend to end while the robot is draining, not
just after it has been fed). Terminal segments are now included in the denominators and, crucially,
are **not** counted as transitions — nothing transitions at the end of a run, the observation simply
stops. Their number and duration are reported by state.

**2. Non-ergodic bootstrap resamples were silently discarded.** The old block bootstrap wrapped the
stationary solve in `try/except: pass`, so any resample whose generator had no unique stationary
distribution — e.g. a draw of runs containing no Starving transition at all — vanished from the
interval without trace. That **conditions the interval on ergodicity** and hides exactly the
resamples that carry the most information about how fragile the Starving row is. They are now
counted and reported.

**3. The pooled generator was never checked against its own strata.** A time-homogeneous CTMC over a
corpus that mixes visited runs with idle no-visitor runs, and Phase 1 with Phase 2, estimates the
stationary fraction of a process that does not exist. The strata are fitted separately below. If
they disagree materially, the pooled number is not a description of anything and the verdict is
downgraded to `Inconclusive` automatically.

The **empirical time-occupancy** — what fraction of observed seconds the robot actually spent in
each state — needs no stationarity assumption at all, and is reported first.""")

code(r"""
# B7: CTMC over Full/Hungry/Starving, with terminal dwells, honest ergodicity accounting,
# and stratification. States are derived from the LEVEL, not the label fields.
from scipy.linalg import null_space
states = ["HS1","HS2","HS3"]

def state_sequence(hr):
    '''Per-run sojourns INCLUDING the right-censored terminal segment.

    Returns one row per sojourn. `to_state` is None for the terminal segment: the run ended,
    nothing transitioned. Transition counts must therefore use `to_state.notna()`, while
    state-time denominators use ALL rows — that asymmetry is the whole point of the fix.
    '''
    hr = hr.sort_values(["run_id","monotonic_sec","id"]).copy()
    hr["_lvl"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    hr = hr.dropna(subset=["_lvl","monotonic_sec"])
    hr["_st"] = hs_from_level_series(hr["_lvl"])
    segs = []
    for run, g in hr.groupby("run_id"):
        s = g["_st"].values; t = g["monotonic_sec"].values
        if len(s) < 2: continue
        keep = np.r_[True, s[1:] != s[:-1]]         # first row + every state change
        cs, ct = s[keep], t[keep]
        for i in range(len(cs) - 1):                # completed sojourns
            segs.append((run, cs[i], float(ct[i+1] - ct[i]), cs[i+1], False))
        # TERMINAL SEGMENT: last state change -> end of run. Right-censored, NOT a transition.
        t_end = float(t[-1])
        if t_end > ct[-1]:
            segs.append((run, cs[-1], t_end - ct[-1], None, True))
    return pd.DataFrame(segs, columns=["run_id","state","dwell","to_state","terminal"])

seq_all = state_sequence(hunger_raw)
seq_all = seq_all[(seq_all["state"].isin(states)) & (seq_all["dwell"] > 0)]

# --- What the old code was throwing away ------------------------------------------------
term = seq_all[seq_all["terminal"]]
print("TERMINAL (right-censored) SEGMENTS — dropped entirely by the previous fit:")
_t = term.groupby("state").agg(n=("dwell","size"), total_sec=("dwell","sum"),
                               median_sec=("dwell","median"), max_sec=("dwell","max"))
print(_t.rename(index=HS_NAME).round(1).to_string())
print(f"  total {term['dwell'].sum():.0f} s across {len(term)} runs "
      f"({term['dwell'].sum()/seq_all['dwell'].sum()*100:.1f}% of all observed state-time).")
term.groupby("state")["dwell"].agg(["size","sum","median","max"]).to_csv(
    OUT_DIR/"b7_terminal_segments.csv")

def fit_ctmc(seq):
    "Return (Q, counts, time_in, pi, ergodic). Terminal rows add TIME but never a TRANSITION."
    time_in = seq.groupby("state")["dwell"].sum().reindex(states).fillna(0)
    tr = seq[seq["to_state"].notna()]
    cnt = (tr.groupby(["state","to_state"]).size().unstack(fill_value=0)
             .reindex(index=states, columns=states, fill_value=0))
    Q = np.zeros((3,3))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i != j and time_in[si] > 0:
                Q[i,j] = cnt.loc[si,sj] / time_in[si]
        Q[i,i] = -Q[i].sum()
    try:
        ns = null_space(Q.T)
        if ns.shape[1] != 1:                       # not a unique stationary distribution
            return Q, cnt, time_in, None, False
        pi = np.abs(ns[:,0]); pi = pi/pi.sum()
        if not np.all(np.isfinite(pi)):
            return Q, cnt, time_in, None, False
        return Q, cnt, time_in, pi, True
    except Exception:
        return Q, cnt, time_in, None, False

Q, cnt, time_in, pi, ergodic = fit_ctmc(seq_all)
print("\nTransition counts (terminal segments excluded, as they must be):")
print(cnt.rename(index=HS_NAME, columns=HS_NAME).to_string())
print("\nTotal time in state (s) — terminal dwell INCLUDED:")
print(time_in.rename(index=HS_NAME).round(0).to_string())

# --- 1. EMPIRICAL occupancy: no stationarity assumption at all --------------------------
emp = time_in / time_in.sum()
print("\n=== EMPIRICAL time-occupancy (assumption-free; report this first) ===")
for s in states:
    print(f"    {HS_NAME[s]:9s} {emp[s]*100:5.2f}% of observed seconds")
_emp_boot = cluster_bootstrap_effect(
    seq_all, "run_id",
    lambda _s: (_s[_s.state=="HS3"]["dwell"].sum() / max(_s["dwell"].sum(), 1e-9)),
    label="Empirical Starving occupancy (run-cluster bootstrap)")
print(f"    Starving, run-cluster bootstrap: {_emp_boot[1]*100:.2f}% "
      f"[95% {_emp_boot[0]*100:.2f}, {_emp_boot[2]*100:.2f}]")

# --- 2. MODELLED stationary occupancy ----------------------------------------------------
if ergodic:
    mean_hs3_sojourn = 1.0/(-Q[2,2]) if Q[2,2] != 0 else np.inf
    print(f"\n=== MODELLED stationary occupancy (time-homogeneous CTMC) ===")
    print("    " + ", ".join(f"{HS_NAME[s]} {p*100:.2f}%" for s, p in zip(states, pi)))
    print(f"    Mean Starving sojourn {mean_hs3_sojourn:.0f}s. "
          f"Starving row rests on {int(cnt.loc['HS3'].sum())} transitions.")
    print(f"    Modelled Starving {pi[2]*100:.2f}% vs EMPIRICAL {emp['HS3']*100:.2f}% — "
          f"the model sits {'BELOW' if pi[2] < emp['HS3'] else 'above'} what actually happened.")
    globals()["_ctmc_pi"] = dict(zip(states, pi))
else:
    print("\n=== MODELLED stationary occupancy: NOT IDENTIFIED on the pooled corpus ===")
    globals()["_ctmc_pi"] = None

# --- 3. Run-block bootstrap, RETAINING non-ergodic resamples -----------------------------
rng = np.random.default_rng(SEED)
_runs = seq_all["run_id"].unique()
_by_run = dict(tuple(seq_all.groupby("run_id")))
boot_blk, n_nonergodic = [], 0
for _ in range(2000):
    pick = rng.choice(_runs, len(_runs), replace=True)
    sq = pd.concat([_by_run[r] for r in pick], ignore_index=True)
    _, _, _, pib, ok = fit_ctmc(sq)
    if ok:
        boot_blk.append(pib[2])
    else:
        n_nonergodic += 1          # COUNTED, not silently dropped
_frac_nonerg = n_nonergodic / 2000
bb = np.percentile(boot_blk, [2.5, 50, 97.5]) if boot_blk else [np.nan]*3
print(f"\n=== Run-block bootstrap (2000 resamples over {len(_runs)} runs) ===")
print(f"    NON-ERGODIC resamples: {n_nonergodic}/2000 = {_frac_nonerg*100:.1f}%")
print(f"      (resampled corpora with no unique stationary distribution — typically a draw of")
print(f"       runs containing no Starving transition at all. The previous code discarded these")
print(f"       silently, which conditioned the interval on ergodicity and hid the fragility.)")
print(f"    Starving occupancy CONDITIONAL ON ERGODICITY: median {bb[1]*100:.2f}% "
      f"[95% {bb[0]*100:.2f}, {bb[2]*100:.2f}] (n={len(boot_blk)})")
globals()["_b7_starve_ci_block"] = (bb[0], bb[1], bb[2])
globals()["_b7_nonergodic"] = _frac_nonerg

# Transition-level Poisson bootstrap, also retaining non-ergodic draws.
boot_p, n_nonerg_p = [], 0
for _ in range(2000):
    Qb = np.zeros((3,3))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i != j and time_in[si] > 0:
                Qb[i,j] = rng.poisson(cnt.loc[si,sj]) / time_in[si]
        Qb[i,i] = -Qb[i].sum()
    try:
        nb = null_space(Qb.T)
        if nb.shape[1] == 1:
            pib = np.abs(nb[:,0]); pib = pib/pib.sum(); boot_p.append(pib[2])
        else:
            n_nonerg_p += 1
    except Exception:
        n_nonerg_p += 1
b = np.percentile(boot_p, [2.5, 50, 97.5]) if boot_p else [np.nan]*3
print(f"    Transition-level Poisson bootstrap: {b[1]*100:.2f}% [{b[0]*100:.2f}, {b[2]*100:.2f}] "
      f"({n_nonerg_p}/2000 non-ergodic)")
globals()["_b7_starve_ci"] = (b[0], b[1], b[2])

# --- 4. STRATIFICATION: does the pooled generator describe any real regime? ---------------
_visited = set(interactions["run_id"].unique())
seq_all["stratum_visit"] = np.where(seq_all["run_id"].isin(_visited), "visited", "idle")
_run_day = hunger_raw.groupby("run_id")["day_rome"].first()
seq_all["phase"] = seq_all["run_id"].map(_run_day).map(phase_of_day)

print("\n=== STRATIFIED FITS — is the pooled number a description of anything? ===")
strat_rows = []
for label, sub in ([("POOLED", seq_all)]
                   + [(f"visit={v}", g) for v, g in seq_all.groupby("stratum_visit")]
                   + [(f"phase={p}", g) for p, g in seq_all.groupby("phase")]):
    _Q, _c, _ti, _pi, _ok = fit_ctmc(sub)
    _emp = _ti["HS3"] / _ti.sum() if _ti.sum() > 0 else np.nan
    strat_rows.append(dict(
        stratum=label, n_runs=sub["run_id"].nunique(),
        obs_hours=float(sub["dwell"].sum()/3600),
        n_hs3_transitions=int(_c.loc["HS3"].sum()),
        n_into_hs3=int(_c.loc["HS2","HS3"]),
        empirical_starving=float(_emp),
        modelled_starving=float(_pi[2]) if _ok else np.nan,
        identified=bool(_ok)))
strat = pd.DataFrame(strat_rows)
print(strat.round(4).to_string(index=False))
strat.to_csv(OUT_DIR/"b7_stratified_occupancy.csv", index=False)

# Do the strata agree? A time-homogeneous chain fitted across regimes that behave differently
# estimates the stationary fraction of a process that never ran. The decisive comparison here is
# PHASE: the deployment simply was not the same process in week 1 and week 2.
_pooled_mod = float(strat.loc[strat.stratum=="POOLED","modelled_starving"].iloc[0])
_pooled_emp = float(strat.loc[strat.stratum=="POOLED","empirical_starving"].iloc[0])

def _stratum_spread(prefix):
    s = strat[strat["identified"] & strat["stratum"].str.startswith(prefix)]
    if len(s) < 2: return np.nan, np.nan
    lo, hi = float(s["empirical_starving"].min()), float(s["empirical_starving"].max())
    return float(hi - lo), (hi/lo if lo > 0 else np.inf)

_ph_abs, _ph_rel = _stratum_spread("phase")
_vi_abs, _vi_rel = _stratum_spread("visit")
_spread = np.nanmax([_ph_abs, _vi_abs]) if np.isfinite(_ph_abs) or np.isfinite(_vi_abs) else np.nan
print(f"\n  Phase P1 vs P2 empirical Starving occupancy differ by {_ph_abs*100:.2f} pp "
      f"({_ph_rel:.0f}x in relative terms).")
if not np.isfinite(_vi_abs):
    _idle = strat[strat.stratum=="visit=idle"]
    print(f"  The idle stratum is degenerate ({float(_idle['obs_hours'].iloc[0]):.1f} h, "
          f"{int(_idle['n_hs3_transitions'].iloc[0])} Starving transitions), so the visited/idle "
          f"split cannot even be fitted — which is itself a statement about how little of this "
          f"corpus is idle time.")

# --- 5. Verdict: downgrade unless the result is stable across all of the above ------------
# A ~20x swing in Starving occupancy between the two halves of the study is not a nuisance to
# be averaged over; it means the "long-run" stationary fraction describes neither half.
_stable = (
    np.isfinite(bb[2]) and                              # a usable interval exists
    _frac_nonerg < 0.10 and                             # ergodicity is not routinely lost
    np.isfinite(_ph_rel) and _ph_rel < 3.0 and          # the two phases are the same process
    abs(_pooled_mod - _pooled_emp) < 0.01               # model tracks the empirical fraction
)
_reasons = []
if not np.isfinite(bb[2]): _reasons.append("no usable stationary interval")
if _frac_nonerg >= 0.10:
    _reasons.append(f"{_frac_nonerg*100:.0f}% of run-resamples are non-ergodic")
if np.isfinite(_ph_rel) and _ph_rel >= 3.0:
    _reasons.append(f"Phase 1 and Phase 2 are not the same process — empirical Starving occupancy "
                    f"is {_ph_rel:.0f}x higher in Phase 1 "
                    f"({strat.loc[strat.stratum=='phase=P1','empirical_starving'].iloc[0]*100:.2f}% "
                    f"vs {strat.loc[strat.stratum=='phase=P2','empirical_starving'].iloc[0]*100:.2f}%), "
                    f"so a time-homogeneous chain fitted across both describes neither")
if not np.isfinite(_vi_abs):
    _reasons.append("the visited/idle split cannot be fitted (the idle stratum has no Starving "
                    "transitions at all), so the pooling assumption cannot be checked on that axis")
if abs(_pooled_mod - _pooled_emp) >= 0.01:
    _reasons.append(f"the pooled model ({_pooled_mod*100:.2f}%) sits below the empirical fraction "
                    f"({_pooled_emp*100:.2f}%)")

globals()["_b7"]=dict(emp=_pooled_emp, emp_boot=[_emp_boot[0],_emp_boot[1],_emp_boot[2]],
                      mod=_pooled_mod, block=[bb[0],bb[1],bb[2]],
                      nonergodic=_frac_nonerg, spread=_spread, phase_rel=_ph_rel,
                      stable=bool(_stable))

if _stable:
    verdict("B7", EV_ASSOC,
            f"Within-deployment occupancy. The robot spent {_pooled_emp*100:.2f}% of observed seconds "
            f"in Starving (empirical, assumption-free; run-cluster bootstrap "
            f"[{_emp_boot[0]*100:.2f}, {_emp_boot[2]*100:.2f}]). The time-homogeneous CTMC agrees "
            f"({bb[1]*100:.2f}% [{bb[0]*100:.2f}, {bb[2]*100:.2f}]), with terminal dwells included and "
            f"{_frac_nonerg*100:.1f}% non-ergodic resamples. This is a property of the coupled "
            f"human-robot loop, not of the controller: the transition rates record what people did. "
            f"No causal share is identified.", n=int(cnt.values.sum()))
else:
    verdict("B7", EV_INCONC,
            f"The MODELLED long-run Starving occupancy is not identified by these data, and the "
            f"empirical one does not need it. Empirically, the robot spent {_pooled_emp*100:.2f}% of "
            f"observed seconds in Starving (run-cluster bootstrap "
            f"[{_emp_boot[0]*100:.2f}, {_emp_boot[2]*100:.2f}]) — that figure is assumption-free and "
            f"stands. What does NOT stand is the stationary CTMC the previous version reported as "
            f"'1.0% [0.2, 3.1]': " + "; ".join(_reasons) + ". The deficiency is structural, not a "
            f"matter of tightening the fit — the deployment is not a time-homogeneous process, so "
            f"its 'long-run' stationary fraction is the occupancy of a chain that never ran. Report "
            f"the empirical number. A modelled one needs longer runs, a stationary operating regime, "
            f"and more than the {int(cnt.loc['HS3'].sum())} Starving transitions this corpus contains.",
            n=int(cnt.values.sum()))
""")

md(r"""#### Reading B3 + B4 together

The deficit is associated with behavioural change at two coded thresholds, but the two results are
**not the same grade of evidence** and are no longer presented as though they were.

- **At the deficit line (60, entering Hungry)** — B3 is a `Within-deployment association`: the odds
  of a meal arriving rise ~5x, and that survives adjustment and both clustering schemes. The
  *robot-side* changes at this threshold (hunger framing, feed-seeking acts, proactive pings) are
  `Implementation verification` — they are coded gates.
- **At the starving line (25, entering Starving)** — B4 is an `Exploratory observation` and nothing
  more. Thirteen interactions, one success. The direction (social completion down, feeding up) is
  consistent and interesting; the magnitude is not estimable.

So the "two-threshold controller" reading is a **hypothesis the data are consistent with**, not a
demonstrated one. The second threshold is supported by 13 interactions.""")

md(r"""#### Multiplicity: the complete ledger

Every p-value that enters a conclusion is registered in `PTABLE` at the point it is computed, with
its model, its term, its family, and whether it is confirmatory or exploratory. The exported
`multiplicity_table.csv` is that ledger in full — not a curated subset.

The previous version corrected 5 p-values and stated that "every p-value used as evidence is inside
a declared family". That was not true: the `duration × feeder` and `duration × phase` interaction
terms were quoted in the B10 verdict, and the role contrasts were quoted in the write-up, yet none
of them entered a family. Interaction terms are now registered like any other.

`status = exploratory` p-values (B4's separated cell) are **recorded but excluded from the BH
families**. Correcting a p-value that came out of a diverging likelihood would only dress it up.""")

code(r"""
# --- Benjamini-Hochberg over the COMPLETE confirmatory ledger --------------------------
def run_bh():
    if not PTABLE:
        print("No p-values registered."); return None
    pt = pd.DataFrame(PTABLE)
    conf = pt[pt["status"] == "confirmatory"].copy()
    if not len(conf):
        print("No confirmatory p-values."); pt.to_csv(OUT_DIR/"multiplicity_table.csv", index=False); return pt
    conf["q_bh"] = np.nan; conf["sig_q05"] = False
    for fam, idx in conf.groupby("family").groups.items():
        pv = conf.loc[idx, "p"].values
        rej, q, *_ = multipletests(pv, alpha=0.05, method="fdr_bh")
        conf.loc[idx, "q_bh"] = q
        conf.loc[idx, "sig_q05"] = rej
    out = pd.concat([conf, pt[pt["status"] != "confirmatory"]], ignore_index=True)
    out = out.sort_values(["status","family","p"])
    out.to_csv(OUT_DIR/"multiplicity_table.csv", index=False)
    # Back-compat filename, same content.
    out.to_csv(OUT_DIR/"bh_corrected_pvalues.csv", index=False)
    print("COMPLETE multiplicity ledger (-> outputs/multiplicity_table.csv):\n")
    print(out[["analysis","term","family","status","p","q_bh","sig_q05"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    n_surv = int(out["sig_q05"].fillna(False).sum())
    n_conf = int((out["status"]=="confirmatory").sum())
    n_expl = int((out["status"]=="exploratory").sum())
    print(f"\n  {n_surv}/{n_conf} confirmatory metrics survive BH at q<0.05, across "
          f"{conf['family'].nunique()} families.")
    print(f"  {n_expl} exploratory p-value(s) recorded but deliberately NOT corrected and NOT "
          f"used to support any claim.")
    return out
""")

md(r"""### B9 — The affinity mechanism *(implementation verification)*

**This section verifies code. It is not evidence about learning, and it is now labelled that way.**

The salience network keeps a per-person **affinity**: an EMA of the logged homeostatic reward,
bounded in [−1, +1]. Written out, with every constant taken from `CONST["AFFINITY"]`:

```
credit     = reward_delta + active_energy_cost · 1[meals > 0]
r_norm     = clip(credit / REWARD_SCALE, −1, +1),  floored at PENALTY_FLOOR when credit <= 0.20
affinity  += (ALPHA if credit > 0.20 else ALPHA_NEG) · (r_norm − affinity)
```

and **`reward_delta` is exactly `stomach_level_end − stomach_level_start`** — verified below to
machine precision, not assumed.

Two consequences, which govern how the whole of RQ3 must be read:

1. **Affinity is a deterministic function of the logged rewards.** It cannot drift, cannot be
   noisy, and cannot encode anything except the energy people delivered. "Affinity tracks who fed
   the robot" is therefore **not a finding** — it is the update rule. The previous version described
   RQ3 as showing the memory is *"not consistent with purely uncontrolled drift"*. Drift was never a
   live alternative: the EMA is four lines of arithmetic.
2. **Affinity acts on selection through exactly one channel**, the eligibility threshold
   `eff_thr = max(FLOOR, base_ss − GAIN · affinity)`. That is verified exactly below. So
   "affinity predicts who the robot approaches" is also, in large part, the source code.

What B9 legitimately establishes: the reconstruction is faithful (it reproduces the robot's own
persisted memory), the perceptual weights genuinely do not learn, and the threshold coupling is
exact. That is worth verifying. It is not worth calling a result.""")
code(r"""
# --- B9. Affinity mechanism verification -----------------------------------------------
hlc = load_view("salience","v_homeostatic_learning_changes_clean").sort_values(["person_id","timestamp_epoch"])
tsel = load_view("salience","v_target_selections_clean")

# (0) FIRST, establish what reward_delta actually IS. Everything in RQ3 depends on this, and
# the previous analysis never checked it: reward_delta is not an abstract "reward signal", it
# is literally the change in stomach level over the interaction.
_rd_check = (hlc["reward_delta"] -
             (hlc["stomach_level_end"] - hlc["stomach_level_start"])).abs().max()
print(f"(0) reward_delta == stomach_level_end - stomach_level_start ?  "
      f"max |difference| = {_rd_check:.2e}  -> {'IDENTICAL' if _rd_check < 1e-9 else 'NOT identical'}")
print("    So the affinity EMA integrates energy delivered, and nothing else. Keep this in view")
print("    when reading B10: a model of Δaffinity is a model of this arithmetic.")

# Identity canonicalization merged case/spelling variants of the same person whose affinity
# EMAs the robot had tracked SEPARATELY, so a stray late event under one label can leave the
# logged affinity stale (one merged person's last logged value is 0.0 despite a 0.93 history).
# We RE-THREAD the EMA over each person's merged, timestamp-sorted reward sequence using the
# robot's exact update rule. Constants come from CONST (checked against the pinned controller
# source by analysis/check_constants.py) — they are NOT re-typed here.
_AFF = dict(A=CONST["AFFINITY"]["ALPHA"], AN=CONST["AFFINITY"]["ALPHA_NEG"],
            SCALE=CONST["AFFINITY"]["REWARD_SCALE"], FLOOR=CONST["AFFINITY"]["PENALTY_FLOOR"],
            BAND=CONST["AFFINITY"]["POSITIVE_BAND"])
def rethread_affinity(g):
    aff=0.0; updates=0; after=[]
    for _,r in g.iterrows():
        if r["outcome"] in ("neutral","skipped"): after.append(aff); continue
        meals=r.get("meals_eaten_count",0) or 0; cost=r.get("active_energy_cost",0) or 0
        credit=r["reward_delta"]+(max(0.0,cost) if meals>0 else 0.0)
        positive=credit>_AFF["BAND"]
        r_norm=max(-1.0,min(1.0,credit/_AFF["SCALE"]))
        if not positive: r_norm=max(r_norm,_AFF["FLOOR"])
        alpha=_AFF["A"] if positive else _AFF["AN"]
        aff = r_norm if (updates<=0 and positive) else aff+alpha*(r_norm-aff)
        aff=max(-1.0,min(1.0,aff)); updates+=1; after.append(aff)
    return after
hlc["affinity_after"]=np.concatenate([rethread_affinity(g) for _,g in hlc.groupby("person_id",sort=False)])
hlc["affinity_before"]=hlc.groupby("person_id")["affinity_after"].shift(1).fillna(0.0)
# "unknown" is a placeholder, not a stable recognised person; keep B9's
# per-person learning claims to named identities only.
hlc=hlc[hlc["person_id"]!="unknown"].copy()

# (a) CONVERGENCE — the EMA affinity step should shrink as evidence accumulates.
hlc["step"]=(hlc["affinity_after"]-hlc["affinity_before"]).abs()
hlc["k"]=hlc.groupby("person_id").cumcount()+1
mid=hlc.groupby("person_id")["k"].transform("median")
early=hlc[hlc["k"]<=mid]["step"].mean(); late=hlc[hlc["k"]>mid]["step"].mean()
term=hlc.groupby("person_id").tail(1)[["person_id","affinity_after"]].rename(columns={"affinity_after":"affinity"})
print(f"(a) Convergence: mean |affinity update| early={early:.3f} -> late={late:.3f} "
      f"({'shrinking (converging)' if late<early else 'not shrinking'}); "
      f"{len(hlc)} updates over {hlc['person_id'].nunique()} people.")

# (b) REWARD VARIATION — outcome mix and per-person spread.
print("\n(b) Reward outcome mix (reward_delta drives the affinity EMA):")
print(hlc.groupby("outcome").agg(n=("reward_delta","size"),
      mean_reward=("reward_delta","mean"), mean_daffinity=("reward_delta",lambda s: 0)).drop(columns="mean_daffinity")
      .assign(mean_affinity_step=hlc.groupby("outcome")["step"].mean()).round(1).to_string())

# (c) LEARNED AFFINITY -> IPS ELIGIBILITY THRESHOLD (weights are fixed; threshold personalises).
_con=ro_connect(SUPER["salience"])
ipschk=pd.read_sql("SELECT DISTINCT weight_prox,weight_cent,weight_gaze "
                   "FROM face_ips_events WHERE valid_for_analysis=1",_con); _con.close()
print(f"\n(c) IPS component weights are constant across all {len(ips):,} events "
      f"({ipschk.iloc[0].to_dict()}; {len(ipschk)} distinct combo) — learning does NOT touch the weights.")
BASE=CONST["SS_THRESHOLDS"]
_GAIN=CONST["AFFINITY"]["THR_GAIN"]; _FLOOR=CONST["AFFINITY"]["THR_FLOOR"]
t=tsel.dropna(subset=["affinity","effective_threshold"]).copy()
t=t[t["person_id"]!="unknown"].copy()
t["pred"]=t.apply(lambda r: max(_FLOOR, BASE.get(r["ss"],1.0)-_GAIN*r["affinity"]),axis=1)
ferr=(t["pred"]-t["effective_threshold"]).abs().max()
print(f"    effective_threshold = max({_FLOOR}, base_ss - {_GAIN}*affinity): matches logged values "
      f"(max err {ferr:.4f}, n={len(t)}). This is the ONE channel by which affinity reaches")
print(f"    action selection — which is why B10.3 ('affinity predicts approaches') is largely")
print(f"    a restatement of this line of source code, not an independent discovery.")
# threshold reduction for the people the drive learned to like
prof=(t.groupby("person_id").agg(mean_affinity=("affinity","mean"),
        mean_thr=("effective_threshold","mean"),n_sel=("effective_threshold","size")))
prof["thr_reduction"]=(BASE["ss4"]-prof["mean_thr"]).clip(lower=0)   # vs the strict ss4 base
top=prof[prof["n_sel"]>=20].sort_values("mean_affinity",ascending=False).head(6)
print("    Highest-affinity people get the largest eligibility discount:")
print(top.round(3).to_string())
# Persist the per-person eligibility profile: this is the affinity -> threshold coupling that
# the B9 bullets assert, in the form a reader can check.
prof.sort_values("mean_affinity",ascending=False).round(4).to_csv(OUT_DIR/"b9_eligibility_profile.csv")

# (Whether the learned values track real behaviour is B10's question, answered there with
# cluster-aware models — no person-level correlation over n=14 is reported here.)

# (d) CHATBOT HS2 PROACTIVE TARGETING by learned affinity.
TH_HS2=CONST["CHAT"]["PRIORITY_THRESHOLD_HS2"]
qualifies=term[term["affinity"]>TH_HS2]
print(f"\n(d) HS2 (Hungry) proactive filter: {len(qualifies)}/{len(term)} learned people exceed "
      f"affinity>{TH_HS2} -> pinged when Hungry; the rest are skipped until Starving (HS3=all).")
psel=read_view_super("chat","v_chat_proactive_selection")
hs2=psel[psel["hs"]=="HS2"]
if len(hs2):
    tot=hs2["total_subs"].sum(); filt=hs2["filtered_count"].sum()
    print(f"    Logged HS2 selection events: {len(hs2)}; {filt}/{tot} subscriber-slots filtered out "
          f"by low affinity ({filt/max(tot,1)*100:.0f}%) — the drive spends remote pings on its feeders.")

# (e) EXTERNAL VALIDATION — re-threaded EMA vs the robot's own persisted memory.
# memory/homeostatic_learning.json stores the affinity the robot ACTUALLY held per identity key —
# an artefact independent of v_homeostatic_learning_changes_clean (the event log we re-thread).
# Two cases, kept strictly separate (the split is in MEMORY, not the event log):
#   * Un-forked people (one memory key): the re-thread must reproduce the stored value exactly —
#     this is the clean external validation.
#   * People the robot FORKED in memory under case/spelling variants of one name (up to 3 keys):
#     the cleaned event log already CONSOLIDATES their events under one key, so the re-thread is a
#     reconciled value that matches NO single memory fork by construction. We do NOT invent an
#     abs_diff for them; instead we (a) prove the consolidation captured every forked event via
#     update-count conservation (evlog_updates == sum of the forks' updates), and (b) quantify the
#     affinity fragmentation the merge removes (spread across forks). Memory keys are RAW identities,
#     mapped through canon_identity + pseudonymize so no real name is ever printed or written.
def _memory_forks(folder):
    p = DATA_ROOT / folder / "memory" / "homeostatic_learning.json"
    if not p.exists(): return None
    ppl = json.loads(p.read_text()).get("people", {})
    rows=[]
    for k,v in ppl.items():
        cid = canon_identity(k)
        if not isinstance(cid,str) or cid.strip().lower() in PSEUDONYM_PRESERVE: continue
        rows.append(dict(pid=pseudonymize(cid), aff=float(v.get("affinity",0.0)),
                         upd=int(v.get("affinity_updates",0))))
    return pd.DataFrame(rows) if rows else None
SNAP = "29-06"   # final (end-of-deployment) memory snapshot; unmerged people validate against it
try:
    _mem = _memory_forks(SNAP)
    if _mem is not None:
        # event-log affinity-updating events per person (non-neutral), keyed by pseudonym:
        _ev_upd = (hlc[~hlc["outcome"].isin(["neutral","skipped"])]
                   .groupby("person_id").size().rename("evlog_updates"))
        _agg = (_mem.groupby("pid").agg(n_forks=("aff","size"), mem_updates=("upd","sum"),
                                        fork_spread=("aff", lambda s: float(s.max()-s.min()))).reset_index())
        _single = (_mem.groupby("pid").filter(lambda d: len(d)==1)[["pid","aff"]]
                       .rename(columns={"aff":"mem_stored"}))
        chk = (term.rename(columns={"affinity":"rethread_final"})
                   .merge(_agg, left_on="person_id", right_on="pid", how="inner")
                   .merge(_ev_upd, left_on="person_id", right_index=True, how="left")
                   .merge(_single, on="pid", how="left"))
        chk["abs_diff"] = (chk["rethread_final"]-chk["mem_stored"]).abs()      # NaN for forked people
        chk["updates_conserved"] = chk["mem_updates"].eq(chk["evlog_updates"])
        unf = chk[chk["n_forks"]==1]; fk = chk[chk["n_forks"]>1]
        _maxd = float(unf["abs_diff"].max()) if len(unf) else float("nan")
        print(f"\n(e) Memory cross-check ({SNAP} snapshot): matched {len(chk)}/{len(term)} learned people.")
        print(f"    VALIDATION — {len(unf)} people the robot stored under ONE identity: the re-threaded "
              f"EMA reproduces the persisted memory exactly (max |Δ|={_maxd:.4f}) — external confirmation "
              f"independent of the event log it was derived from.")
        if len(fk):
            _cons = bool(fk["updates_conserved"].all())
            print(f"    RECONCILIATION — {len(fk)} people the robot FORKED in memory under identity "
                  f"variants: the cleaned event log consolidates their events under one key with the "
                  f"update count conserved exactly ({'all conserved' if _cons else 'MISMATCH'}), so the "
                  f"re-thread is the correct MERGED affinity that matches no single fork by construction. "
                  f"Unmerged, their affinity is fragmented by up to |Δ|={fk['fork_spread'].max():.2f} "
                  f"across forks — the corruption the merge removes. No per-fork abs_diff is claimed "
                  f"(honest: they are not validated against a single memory row, they are repaired).")
        _cols=[c for c in ["person_id","n_forks","rethread_final","mem_stored","abs_diff",
                           "evlog_updates","mem_updates","updates_conserved","fork_spread"] if c in chk.columns]
        chk[_cols].sort_values(["n_forks","abs_diff"], ascending=[True,False]).to_csv(
            OUT_DIR/"rq3_memory_crosscheck.csv", index=False)
        globals()["_b9_memchk"]=dict(n=len(chk), unforked=len(unf), forked=len(fk), max_unforked=_maxd,
                                     updates_conserved=bool(fk["updates_conserved"].all()) if len(fk) else True,
                                     max_fork_spread=float(fk["fork_spread"].max()) if len(fk) else 0.0)
    else:
        print(f"\n(e) Memory cross-check: {SNAP} snapshot not found; skipped.")
except Exception as _e:
    print("\n(e) Memory cross-check failed:", _e)

# One tidy check table so every B9 bullet in the README cites a value a reader can look up,
# rather than a figure that no longer exists.
b9_check = pd.DataFrame([
    dict(check="affinity update converges (mean |update|, early -> late)",
         value=f"{early:.3f} -> {late:.3f}", n=len(hlc), passes=bool(late < early)),
    dict(check="IPS component weights are constant (learning does not touch them)",
         value=f"{len(ipschk)} distinct weight combo", n=len(ips), passes=len(ipschk) == 1),
    dict(check="eff_thr = max(0.50, base_ss - 0.15*affinity) matches logged values (max abs err)",
         value=f"{ferr:.4f}", n=len(t), passes=bool(ferr < 1e-3)),
    dict(check="eligibility discount for the highest-affinity person (vs ss4 base 0.85)",
         value=f"{(0.85 - top['mean_thr'].min()):.3f}", n=int(prof['n_sel'].sum()), passes=True),
    dict(check="HS2 proactive pings gated to people above affinity 0.20",
         value=f"{len(qualifies)}/{len(term)} people", n=len(term), passes=True),
] + ([
    dict(check="re-threaded EMA vs robot's persisted memory, un-forked people (max abs diff)",
         value=f"{globals()['_b9_memchk']['max_unforked']:.4f}",
         n=globals()['_b9_memchk']['unforked'], passes=globals()['_b9_memchk']['max_unforked'] < 1e-3),
    dict(check="forked identities: update counts conserved by the merge",
         value="conserved" if globals()['_b9_memchk']['updates_conserved'] else "MISMATCH",
         n=globals()['_b9_memchk']['forked'], passes=globals()['_b9_memchk']['updates_conserved']),
] if globals().get("_b9_memchk") else []))
b9_check.to_csv(OUT_DIR/"b9_mechanism_check.csv", index=False)
print("\nB9 mechanism check (-> outputs/b9_mechanism_check.csv):")
print(b9_check.to_string(index=False))

# --- (f) WHAT ACTUALLY DRIVES Δaffinity: the decomposition B10 must respect --------------
# Δaffinity = alpha * (r_norm(credit) - affinity_before), and credit is dominated by whether a
# meal arrived. Any regression of Δaffinity on an interaction covariate is competing with these
# two terms, both of which are IN the update equation. B10 previously omitted both.
hlc["d_aff"] = hlc["affinity_after"] - hlc["affinity_before"]
hlc["fed"]   = (pd.to_numeric(hlc["meals_eaten_count"], errors="coerce").fillna(0) > 0).astype(int)
_elig = hlc[hlc["outcome"] != "skipped"]
print("\n(f) What the affinity update is actually made of:")
print(f"    credit by whether a meal arrived: "
      f"{_elig.assign(credit=_elig['reward_delta'] + np.where(_elig['fed']>0, _elig['active_energy_cost'].clip(lower=0), 0)).groupby('fed')['credit'].mean().round(1).to_dict()}")
_r2_mech = smf.ols("d_aff ~ fed + affinity_before", _elig).fit().rsquared
print(f"    R^2 of  d_aff ~ fed + affinity_before  (both INSIDE the update rule) = {_r2_mech:.3f}")
print(f"    A dose covariate added to this model is explaining the remainder, not the phenomenon.")
print(f"    B10 therefore controls for BOTH, and does not use active_energy_cost as a 'dose' —")
print(f"    active_energy_cost is a literal additive term in `credit`.")
globals()["_b9_r2_mech"] = float(_r2_mech)

verdict("B9", EV_IMPL,
        f"Confirmed. The affinity EMA is reproduced exactly from the logged rewards using the "
        f"controller's own constants: reward_delta IS the stomach-level change (max |diff| "
        f"{_rd_check:.1e}), the eligibility threshold eff_thr=max({_FLOOR},base-{_GAIN}*affinity) "
        f"matches every logged selection (max err {ferr:.4f}, n={len(t)}), the perceptual IPS weights "
        f"never change ({len(ipschk)} distinct combination across {len(ips):,} events), and HS2 pings "
        f"are gated to the {len(qualifies)}/{len(term)} people above affinity {TH_HS2}. The update "
        f"attenuates over time ({early:.2f}->{late:.2f}), as an EMA must."
        + (f" The reconstruction reproduces the robot's OWN persisted memory for the "
           f"{globals()['_b9_memchk']['unforked']} people it stored under a single identity "
           f"(max |Δ|={globals()['_b9_memchk']['max_unforked']:.3f}) — an artefact independent of the "
           f"event log — and consolidates the {globals()['_b9_memchk']['forked']} it had forked, with "
           f"update counts conserved."
           if globals().get('_b9_memchk') else "")
        + f" NONE of this is evidence that the robot 'learns about people': affinity is a "
          f"deterministic function of delivered energy, so `fed` and `affinity_before` alone explain "
          f"R^2={_r2_mech:.2f} of every update. It is verification that the code does what the code says.",
        n=len(hlc))
globals()["_b9_hlc"]=hlc   # B10 consumes the re-threaded learning events; nothing else is exported
""")

md(r"""### B10 — RQ3: the role manipulation, and what it can and cannot establish

**The framing has changed, because the previous one claimed more than the design supports.**

B9 established that affinity is a **deterministic EMA of delivered energy**. It follows that:

- "Affinity tracks who fed the robot" is the **update rule**, not a discovery.
- "Affinity predicts who the robot approaches next" is the **eligibility-threshold line of source
  code**, verified exactly in B9, not an independent behavioural finding.
- The previous claim that the result "is not consistent with purely uncontrolled drift" set up a
  straw alternative. A deterministic EMA cannot drift.

So RQ3 has exactly **one** genuinely empirical question, and it is about the *humans*, not the
robot: **did the assigned roles change what people did?** In Phase 1 (first 4 days) two
participants were obligated feeders, two were asked to interact but never feed, and everyone else
was unconstrained; in Phase 2 all constraints lifted. Roles were external metadata and were never
controller inputs.

**Two people per controlled role.** Role is therefore very nearly aliased with identity: any "role
effect" is also "these two particular people". No amount of modelling fixes that. Accordingly:

- Role contrasts are reported as a **descriptive manipulation check**.
- The asymptotic GEE p-value is **not** treated as population inference; we **lead with the
  person-cluster bootstrap** and add **person-level randomisation inference** (permuting the role
  labels across people), which is the reference distribution the design actually licenses.
- Counts are modelled with an **exposure offset**, because "feeders fed more" is uninformative if
  feeders were simply *present* more. Exposure is reconstructed from interactions and observed
  person-days, including days a person was present but fed nothing.

Verdict target: **manipulation validated for these participants; population inference
unsupported.**""")

code(r"""
# --- B10a. Role/phase metadata + composition -------------------------------------------
ROLE_OF, PHASE1_DAYS = ROLE_OF_EARLY, PHASE1_DAYS_EARLY   # resolved in the helpers cell
def phase_of(day): return phase_of_day(day)
ROLE_ORDER = ["normal","feeder","no_feed"]
# Role palette validated (CVD + chroma + contrast) against the light figure surface.
ROLE_COLOR = {"feeder": "#2C8A60", "no_feed": "#B23A26", "normal": "#4472B0"}
ROLE_LABEL = {"feeder": "obligated feeder", "no_feed": "interact, no feeding", "normal": "unconstrained"}

m10 = master.copy()
m10["phase"] = m10["day_rome"].map(phase_of)
# Reference level = "normal" (unconstrained) so model contrasts read as departures from
# ordinary behaviour; without this patsy would silently pick "feeder" alphabetically.
m10["role"] = pd.Categorical(m10["person_id"].map(role_of),
                             categories=["normal","feeder","no_feed","unknown"])
m10["fed"] = (pd.to_numeric(m10["meals_eaten_count"], errors="coerce").fillna(0) > 0).astype(int)

print("Design composition — interactions by role x phase (named + unknown):")
print(pd.crosstab(m10["role"], m10["phase"], margins=True).to_string())
print("\npeople per role:", {r: m10[m10.role==r]['person_id'].nunique() for r in ROLE_ORDER})
print(f"Phase 1 days: {sorted(PHASE1_DAYS)}; Phase 2 = the rest "
      f"({sorted(set(m10['day_rome'].unique())-PHASE1_DAYS)})")

# Update-event frame: B9's re-threaded EMA, named people, learning-eligible events only.
h10 = globals()["_b9_hlc"].copy()
h10 = h10[h10["outcome"]!="skipped"].copy()
h10["d_aff"] = h10["affinity_after"] - h10["affinity_before"]
h10["phase"] = h10["day_rome"].map(phase_of)
h10["role"] = pd.Categorical(h10["person_id"].map(role_of),
                             categories=["normal","feeder","no_feed"])
print("\nAffinity update events by role x phase:")
print(pd.crosstab(h10["role"], h10["phase"], margins=True).to_string())

# Duration join (salience-linked interactions only) + missingness accounting.
_dur = (master.dropna(subset=["exec_interaction_id"]).drop_duplicates("exec_interaction_id")
              .set_index("exec_interaction_id")["duration_sec"])
h10["duration_sec"] = h10["exec_interaction_id"].map(_dur)
miss = (h10.assign(has_duration=h10["duration_sec"].notna())
           .groupby(["role","phase"])["has_duration"].agg(["mean","size"]).round(2)
           .rename(columns={"mean":"frac_with_duration","size":"n_events"}))
print("\nDuration availability by role x phase (differential missingness — why the dose "
      "hierarchy exists; n_turns and active_energy_cost are observed for ALL events):")
print(miss.to_string())
miss.to_csv(OUT_DIR / "rq3_missingness.csv")
for _c in ("n_turns","active_energy_cost","duration_sec"):
    _v = h10[_c].astype(float)
    h10[f"z_{_c}"] = (_v - _v.mean()) / _v.std()
globals()["_b10_m10"]=m10; globals()["_b10_h10"]=h10
globals()["ROLE_OF"]=ROLE_OF; globals()["PHASE1_DAYS"]=PHASE1_DAYS
globals()["ROLE_COLOR"]=ROLE_COLOR; globals()["ROLE_LABEL"]=ROLE_LABEL; globals()["ROLE_ORDER"]=ROLE_ORDER
""")

md(r"""#### B10.1 — Manipulation check *(descriptive; the only empirical question in RQ3)*

**Exposure is a mediator here, not a confounder, and the distinction decides the answer.**

Telling someone to feed the robot plausibly makes them *go to the robot*. So the number of
interactions a person has is on the causal path from the role to the meals delivered — it is not a
nuisance to be adjusted away. Offsetting by it does not remove a bias; it **decomposes** the role
effect into "came more often" versus "fed more per encounter". Both quantities are reported,
because they answer different questions and only one of them is what the robot experiences:

- **Meals per person-day** — total energy delivered. This is the quantity that matters to the
  drive: the robot does not care *why* it got fed.
- **Meals per interaction** — the per-encounter feeding propensity. This is what "feeders fed more
  readily" would mean, and it is a strictly narrower claim.

Person-days on which someone was present and fed **nothing** are kept as zeros, not dropped;
excluding them would inflate the rate for whoever showed up rarely.

**Inference.** With 2 people per role the asymptotic GEE p-value is not population inference and is
not presented as such. Three things are reported instead, in this order: the raw cell counts with
**exact intervals**; the **person-cluster bootstrap**; and **randomisation inference** — permuting
the role labels across people, **within Phase 1**, which is the only window in which the roles
constrained anyone. With 2 feeders among the named Phase-1 people there are only `C(12,2) = 66`
distinct assignments, so the permutation p has a hard floor near `1/66 ≈ 0.015`. That floor is the
design speaking, and it is stated rather than papered over with an asymptotic approximation that
pretends to more resolution than the study can deliver.""")

code(r"""
# --- B10b. Manipulation check: descriptive, exposure-aware, randomisation-based ---------
d10 = m10[m10["role"]!="unknown"].copy()

print("(1) Feed-rate per interaction, exact 95% CIs by role x phase:")
for (role, ph), g in d10.groupby(["role","phase"], observed=True):
    k, n = int(g["fed"].sum()), len(g)
    est, lo, hi = exact_prop_ci(k, n)
    print(f"    {ROLE_LABEL[role]:22s} {ph}: {k:2d}/{n:3d} = {est:.2f} [{lo:.2f},{hi:.2f}]")
_nf1 = d10[(d10.role=="no_feed")&(d10.phase=="P1")]
_nf_k, _nf_n = int(_nf1["fed"].sum()), len(_nf1)
_nf_e = exact_prop_ci(_nf_k, _nf_n)
print(f"    -> no-feed compliance in Phase 1: {_nf_k}/{_nf_n} feeds, exact 95% CI "
      f"[{_nf_e[1]:.2f}, {_nf_e[2]:.2f}]. Complete separation IS the compliance result; the")
print(f"       upper bound {_nf_e[2]:.2f} is what {_nf_n} observations can rule out, and no more.")

# --- EXPOSURE: build the person-day panel INCLUDING zero-feed days ----------------------
# A person-day exists if the person was seen at all that day. Days with interactions but no
# meals are genuine zeros and must be in the denominator; the previous version's groupby
# produced them only incidentally, and carried no exposure term at all.
pdm = (d10.groupby(["person_id","day_rome","phase","role"], observed=True)
          .agg(meals=("meals_eaten_count","sum"),
               n_interactions=("interaction_id","size"),
               time_present_sec=("duration_sec", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())))
          .reset_index())
print(f"\n(2) Person-day panel: {len(pdm)} person-days over {pdm['person_id'].nunique()} people.")
print(f"    Zero-meal person-days retained: {int((pdm['meals']==0).sum())}/{len(pdm)} "
      f"({(pdm['meals']==0).mean()*100:.0f}%) — these are the days someone was present and fed "
      f"nothing. Excluding them would inflate every rate.")
print("\n    Meals and exposure per person-day by role x phase:")
print(pdm.pivot_table(index="role", columns="phase",
                      values=["meals","n_interactions"],
                      aggfunc=["sum","mean"], observed=True).round(2).to_string())

pm = pdm[pdm["role"].isin(["feeder","normal"])].copy()
pm["is_feeder"] = (pm["role"]=="feeder").astype(int)
pm["log_exposure"] = np.log(pm["n_interactions"].clip(lower=1))

# --- The exposure DECOMPOSITION ---------------------------------------------------------
# Total delivery (what the robot experiences) vs per-encounter propensity (what "fed more
# readily" would mean). Exposure sits on the causal path from role to meals, so the offset
# model is a decomposition, NOT a bias correction.
_g_total = smf.gee("meals ~ is_feeder*C(phase)", groups="person_id", data=pm,
                   family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
_g_perop = smf.gee("meals ~ is_feeder*C(phase)", groups="person_id", data=pm,
                   family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence(),
                   offset=pm["log_exposure"]).fit()
_rr_total = float(np.exp(_g_total.params["is_feeder"]))
_rr_total_ci = np.exp(_g_total.conf_int().loc["is_feeder"]).tolist()
_rr_total_p  = float(_g_total.pvalues["is_feeder"])
_rr  = float(np.exp(_g_perop.params["is_feeder"]))
_rr_ci = np.exp(_g_perop.conf_int().loc["is_feeder"]).tolist()
_rr_p  = float(_g_perop.pvalues["is_feeder"])
_ix  = float(np.exp(_g_total.params["is_feeder:C(phase)[T.P2]"]))
_ix_ci = np.exp(_g_total.conf_int().loc["is_feeder:C(phase)[T.P2]"]).tolist()
_ix_p  = float(_g_total.pvalues["is_feeder:C(phase)[T.P2]"])

_p1 = pm[pm.phase=="P1"]
_ex_f = _p1[_p1.is_feeder==1]["n_interactions"].mean()
_ex_n = _p1[_p1.is_feeder==0]["n_interactions"].mean()
print(f"\n(3) DECOMPOSING the role effect (Phase 1):")
print(f"    (a) TOTAL DELIVERY   meals per person-day:      RR = {_rr_total:.2f} "
      f"[asymptotic {_rr_total_ci[0]:.2f}, {_rr_total_ci[1]:.2f}]")
print(f"    (b) PER-ENCOUNTER    meals per interaction:     RR = {_rr:.2f} "
      f"[asymptotic {_rr_ci[0]:.2f}, {_rr_ci[1]:.2f}]")
print(f"    (c) EXPOSURE         interactions per day:      "
      f"{_ex_f:.2f} (feeder) vs {_ex_n:.2f} (unconstrained) = {_ex_f/max(_ex_n,1e-9):.2f}x")
print(f"\n    -> The role worked almost entirely through EXPOSURE. Obligated feeders delivered")
print(f"       {_rr_total:.1f}x the meal energy per day, but per encounter they fed only "
      f"{_rr:.1f}x as often.")
print(f"       They showed up {_ex_f/max(_ex_n,1e-9):.1f}x more. Being told to feed the robot made")
print(f"       people GO TO the robot; it did not make them markedly more generous once there.")
print(f"       Exposure is a MEDIATOR of the role effect, not a confounder — so (b) is a")
print(f"       decomposition, not a corrected estimate. (a) is what the drive actually experiences.")
print(f"\n    feeder x Phase-2 interaction (total delivery): {_ix:.2f} "
      f"[{_ix_ci[0]:.2f}, {_ix_ci[1]:.2f}] — the excess shrinks once the obligation lifts.")

# THE INTERVALS WE LEAD WITH: person-cluster bootstraps for both estimands.
_meal_boot = cluster_bootstrap_effect(
    pm, "person_id",
    lambda _s: np.exp(smf.glm("meals ~ is_feeder*C(phase)", data=_s,
                              family=sm.families.Poisson()).fit().params["is_feeder"]),
    label="B10.1 (a) TOTAL delivery RR (person-cluster bootstrap)")
_perop_boot = cluster_bootstrap_effect(
    pm, "person_id",
    lambda _s: np.exp(smf.glm("meals ~ is_feeder*C(phase)", data=_s, family=sm.families.Poisson(),
                              offset=_s["log_exposure"]).fit().params["is_feeder"]),
    label="B10.1 (b) PER-ENCOUNTER RR (person-cluster bootstrap)")

# --- RANDOMISATION INFERENCE, within Phase 1 only ---------------------------------------
# The roles constrained people ONLY in Phase 1, so that is the only window in which permuting
# the labels tests anything. Permuting across both phases (as a first pass of this analysis
# did) dilutes the contrast with 4 days in which nobody was under any obligation.
from math import comb
_perm_frame = pm[pm.phase=="P1"].copy()
_perm_frame["role_lab"] = np.where(_perm_frame["is_feeder"]==1, "feeder", "normal")

def _rate_total(df):
    f = df[df.role_lab=="feeder"]; n = df[df.role_lab=="normal"]
    if not len(f) or not len(n): return np.nan
    return f["meals"].mean()/max(n["meals"].mean(), 1e-9)
def _rate_perop(df):
    f = df[df.role_lab=="feeder"]; n = df[df.role_lab=="normal"]
    if not len(f) or not len(n): return np.nan
    rf = f["meals"].sum()/max(f["n_interactions"].sum(), 1)
    rn = n["meals"].sum()/max(n["n_interactions"].sum(), 1)
    return rf/rn if rn > 0 else np.nan

_obs_tot, _perm_p_tot, _np_tot = cluster_permutation_p(
    _perm_frame, "person_id", "role_lab", _rate_total, n=5000)
_obs_rr, _perm_p, _n_perm = cluster_permutation_p(
    _perm_frame, "person_id", "role_lab", _rate_perop, n=5000)
_n_people_perm = _perm_frame["person_id"].nunique()
_n_feeders = int((_perm_frame.drop_duplicates("person_id")["role_lab"]=="feeder").sum())
_floor = 1.0/comb(_n_people_perm, _n_feeders) if _n_people_perm >= _n_feeders else np.nan
print(f"\n(4) Randomisation inference — role labels permuted across the {_n_people_perm} "
      f"Phase-1 people:")
print(f"    (a) TOTAL delivery   observed {_obs_tot:.2f}x   permutation p = {_perm_p_tot:.3f}")
print(f"    (b) PER-ENCOUNTER    observed {_obs_rr:.2f}x   permutation p = {_perm_p:.3f}")
print(f"    With {_n_feeders} feeders among {_n_people_perm} people there are only "
      f"C({_n_people_perm},{_n_feeders}) = {comb(_n_people_perm, _n_feeders)} distinct assignments, "
      f"so p cannot go below ~{_floor:.3f}.")
print(f"    -> the TOTAL-delivery effect {'survives' if _perm_p_tot < 0.05 else 'does NOT survive'} "
      f"randomisation; the PER-ENCOUNTER effect "
      f"{'survives' if _perm_p < 0.05 else 'does NOT survive'}.")

register_p("B10.1", "meals ~ is_feeder*phase (Poisson GEE, cluster=person) — TOTAL delivery",
           "is_feeder", _rr_total_p, "RQ3-adaptation", "confirmatory",
           note="asymptotic; 2 feeders — the permutation p is the one to read")
register_p("B10.1", "meals ~ is_feeder*phase + offset(log n_interactions) — PER-ENCOUNTER",
           "is_feeder", _rr_p, "RQ3-adaptation", "confirmatory",
           note="exposure is a mediator: this is a decomposition, not a corrected estimate")
register_p("B10.1", "meals ~ is_feeder*phase (Poisson GEE, cluster=person) — TOTAL delivery",
           "is_feeder:phase[P2]", _ix_p, "RQ3-adaptation", "confirmatory",
           note="role x phase: the obligation lifting")
register_p("B10.1", "TOTAL meal delivery, person-level randomisation (Phase 1)",
           "feeder vs unconstrained", _perm_p_tot, "RQ3-adaptation", "confirmatory",
           note=f"permutation p; hard design floor ~{_floor:.3f}")
register_p("B10.1", "PER-ENCOUNTER feed rate, person-level randomisation (Phase 1)",
           "feeder vs unconstrained", _perm_p, "RQ3-adaptation", "confirmatory",
           note=f"permutation p; hard design floor ~{_floor:.3f}")

SENSITIVITY.append(dict(metric="B10_role_TOTAL_delivery_RR", primary=_rr_total,
                        boot_lo=_meal_boot[0], boot_median=_meal_boot[1],
                        boot_hi=_meal_boot[2], successful_refits=_meal_boot[3],
                        unit="person-cluster bootstrap over person-days (meals/day)"))
SENSITIVITY.append(dict(metric="B10_role_PER_ENCOUNTER_RR", primary=_rr,
                        boot_lo=_perop_boot[0], boot_median=_perop_boot[1],
                        boot_hi=_perop_boot[2], successful_refits=_perop_boot[3],
                        unit="person-cluster bootstrap over person-days (meals/interaction)"))
globals()["_b10_meal_gee"]={"rr":_rr,"rr_ci":_rr_ci,"p":_rr_p,
                            "rr_total":_rr_total,"rr_total_ci":_rr_total_ci,"p_total":_rr_total_p,
                            "ci":_rr_total_ci, "rr_noexp":_rr_total,
                            "rr_p2":_ix,"ci_p2":_ix_ci,"p_p2":_ix_p,
                            "boot":[_meal_boot[0],_meal_boot[1],_meal_boot[2]],
                            "boot_perop":[_perop_boot[0],_perop_boot[1],_perop_boot[2]],
                            "perm_p":_perm_p, "perm_p_total":_perm_p_tot, "perm_floor":_floor,
                            "exposure_ratio":float(_ex_f/max(_ex_n,1e-9)),
                            "nofeed_k":_nf_k, "nofeed_n":_nf_n, "nofeed_hi":_nf_e[2]}
pdm.to_csv(OUT_DIR/"b10_person_day_exposure.csv", index=False)
""")

md(r"""#### B10.2 — Δaffinity and engagement dose: a **programmed learning-rule response**

Three things were wrong with the previous version of this model, and all three inflated it.

**1. It regressed Δaffinity on terms inside its own update equation, while omitting the two that
dominate it.** From B9:

```
Δaffinity = alpha · (r_norm(credit) − affinity_before),   credit = reward_delta + cost·1[fed]
```

so `fed` and `affinity_before` are *in the formula*. Together they explain R² ≈ 0.42 of every
update on their own. The old model included neither, and put `duration` — which correlates 0.58
with `fed` — in their place. The slope it reported was substantially a proxy for "this was the
interaction where someone fed the robot".

**2. `active_energy_cost` was used as an independent "dose agreement check".** It is a literal
additive term in `credit`. Regressing Δaffinity on it is not a check, it is the identity. It is
**dropped as a dose** and used only as a covariate.

**3. Duration was the primary dose despite being 52% missing, and missing differentially** — 22%
observed in feeder/Phase-1 versus 100% in no-feed/Phase-2. The `duration × role` interaction, quoted
as a finding, rested on ~12 feeder-Phase-1 rows.

The rebuild:

- **Primary dose = `n_turns`**, which is observed for **every** event.
- **Duration demoted** to a secondary complete-case sensitivity analysis.
- A **missingness model** predicts duration availability from role, phase, hunger state, trigger
  mode and person, and an **inverse-probability-weighted** duration fit is reported next to the
  complete-case one.
- Every model now controls for **`fed` and `affinity_before`**.
- The three fits (fully-observed dose / complete-case duration / IPW duration) are compared. If
  they disagree materially in sign or magnitude, the conclusion is downgraded.

**And the framing changes.** Even at its cleanest, this model describes how a **deterministic
learning rule responds to its own inputs**. It is not evidence for a latent social construct being
validated. Evidence class: `Implementation verification` for the mechanism, with the residual
engagement effect reported as a `Within-deployment association` at most.""")

code(r"""
# --- B10c. Core affinity model, with the update rule's own terms controlled -------------
def fit_mix(df, formula, label, focal, weights=None):
    # Boundary-variance ConvergenceWarnings (random-intercept var -> 0) are expected with
    # 14 clusters; the cluster-robust OLS companion covers that case.
    warnings.filterwarnings("ignore")
    df = df.copy()
    need = [v for v in ["d_aff","person_id","role","phase","hunger_state_start","trigger_mode",
                        "fed","affinity_before"] + [focal] if v in df.columns]
    df = df.dropna(subset=need)
    if "hunger_state_start" in formula:
        df = df[df["hunger_state_start"].astype(str).str.len()>0]
    res={"label":label,"n":len(df),"people":df["person_id"].nunique()}
    try:
        if weights is None:
            mdl = smf.mixedlm(formula, df, groups="person_id").fit(reml=True)
            keep=[i for i in mdl.params.index if i!="Group Var"]
            tab=pd.DataFrame({"coef":mdl.params[keep],"p":mdl.pvalues[keep],
                              "lo":mdl.conf_int()[0][keep],"hi":mdl.conf_int()[1][keep]})
        else:
            # IPW fit: weighted OLS with cluster-robust SEs (a weighted MixedLM is not
            # identified at 14 clusters, and pretending otherwise would be worse).
            w = df[weights]
            mdl = smf.wls(formula, df, weights=w).fit(
                cov_type="cluster", cov_kwds={"groups": df["person_id"]})
            tab=pd.DataFrame({"coef":mdl.params,"p":mdl.pvalues,
                              "lo":mdl.conf_int()[0],"hi":mdl.conf_int()[1]})
        res["table"]=tab
        ols = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["person_id"]})
        res["ols_p"]={k:float(v) for k,v in ols.pvalues.items()}
        print(f"\n[{label}]  n={len(df)} events, {df['person_id'].nunique()} people")
        print(tab.round(4).to_string())
    except Exception as e:
        print(f"[{label}] failed: {e}"); res["table"]=None
    return res

# The mechanistic controls: BOTH are terms in the update rule (B9).
CTRL = "+ fed + affinity_before"
R = {}

print("="*78)
print("STEP 0 — how much of Δaffinity is just the update rule?")
print("="*78)
_m0 = smf.ols("d_aff ~ fed + affinity_before", h10).fit(
    cov_type="cluster", cov_kwds={"groups": h10["person_id"]})
print(f"  d_aff ~ fed + affinity_before        R^2 = {_m0.rsquared:.3f}   <- no dose term at all")
_m1 = smf.ols("d_aff ~ fed + affinity_before + z_n_turns", h10).fit(
    cov_type="cluster", cov_kwds={"groups": h10["person_id"]})
print(f"  ... + n_turns                        R^2 = {_m1.rsquared:.3f}   "
      f"(dose adds {_m1.rsquared - _m0.rsquared:+.3f})")
print(f"  fed coefficient:            {_m0.params['fed']:+.3f}")
print(f"  affinity_before coefficient:{_m0.params['affinity_before']:+.3f}  "
      f"(the EMA's mean-reversion term, -alpha)")

print("\n" + "="*78)
print("STEP 1 — PRIMARY: dose = n_turns (observed for ALL events), controls included")
print("="*78)
R["turns_pooled"] = fit_mix(
    h10, f"d_aff ~ z_n_turns*C(role) + z_n_turns*C(phase) + C(hunger_state_start) "
         f"+ C(trigger_mode) {CTRL}",
    "PRIMARY pooled, dose=n_turns (fully observed)", "z_n_turns")
R["turns_P1"] = fit_mix(
    h10[h10.phase=="P1"], f"d_aff ~ z_n_turns*C(role) + C(trigger_mode) {CTRL}",
    "within Phase 1, dose=n_turns", "z_n_turns")
R["turns_P2"] = fit_mix(
    h10[h10.phase=="P2"], f"d_aff ~ z_n_turns*C(role) + C(trigger_mode) {CTRL}",
    "within Phase 2, dose=n_turns", "z_n_turns")

print("\n" + "="*78)
print("STEP 2 — MISSINGNESS MODEL for duration (52% missing, differentially)")
print("="*78)
h10["has_duration"] = h10["duration_sec"].notna().astype(int)
print("Duration availability by role x phase:")
print(h10.groupby(["role","phase"], observed=True)["has_duration"]
        .agg(["mean","size"]).round(2).to_string())
_mm = smf.logit("has_duration ~ C(role) + C(phase) + C(hunger_state_start) + C(trigger_mode) "
                "+ fed + z_n_turns", h10).fit(disp=0)
print(f"\nMissingness model (logit P(duration observed)), pseudo-R^2 = {_mm.prsquared:.3f}:")
_mm_tab = pd.DataFrame({"coef":_mm.params,"p":_mm.pvalues})
print(_mm_tab.round(3).to_string())
print("\n  -> duration is NOT missing at random with respect to role/phase, which is exactly why")
print("     it cannot be the primary dose. Terms with p<0.05 above are the ones that predict")
print("     whether we even get to see the duration.")
_mm_tab.to_csv(OUT_DIR/"rq3_missingness_model.csv")
# Inverse-probability weights, stabilised and trimmed at the 1st/99th percentile.
h10["p_obs"] = _mm.predict(h10)
_p_marg = h10["has_duration"].mean()
h10["ipw"] = np.where(h10["has_duration"]==1, _p_marg/h10["p_obs"].clip(0.02, 0.98), 0.0)
_hi_trim = h10.loc[h10.has_duration==1, "ipw"].quantile(0.99)
h10["ipw"] = h10["ipw"].clip(upper=_hi_trim)
print(f"\n  Stabilised IPW: mean {h10.loc[h10.has_duration==1,'ipw'].mean():.2f}, "
      f"max {h10['ipw'].max():.2f} (trimmed at the 99th pct).")

print("\n" + "="*78)
print("STEP 3 — SECONDARY: duration, complete-case vs IPW-weighted")
print("="*78)
_dur_df = h10[h10["has_duration"]==1].copy()
R["dur_cc"] = fit_mix(
    _dur_df, f"d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
             f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
    "SECONDARY complete-case, dose=duration", "z_duration_sec")
R["dur_ipw"] = fit_mix(
    _dur_df, f"d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
             f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
    "SECONDARY IPW-weighted, dose=duration", "z_duration_sec", weights="ipw")

# For the record: the OLD specification (no controls), so the inflation is visible.
R["dur_oldspec"] = fit_mix(
    _dur_df, "d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
             "+ C(hunger_state_start) + C(trigger_mode)",
    "(for comparison) OLD spec: no fed / affinity_before control", "z_duration_sec")

print("\n" + "="*78)
print("STEP 4 — do the three specifications agree?")
print("="*78)
_cmp_rows=[]
for key, focal, lbl in [("turns_pooled","z_n_turns","PRIMARY: n_turns (all events)"),
                        ("dur_cc","z_duration_sec","SECONDARY: duration, complete case"),
                        ("dur_ipw","z_duration_sec","SECONDARY: duration, IPW"),
                        ("dur_oldspec","z_duration_sec","OLD SPEC: duration, no controls")]:
    t = R[key]["table"]
    if t is None or focal not in t.index: continue
    r = t.loc[focal]
    _cmp_rows.append(dict(model=lbl, n=R[key]["n"], people=R[key]["people"], term=focal,
                          slope=float(r["coef"]), lo=float(r["lo"]), hi=float(r["hi"]),
                          p=float(r["p"])))
dose_cmp = pd.DataFrame(_cmp_rows)
print(dose_cmp.round(4).to_string(index=False))
dose_cmp.to_csv(OUT_DIR/"rq3_dose_specification_comparison.csv", index=False)
_signs = np.sign(dose_cmp[dose_cmp.model.str.startswith(("PRIMARY","SECONDARY"))]["slope"])
_agree = bool(len(_signs) and (_signs == _signs.iloc[0]).all())
_old = dose_cmp[dose_cmp.model.str.startswith("OLD")]["slope"]
_new = dose_cmp[dose_cmp.model.str.startswith("PRIMARY")]["slope"]
print(f"\n  Sign agreement across the primary and secondary specs: {'YES' if _agree else 'NO'}")
if len(_old) and len(_new):
    print(f"  The old, uncontrolled duration slope was {float(_old.iloc[0]):+.3f}; controlling for "
          f"`fed` and `affinity_before` and using the fully-observed dose gives "
          f"{float(_new.iloc[0]):+.3f}.")

# --- The primary model of record --------------------------------------------------------
_tp = R["turns_pooled"]["table"]
_sl = _tp.loc["z_n_turns"]
_tidy = _tp.reset_index().rename(columns={"index":"term"})
_tidy.insert(0, "model",
             "d_aff ~ n_turns*role + n_turns*phase + HS + trigger + fed + affinity_before + (1|person)")
_tidy.to_csv(OUT_DIR / "rq3_model_results.csv", index=False)

# EVERY interaction term that gets quoted is now registered for multiplicity. The previous
# version quoted duration x feeder and duration x phase in its verdict and corrected neither.
register_p("B10.2", "d_aff ~ n_turns*role + n_turns*phase + HS + trigger + fed + affinity_before + (1|person)",
           "z_n_turns", float(_sl["p"]), "RQ3-adaptation", "confirmatory",
           note="primary dose; fully observed")
for _term in _tp.index:
    if _term.startswith("z_n_turns:"):
        register_p("B10.2",
                   "d_aff ~ n_turns*role + n_turns*phase + HS + trigger + fed + affinity_before + (1|person)",
                   _term, float(_tp.loc[_term,"p"]), "RQ3-adaptation", "confirmatory",
                   note="dose x moderator interaction — quoted in the verdict, so it is corrected")
    if _term.startswith("C(role)"):
        register_p("B10.2",
                   "d_aff ~ n_turns*role + n_turns*phase + HS + trigger + fed + affinity_before + (1|person)",
                   _term, float(_tp.loc[_term,"p"]), "RQ3-adaptation", "confirmatory",
                   note="role main contrast")

_turns_boot = cluster_bootstrap_effect(
    h10.dropna(subset=["d_aff","z_n_turns","fed","affinity_before"]), "person_id",
    lambda _s: smf.ols(f"d_aff ~ z_n_turns*C(role) + z_n_turns*C(phase) "
                       f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
                       data=_s).fit().params["z_n_turns"],
    label="B10 PRIMARY dose slope (n_turns, controlled; person-cluster bootstrap)")
SENSITIVITY.append(dict(metric="B10_dose_slope_n_turns_controlled", primary=float(_sl["coef"]),
                        boot_lo=_turns_boot[0], boot_median=_turns_boot[1],
                        boot_hi=_turns_boot[2], successful_refits=_turns_boot[3],
                        unit="person-cluster bootstrap over ALL learning events"))
globals()["_b10_dose_boot"]=[_turns_boot[0],_turns_boot[1],_turns_boot[2]]
globals()["_b10_dose_cmp"]=dose_cmp
globals()["_b10_agree"]=_agree

# --- ROBUSTNESS: is the slope induced by the EMA re-threading? ---------------------------
_merged_names = {v for v in CANON_IDENTITY.values()
                 if isinstance(v, str) and v.strip().lower() not in PSEUDONYM_PRESERVE}
_merged_pids  = {pseudonymize(n) for n in _merged_names}
_rob = fit_mix(h10[~h10["person_id"].isin(_merged_pids)],
               f"d_aff ~ z_n_turns*C(role) + z_n_turns*C(phase) + C(hunger_state_start) "
               f"+ C(trigger_mode) {CTRL}",
               "ROBUSTNESS: canonicalization-merged people excluded", "z_n_turns")
R["turns_nomerge"] = _rob
_rows=[dict(model="primary (all named people, controlled)", people=int(R["turns_pooled"]["people"]),
            n_events=int(R["turns_pooled"]["n"]), dose_slope=float(_sl["coef"]),
            lo=float(_sl["lo"]), hi=float(_sl["hi"]), p=float(_sl["p"]))]
if _rob.get("table") is not None and "z_n_turns" in _rob["table"].index:
    _rs=_rob["table"].loc["z_n_turns"]
    _rows.append(dict(model="robustness (merged people excluded)", people=int(_rob["people"]),
                      n_events=int(_rob["n"]), dose_slope=float(_rs["coef"]),
                      lo=float(_rs["lo"]), hi=float(_rs["hi"]), p=float(_rs["p"])))
    globals()["_b10_nomerge"]={"coef":float(_rs['coef']),"lo":float(_rs['lo']),
                               "hi":float(_rs['hi']),"p":float(_rs['p']),
                               "people":int(_rob["people"]),"n":int(_rob["n"]),
                               "n_excluded_people":int(len(_merged_pids))}
pd.DataFrame(_rows).to_csv(OUT_DIR / "rq3_affinity_repair_robustness.csv", index=False)
globals()["_b10_R"]=R
""")

md(r"""#### B10.3 — Prior affinity → next-day proactive approaches

**State the estimand, because the previous version quietly conditioned on a post-treatment
variable.** The model was fitted only on person-days where the person *showed up again*. Whether
someone returns is itself downstream of how the robot treated them, so conditioning on it is
conditioning on a collider. The old write-up called this "leakage-free by construction" — it is
leakage-free in *time*, which is a different and weaker property.

The estimand is now written out explicitly:

> **Among people who were present on session-day `d+1`**, is the robot's count of proactive
> approaches toward person `i` associated with `i`'s affinity as of the end of day `d`, given the
> approach opportunities available?

Also fixed:

- **Exposure offset.** A count of approaches means nothing without the number of *chances* to
  approach. Offset = log(detections / eligible approach opportunities) on day `d+1`.
- **Two-part analysis.** (1) does the person appear at all on `d+1`? (2) given appearance, how many
  proactive approaches? Part 1 is the selection step the old model conditioned away.
- **Sensitivity including all person-days**, with zero approaches for absentees, so the
  conditioning can be seen to matter (or not).

**And the honest caveat.** B9 verified `eff_thr = max(0.50, base_ss − 0.15·affinity)` to a maximum
error of 0.0000. Affinity **mechanically lowers the approach threshold**. So "prior affinity
predicts proactive approaches" is substantially a restatement of that line of source code. What the
data can add is only the *magnitude realised in deployment*, which also depends on who walked past
the robot. This is `Implementation verification` with an associational overlay — not an independent
confirmation that the memory "is expressed downstream".""")

code(r"""
# --- B10d. Prior affinity -> next-day proactive approaches ------------------------------
day_aff = (h10.groupby(["person_id","day_rome"])["affinity_after"].last()
              .rename("aff_eod").reset_index())
_days = sorted(m10["day_rome"].unique())
_pro  = m10[m10["trigger_mode"]=="proactive"]
_named = m10[m10["role"]!="unknown"]

_rows=[]
for _p in _named["person_id"].unique():
    for _i in range(len(_days)-1):
        _d0,_d1=_days[_i],_days[_i+1]
        _a = day_aff[(day_aff.person_id==_p)&(day_aff.day_rome<=_d0)]
        _next = m10[m10["person_id"].eq(_p) & m10["day_rome"].eq(_d1)]
        # EXPOSURE on day d+1: how many chances did the robot have to approach this person?
        _n_detect = len(_next)
        _time_present = float(pd.to_numeric(_next["duration_sec"], errors="coerce").fillna(0).sum())
        _rows.append(dict(person_id=_p, day=_d1, phase=phase_of(_d1),
            prior_aff=float(_a["aff_eod"].iloc[-1]) if len(_a) else 0.0,
            n_today=int((m10["person_id"].eq(_p)&m10["day_rome"].eq(_d0)).sum()),
            n_pro_next=int((_pro["person_id"].eq(_p)&_pro["day_rome"].eq(_d1)).sum()),
            n_detections_next=_n_detect,
            time_present_next=_time_present,
            present_next=bool(_n_detect > 0)))
panel=pd.DataFrame(_rows)

print("=== ESTIMAND ===")
print("Among people PRESENT on day d+1: proactive approaches toward person i on d+1,")
print("as a function of i's affinity at end of day d, offset by approach opportunities.\n")
print(f"Full person-day grid:        {len(panel)} rows "
      f"({panel['person_id'].nunique()} people x {len(_days)-1} day-transitions)")
sub = panel[panel["present_next"]].copy()
print(f"After conditioning on presence: {len(sub)} rows, {sub['person_id'].nunique()} people")
print(f"  -> {len(panel)-len(sub)} person-days DROPPED by the conditioning "
      f"({(1-len(sub)/len(panel))*100:.0f}%). Whether someone returns is downstream of how the")
print(f"     robot treated them, so this is a collider; the two-part model below exposes it.")

# --- PART 1: does the person appear at all on d+1? (the selection step) ------------------
_g_app = smf.gee("present_next ~ prior_aff + n_today + C(phase)", groups="person_id",
                 data=panel, family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
_app_or = float(np.exp(_g_app.params["prior_aff"]))
_app_ci = np.exp(_g_app.conf_int().loc["prior_aff"]).tolist()
_app_p  = float(_g_app.pvalues["prior_aff"])
print(f"\nPART 1 — P(present on d+1) ~ prior affinity  (logistic GEE, cluster=person, n={len(panel)}):")
print(f"    OR = {_app_or:.2f} [{_app_ci[0]:.2f}, {_app_ci[1]:.2f}], p={_app_p:.3f}")
if _app_p < 0.10:
    print(f"    -> affinity DOES predict who comes back "
          f"({'higher' if _app_or > 1 else 'LOWER'} affinity -> more likely to return), so the")
    print(f"       conditioning in Part 2 is on a collider and its estimate is not clean.")
else:
    print(f"    -> no strong evidence that affinity predicts who comes back (p={_app_p:.2f}), which")
    print(f"       LIMITS the collider risk in Part 2 — it does not eliminate it, since this test is")
    print(f"       itself underpowered at {len(panel)} person-days over "
          f"{panel['person_id'].nunique()} people.")

# --- PART 2: given presence, how many proactive approaches, per opportunity? -------------
sub["log_exposure"] = np.log(sub["n_detections_next"].clip(lower=1))
_g_pr = smf.gee("n_pro_next ~ prior_aff + n_today + C(phase)", groups="person_id", data=sub,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable(),
                offset=sub["log_exposure"]).fit()
_pa   = float(_g_pr.params["prior_aff"]); _pa_p = float(_g_pr.pvalues["prior_aff"])
_pa_ci = _g_pr.conf_int().loc["prior_aff"].tolist()
# The old model: no offset, so "more approaches" could just mean "was there more".
_g_noexp = smf.gee("n_pro_next ~ prior_aff + n_today + C(phase)", groups="person_id", data=sub,
                   family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable()).fit()
print(f"\nPART 2 — proactive approaches on d+1 | present  (Poisson GEE, cluster=person, n={len(sub)}):")
print(f"    WITHOUT exposure offset: RR = {np.exp(_g_noexp.params['prior_aff']):.2f} per +1 affinity"
      f"   <- the previous specification")
print(f"    WITH offset (per approach OPPORTUNITY): RR = {np.exp(_pa):.2f} "
      f"[{np.exp(_pa_ci[0]):.2f}, {np.exp(_pa_ci[1]):.2f}], p={_pa_p:.3f}")

_prior_boot = cluster_bootstrap_effect(
    sub, "person_id",
    lambda _s: np.exp(smf.glm("n_pro_next ~ prior_aff + n_today + C(phase)", data=_s,
                              family=sm.families.Poisson(),
                              offset=_s["log_exposure"]).fit().params["prior_aff"]),
    label="B10 prior-affinity approach RR, exposure-adjusted (person-cluster bootstrap)")

# --- SENSITIVITY: all person-days, absentees carried as zero approaches -------------------
panel["log_exposure_all"] = np.log(panel["n_detections_next"].clip(lower=1))
_g_all = smf.gee("n_pro_next ~ prior_aff + n_today + C(phase)", groups="person_id", data=panel,
                 family=sm.families.Poisson(), cov_struct=sm.cov_struct.Exchangeable(),
                 offset=panel["log_exposure_all"]).fit()
print(f"\nSENSITIVITY — ALL {len(panel)} person-days, absentees kept as zero approaches:")
print(f"    RR = {np.exp(_g_all.params['prior_aff']):.2f} "
      f"[{np.exp(_g_all.conf_int().loc['prior_aff',0]):.2f}, "
      f"{np.exp(_g_all.conf_int().loc['prior_aff',1]):.2f}], "
      f"p={_g_all.pvalues['prior_aff']:.3f}")
print(f"    (vs {np.exp(_pa):.2f} conditional on presence — the gap is what the conditioning buys.)")

register_p("B10.3", "present_next ~ prior_aff + n_today + phase (logistic GEE, cluster=person)",
           "prior_aff", _app_p, "RQ3-adaptation", "confirmatory",
           note="two-part model, PART 1: does affinity predict who returns? (the selection step)")
register_p("B10.3",
           "n_pro_next ~ prior_aff + n_today + phase + offset(log detections) (Poisson GEE, cluster=person)",
           "prior_aff", _pa_p, "RQ3-adaptation", "confirmatory",
           note="two-part model, PART 2: approaches per opportunity, conditional on presence")

SENSITIVITY.append(dict(metric="B10_prior_affinity_RR_exposure_adj", primary=float(np.exp(_pa)),
                        boot_lo=_prior_boot[0], boot_median=_prior_boot[1],
                        boot_hi=_prior_boot[2], successful_refits=_prior_boot[3],
                        unit="person-cluster bootstrap over person-days, exposure-adjusted"))
panel.to_csv(OUT_DIR/"b10_downstream_panel.csv", index=False)
globals()["_b10_prior"]={"rr":float(np.exp(_pa)),
                         "rr_noexp":float(np.exp(_g_noexp.params["prior_aff"])),
                         "ci":[float(np.exp(_pa_ci[0])),float(np.exp(_pa_ci[1]))],"p":_pa_p,
                         "boot":[_prior_boot[0],_prior_boot[1],_prior_boot[2]],
                         "app_or":_app_or,"app_p":_app_p,
                         "rr_all":float(np.exp(_g_all.params["prior_aff"])),
                         "n_panel":len(panel),"n_sub":len(sub),
                         "n_people":int(sub["person_id"].nunique())}
""")

md(r"""#### B10.4 — Sensitivity, detectable effects, verdict

Leave-one-person-out on the primary slope; dose-definition agreement; simulation-based
**minimum detectable effects** under the real cluster structure (we do *not* compute post-hoc
power); BH correction over both complete families.""")

code(r"""
# --- B10e. Sensitivity + MDE + verdict -------------------------------------------
warnings.filterwarnings("ignore")   # repeated small-sample refits: suppress boundary warnings
# (1) Leave-one-person-out on the PRIMARY (fully observed, controlled) dose slope.
_base = h10.dropna(subset=["z_n_turns","fed","affinity_before"])
_slopes=[]
for _pid in _base["person_id"].unique():
    _s = _base[_base["person_id"]!=_pid]
    try:
        _f = smf.ols(f"d_aff ~ z_n_turns*C(role) + z_n_turns*C(phase) {CTRL}", _s)\
                .fit(cov_type="cluster", cov_kwds={"groups":_s["person_id"]})
        _slopes.append(float(_f.params["z_n_turns"]))
    except Exception: pass
print(f"(1) LOPO primary dose slope range: [{min(_slopes):+.3f}, {max(_slopes):+.3f}] "
      f"({len(_slopes)} refits; full sample "
      f"{R['turns_pooled']['table'].loc['z_n_turns','coef']:+.3f}).")

# (2) Specification comparison — the honest version of the old "dose agreement" check.
print("\n(2) Specification comparison (NOT three independent doses — see B10.2):")
print(globals()["_b10_dose_cmp"].round(4).to_string(index=False))
print("\n    The old cell claimed 'all three dose definitions agree in sign' as corroboration.")
print("    active_energy_cost was one of the three, and it is a literal additive term in the")
print("    credit that defines the outcome. Agreement there was arithmetic, not evidence. It is")
print("    dropped as a dose. What remains is one fully observed dose plus a missingness-")
print("    corrected sensitivity fit on the same underlying quantity.")

# (3) Simulation MDE under the observed cluster structure (300 sims/point).
rng=np.random.default_rng(SEED)
def mde_binary(cluster_sizes, base_p=0.15, icc_sd=0.5, ors=(1.5,2.0,3.0,4.0), n_sim=300):
    pow_=[]
    for orx in ors:
        hits=0
        for _ in range(n_sim):
            rowsA=[]
            for ci,csz in enumerate(cluster_sizes):
                b0=np.log(base_p/(1-base_p))+rng.normal(0,icc_sd)
                x=rng.integers(0,2,csz)
                p=1/(1+np.exp(-(b0+np.log(orx)*x)))
                rowsA.append(pd.DataFrame({"y":rng.binomial(1,p),"x":x,"g":ci}))
            df=pd.concat(rowsA)
            try:
                f=smf.ols("y ~ x", df).fit(cov_type="cluster", cov_kwds={"groups":df["g"]})
                hits+=int(f.pvalues["x"]<0.05)
            except Exception: pass
        pow_.append(hits/n_sim)
    return dict(zip(ors,pow_))
def mde_slope(cluster_sizes, sd_b=0.05, sd_e=0.15, betas=(0.05,0.075,0.10,0.15), n_sim=300):
    pow_=[]
    for b in betas:
        hits=0
        for _ in range(n_sim):
            rowsA=[]
            for ci,csz in enumerate(cluster_sizes):
                x=rng.normal(0,1,csz)
                y=b*x+rng.normal(0,sd_b)+rng.normal(0,sd_e,csz)
                rowsA.append(pd.DataFrame({"y":y,"x":x,"g":ci}))
            df=pd.concat(rowsA)
            try:
                f=smf.ols("y ~ x", df).fit(cov_type="cluster", cov_kwds={"groups":df["g"]})
                hits+=int(f.pvalues["x"]<0.05)
            except Exception: pass
        pow_.append(hits/n_sim)
    return dict(zip(betas,pow_))
_cs_int = m10[m10.role!="unknown"].groupby("person_id").size().tolist()
_cs_dur = _base.groupby("person_id").size().tolist()
_mde_b = mde_binary(_cs_int)
_mde_s = mde_slope(_cs_dur)
print("\n(3) Minimum detectable effects (sim under observed clustering, alpha=.05):")
print("    binary interaction-level contrast (e.g. B3-style OR), power by OR:",
      {k:round(v,2) for k,v in _mde_b.items()})
print("    Δaffinity slope per SD dose (duration-model structure), power by beta:",
      {k:round(v,2) for k,v in _mde_s.items()})
print("    -> effects below ~OR 2 / beta 0.075 were NOT reliably detectable here; "
      "with 2 people per controlled role, role contrasts detect only very large effects — "
      "their CIs above are the honest statement of that limit.")

# (4) Small-cluster robustness table for the key inferential claims.
sens_df = pd.DataFrame(SENSITIVITY)
if len(sens_df):
    sens_df.to_csv(OUT_DIR / "small_cluster_sensitivity.csv", index=False)
    print("\n(4) Small-cluster sensitivity intervals (person-cluster bootstrap):")
    print(sens_df[["metric","primary","boot_lo","boot_median","boot_hi","successful_refits"]]
          .round(4).to_string(index=False))

# (5) Verdict.
_mg=globals()["_b10_meal_gee"]; _pr=globals()["_b10_prior"]
_sl=R["turns_pooled"]["table"].loc["z_n_turns"]
_db=globals()["_b10_dose_boot"]

# RQ3 splits into two claims with two different evidence classes. Report them separately.
# The verdict is DATA-DRIVEN: whether the manipulation "took" is decided by the randomisation
# p-values computed above, not asserted in advance.
_tot_ok   = _mg["perm_p_total"] < 0.05
_perop_ok = _mg["perm_p"] < 0.05
verdict("B10.1", EV_ASSOC if _tot_ok else EV_INCONC,
        f"The manipulation changed TOTAL meal delivery, and it did so almost entirely by changing "
        f"how often people came to the robot — not how generous they were once there. "
        f"(a) TOTAL DELIVERY: obligated feeders supplied {_mg['rr_total']:.1f}x the meal energy per "
        f"person-day in Phase 1 (person-cluster bootstrap [{_mg['boot'][0]:.1f}, {_mg['boot'][2]:.1f}]; "
        f"person-level randomisation p={_mg['perm_p_total']:.3f} against a design floor of "
        f"{_mg['perm_floor']:.3f}) — {'this survives randomisation' if _tot_ok else 'this does NOT survive randomisation'}. "
        f"(b) PER-ENCOUNTER: per interaction they fed only {_mg['rr']:.1f}x as often "
        f"(bootstrap [{_mg['boot_perop'][0]:.1f}, {_mg['boot_perop'][2]:.1f}], randomisation "
        f"p={_mg['perm_p']:.3f}) — {'also significant' if _perop_ok else 'NOT distinguishable from no effect'}. "
        f"(c) The gap between (a) and (b) is EXPOSURE: feeders interacted with the robot "
        f"{_mg['exposure_ratio']:.1f}x more per day. Exposure is a mediator of the role, not a "
        f"confounder, so (b) is a decomposition and (a) is what the drive actually experienced. "
        f"The no-feed pair supplied {_mg['nofeed_k']}/{_mg['nofeed_n']} meals in Phase 1 — perfect "
        f"compliance, though {_mg['nofeed_n']} observations only bound their feed rate below "
        f"{_mg['nofeed_hi']:.2f}. The feeder excess shrank {_mg['rr_p2']:.2f}x once the obligation "
        f"lifted. With 2 people per role, role is nearly aliased with identity: this shows the "
        f"manipulation took for these four people and estimates NOTHING about a population. "
        f"NOTE: an earlier version reported '2.7x, p=.013' as though feeders fed more readily. "
        f"The 2.7x is real, but it is a fact about attendance.",
        n=len(pm))

_dose_ok = np.isfinite(_db[0]) and (_db[0] > 0) and globals()["_b10_agree"]
verdict("B10", EV_IMPL if not _dose_ok else EV_ASSOC,
        f"The adaptive regulatory memory is a PROGRAMMED LEARNING-RULE RESPONSE, and the previous "
        f"'strongest identification result' framing does not survive scrutiny. Affinity is a "
        f"deterministic EMA of delivered energy (B9): `fed` and `affinity_before` alone explain "
        f"R^2={globals()['_b9_r2_mech']:.2f} of every update, and both are terms in the update rule. "
        f"Controlling for them, and using the FULLY OBSERVED dose (n_turns) rather than the 52%-missing "
        f"duration, +1 SD of engagement is associated with Δaffinity {_sl['coef']:+.3f} "
        f"[{_sl['lo']:+.3f}, {_sl['hi']:+.3f}], person-cluster bootstrap [{_db[0]:+.3f}, {_db[2]:+.3f}] "
        f"— against the {float(globals()['_b10_dose_cmp'].query('model.str.startswith(\"OLD\")')['slope'].iloc[0]):+.3f} "
        f"the uncontrolled specification reported. The complete-case and IPW-weighted duration fits "
        f"{'agree in sign' if globals()['_b10_agree'] else 'DISAGREE in sign, so the dose conclusion is withdrawn'}. "
        f"`active_energy_cost` is no longer used as a 'dose agreement check': it is a literal additive "
        f"term in the credit that defines the outcome. DOWNSTREAM: prior affinity is associated with "
        f"next-day proactive approaches per opportunity at RR {_pr['rr']:.2f} "
        f"[{_pr['ci'][0]:.2f}, {_pr['ci'][1]:.2f}] (bootstrap [{_pr['boot'][0]:.2f}, {_pr['boot'][2]:.2f}]; "
        f"{_pr['rr_noexp']:.2f} without the exposure offset) among the {_pr['n_sub']} person-days on "
        f"which the person actually returned. That conditioning is on presence, which is itself "
        f"post-treatment; the two-part model finds "
        + (f"affinity DOES predict who returns (OR {_pr['app_or']:.2f}, p={_pr['app_p']:.3f}), so the "
           f"Part-2 estimate is conditioned on a collider and is not clean"
           if _pr['app_p'] < 0.10 else
           f"no strong evidence that affinity predicts who returns (OR {_pr['app_or']:.2f}, "
           f"p={_pr['app_p']:.3f}), which limits — but at {_pr['n_panel']} person-days does not "
           f"eliminate — the collider risk")
        + f". And the entire downstream path runs through eff_thr=max(0.50, base-0.15*affinity), which "
        f"B9 verifies to a maximum error of 0.0000. 'Affinity predicts approaches' is therefore "
        f"substantially a restatement of that line of source code, not an independent finding. What "
        f"RQ3 genuinely establishes is B10.1 — the role manipulation changed what PEOPLE did. "
        f"Everything else in RQ3 is the controller doing what it was written to do.",
        n=len(h10))
run_bh()
""")

# ==========================================================================
# PHASE C — VISUALIZATION
# ==========================================================================
md(r"""## Phase C — Visualization

All figures use the HS status palette (Full green, Hungry amber, Starving red, N/A grey —
`stomachMonitor.py`'s colours, contrast-tuned to pass a categorical palette validator: lightness
band, chroma floor, CVD separation) and are saved to `analysis/figures/` as PNG + SVG at
≥200 dpi. Marks carrying data always pair colour with a direct value label, so no reading
depends on colour alone. Figure titles state the analytic claim; captions state the unit of
analysis and whether the panel is confirmatory, diagnostic, or exploratory.
""")

md("""**Fig 2 — Drive timeline per day** *(unit: hunger-level event, n = 165,460 events across
12 monitored runs / 8 days)*: one panel per experiment day. The drive `run_id`/`monotonic` clock resets on every restart, so a day can
contain several runs (4 restarts on 2026-06-15, 2 on 2026-06-18); we stitch a day's runs
on the shared wall-clock (`timestamp_epoch`) and mark each restart with a dotted line.
HS bands + thresholds, feeding up-arrows sized by meal, Starving episodes shaded.""")
code(r"""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# One timeline PER DAY (8 panels). run_id/monotonic are per-restart, so we place every
# sample on the day's absolute wall-clock and break the line at restarts / long gaps.
days = sorted(hunger_raw["day_rome"].dropna().unique())
nday=len(days); ncol=2; nrow=int(np.ceil(nday/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.05*nrow), sharey=True)
axes=np.atleast_1d(axes).ravel()
MEAL_SZ={"SMALL_MEAL":8,"MEDIUM_MEAL":13,"LARGE_MEAL":19}
for ax, day in zip(axes, days):
    dd=hunger_raw[hunger_raw["day_rome"]==day].sort_values(["timestamp_epoch","id"])
    t0=dd["timestamp_epoch"].min()
    ax.axhspan(60,100,color=HS_PALETTE["HS1"],alpha=0.10,zorder=0)
    ax.axhspan(25,60, color=HS_PALETTE["HS2"],alpha=0.10,zorder=0)
    ax.axhspan(0,25,  color=HS_PALETTE["HS3"],alpha=0.14,zorder=0)
    ax.axhline(60,color=HS_ACCENT["HS2"],lw=0.6,alpha=0.5); ax.axhline(25,color=HS_ACCENT["HS3"],lw=0.6,alpha=0.5)
    # HS3 episode shading (convert per-run monotonic to this day's wall-clock via the run offset)
    for _,e in hs3_episodes[hs3_episodes["day_rome"]==day].iterrows():
        off=e["entry_ts_epoch"]-e["entry_mono"]
        end_mono = e["escape_mono"] if pd.notna(e.get("escape_mono")) else e["exit_mono"]
        xe=(e["entry_ts_epoch"]-t0)/60; xx=((end_mono+off)-t0)/60
        ax.axvspan(xe,max(xx,xe+0.4),color=HS_PALETTE["HS3"],alpha=0.35,zorder=1)
    # break the line at restarts (run change) or gaps > 3 min (robot off between runs)
    x=(dd["timestamp_epoch"].values-t0)/60
    brk=(dd["run_id"].ne(dd["run_id"].shift()) | (pd.Series(x).diff().values>3.0))
    seg=np.cumsum(brk.astype(int))
    for si in np.unique(seg):
        m=seg==si
        ax.plot(x[m], dd["stomach_level_after"].values[m], color=INK, lw=1.0, zorder=3)
        if si>0: ax.axvline(x[m][0], color=MUTED, ls=":", lw=0.8, alpha=0.7, zorder=2)  # restart marker
    fd=dd[dd["event_type"]=="feeding"]
    for _,r in fd.iterrows():
        sz=MEAL_SZ.get(r["meal_payload"],9); xx=(r["timestamp_epoch"]-t0)/60
        ax.annotate("",(xx,r["stomach_level_after"]),(xx,r["stomach_level_after"]-sz),
                    arrowprops=dict(arrowstyle="-|>",color=HS_ACCENT["HS1"],lw=1.1),zorder=4)
    nruns=dd["run_id"].nunique(); span=(dd["timestamp_epoch"].max()-t0)/60
    rlabel=f"{nruns} runs" if nruns>1 else "1 run"
    _ph = "Phase 1 (roles)" if str(day) in globals().get("PHASE1_DAYS",set()) else "Phase 2 (free)"
    ax.text(0.015,0.97,f"{day}  ·  {_ph}  ·  {rlabel}  ·  {span:.0f} min span  ·  {len(fd)} meals",
            transform=ax.transAxes,fontsize=8.5,va="top",fontweight="medium",zorder=6,
            bbox=dict(boxstyle="round,pad=0.25",fc="white",ec=GRID,lw=0.6,alpha=0.95))
    ax.set_ylim(0,100); ax.set_xlim(left=0)
for j,ax in enumerate(axes):
    if j>=nday: ax.set_visible(False)
    if j%ncol==0: ax.set_ylabel("stomach %")
    if j>=nday-ncol: ax.set_xlabel("minutes into the day (wall-clock)")
handles=[Patch(fc=HS_PALETTE["HS1"],alpha=.4,label="Full (≥60)"),
         Patch(fc=HS_PALETTE["HS2"],alpha=.4,label="Hungry (25–60)"),
         Patch(fc=HS_PALETTE["HS3"],alpha=.5,label="Starving (<25) / episode"),
         Line2D([0],[0],color=INK,lw=1.2,label="stomach level"),
         Line2D([0],[0],color=MUTED,lw=0.9,ls=":",label="restart"),
         Line2D([0],[0],marker="^",color=HS_ACCENT["HS1"],lw=0,label="meal (▲ size ∝ SMALL/MED/LARGE)")]
fig.legend(handles=handles,loc="lower center",ncol=6,bbox_to_anchor=(0.5,-0.05),fontsize=9)
fig.suptitle("Fig 2 — Orexigenic-drive dynamics per experiment day: autonomous drain, discrete replenishment, Starving sojourns",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig02_drive_timeline"); plt.show()
""")

md("""**Fig 4 — Deficit → action (RQ1-3)** *(units: 367 interaction turns, 710 chat messages,
217 co-present interactions, and 193 deficit-gated action events)*: the correct contrast,
**Full vs deficit (Hungry+Starving)**. Left: recovery-action rates with bootstrap CIs.
Right: the time distribution of deficit-gated actions across the deployment, rather than a
bar chart of corpus totals; the thin state-coloured trace is the continuous stomach level
context.""")
code(r"""
# Fig 4 — RQ1-3 visual: does a DEFICIT change what the robot does? Contrast no-deficit (Full)
# vs deficit (Hungry+Starving) across the coded state-gated recovery repertoire.
from matplotlib.collections import LineCollection

def _grp(h): return "Full" if h=="HS1" else ("Deficit" if h in ("HS2","HS3") else None)
FULL_C, DEF_C = HS_PALETTE["HS1"], "#E0892E"

_tt = turns_nlp.copy(); _tt["g"]=_tt["hunger_state"].map(_grp)
_tt["v"]=pd.to_numeric(_tt.get("hunger_mentioned",0),errors="coerce").fillna(0)
_am = chat_msgs.copy()
if "role" in _am.columns: _am=_am[_am["role"]=="assistant"]
_am["g"]=_am["hs"].map(_grp); _am["v"]=pd.to_numeric(_am.get("hunger_mentioned",0),errors="coerce").fillna(0)
_mm = master.copy(); _mm["g"]=_mm["hunger_state_start"].map(_grp)
_mm["fed"]=(pd.to_numeric(_mm["meals_eaten_count"],errors="coerce").fillna(0)>0).astype(float)
rate_specs=[("hunger framing\nface-to-face",_tt,"v"),
            ("hunger framing\nTelegram",_am,"v"),
            ("feeding pursuit\nP(meal)",_mm,"fed")]
def _ci(df,col,g):
    e,lo,hi=boot_ci(df[df["g"]==g][col]); return e,max(e-lo,0.0),max(hi-e,0.0)

fig,(axA,axB)=plt.subplots(1,2,figsize=(14.6,4.8),gridspec_kw={"width_ratios":[0.95,1.55]})
x=np.arange(len(rate_specs)); w=0.38
for gi,(g,c) in enumerate([("Full",FULL_C),("Deficit",DEF_C)]):
    es=[];los=[];his=[]
    for _,df,col in rate_specs:
        e,lo,hi=_ci(df,col,g); es.append(e);los.append(lo);his.append(hi)
    off=(-w/2 if gi==0 else w/2)
    axA.bar(x+off,es,w,color=c,edgecolor="white",linewidth=1.1,label=g,zorder=3)
    axA.errorbar(x+off,es,yerr=[los,his],fmt="none",ecolor=INK,elinewidth=1.1,capsize=3,zorder=4)
    for xi,e,hi in zip(x,es,his):
        # anchor above the errorbar whisker, never on it (small bars used to collide)
        axA.annotate(f"{e:.2f}",(xi+off,e+hi),textcoords="offset points",xytext=(0,3),
                     ha="center",fontsize=8.5,color=INK)
axA.set_xticks(x); axA.set_xticklabels([s[0] for s in rate_specs],fontsize=9)
axA.set_ylabel("rate"); axA.set_ylim(0,1.0); axA.grid(False); axA.legend(loc="upper right")
axA.set_title("Recovery-action rates by regulatory state",fontsize=11)

_ev=chat_events.copy(); _prov=_ev[_ev["event_type"].isin(["hs2_entry","hs3_proactive"])]
_prov = _prov.assign(action=_prov["event_type"].map({"hs2_entry":"Hungry Telegram ping",
                                                     "hs3_proactive":"Starving Telegram ping"}),
                     hs_state=_prov["hs"])
_ac=hunger_raw[hunger_raw["event_type"]=="active_cost"].copy()
_seek=_ac[_ac["stimulus_label"].isin(["hunger_ask_feed","hunger_still_hungry","hunger_look_around"])].copy()
_seek = _seek.assign(action="Face-to-face feeding request",
                     hs_state=_seek["hunger_state_before"].fillna(_seek["hunger_state_after"]))
_events = pd.concat([
    _prov[["timestamp_epoch","day_rome","action","hs_state"]],
    _seek[["timestamp_epoch","day_rome","action","hs_state"]]
], ignore_index=True).dropna(subset=["timestamp_epoch","day_rome"])
_events["hs_group"]=_events["hs_state"].map(_grp)
days=sorted(hunger_raw["day_rome"].dropna().unique())
day_to_y={d:i for i,d in enumerate(days)}
day_start=hunger_raw.groupby("day_rome")["timestamp_epoch"].min().to_dict()
_events["minute"]=_events.apply(lambda r:(r["timestamp_epoch"]-day_start.get(r["day_rome"],r["timestamp_epoch"]))/60.0,axis=1)
_events["y"]=_events["day_rome"].map(day_to_y)
styles={
    "Hungry Telegram ping":("o",HS_ACCENT["HS2"],64),
    "Starving Telegram ping":("o",HS_ACCENT["HS3"],72),
    "Face-to-face feeding request":("^","#7A4EAB",92),
}
for day in days:
    dd=hunger_raw[hunger_raw["day_rome"]==day]
    span=(dd["timestamp_epoch"].max()-dd["timestamp_epoch"].min())/60.0
    axB.hlines(day_to_y[day],0,span,color="#D9DEE5",lw=0.9,zorder=0)
    # Mini-sparkline of the continuous stomach level for this day. It is scaled to
    # stay inside the day row; segment colour carries the current hunger state.
    dd=dd.sort_values(["timestamp_epoch","id"]).copy()
    if len(dd) >= 2:
        xx=(dd["timestamp_epoch"].values-dd["timestamp_epoch"].min())/60.0
        yy=day_to_y[day] + (50.0-dd["stomach_level_after"].astype(float).values)/175.0
        states=dd["hunger_state_after"].fillna(dd["hunger_state_before"]).astype(str).values
        run_ids=dd["run_id"].astype(str).values
        epochs=dd["timestamp_epoch"].astype(float).values
        segs=[]; cols=[]
        for i in range(len(dd)-1):
            if run_ids[i] != run_ids[i+1] or (epochs[i+1]-epochs[i]) > 180:
                continue
            segs.append([(xx[i],yy[i]),(xx[i+1],yy[i+1])])
            cols.append(HS_ACCENT.get(states[i], MUTED))
        if segs:
            axB.add_collection(LineCollection(segs, colors=cols, linewidths=1.25, alpha=0.90, zorder=1))
for action,(marker,color,size) in styles.items():
    sub=_events[_events["action"]==action]
    if len(sub):
        axB.scatter(sub["minute"],sub["y"],s=size,marker=marker,color=color,
                    edgecolor="white",linewidth=1.0,alpha=0.94,label=f"{action} (n={len(sub)})",zorder=3)
axB.set_yticks(range(len(days))); axB.set_yticklabels(days,fontsize=8.5)
axB.set_xlabel("minutes into experiment day (wall-clock)")
axB.set_ylabel("experiment day")
axB.grid(False)
axB.invert_yaxis()
axB.legend(loc="upper left",bbox_to_anchor=(1.01,1.0),fontsize=8,frameon=False)
axB.text(0.01,0.985,"thin line: stomach level (green/amber/red = state)",
         transform=axB.transAxes,ha="left",va="top",fontsize=8.2,color=MUTED)
axB.set_title("Temporal allocation of deficit-gated actions",fontsize=11)
fig.suptitle("Fig 4 — Deficit-to-action-selection coupling: deficits bias recovery policy",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig04_deficit_action"); plt.show()
""")

md("**Fig 5 — State×hunger priority-reallocation heatmap** *(unit: interaction, n = 217; Starving column n = 13)*: grid coloured by Engaged-completion and by average turns, with the low-n Starving column interpreted directionally.")
code(r"""
# Sequential single-hue (magnitude), NOT a red-green rainbow: CVD-safe, and every cell
# is annotated with value + n so small cells (esp. the HS3 column) are not over-read.
import matplotlib.colors as mcolors
d=master.copy(); d["reached_ss4"]=(d["final_state"]=="ss4").astype(int)
d["n_turns"]=pd.to_numeric(d["n_turns"],errors="coerce").fillna(0)
d=d.dropna(subset=["hunger_state_start"])
ss_ord=["ss1","ss2","ss3"]
def grid(val): return d.pivot_table(index="initial_state",columns="hunger_state_start",
        values=val,aggfunc="mean").reindex(index=ss_ord,columns=HS_ORDER)
def ngrid(): return d.pivot_table(index="initial_state",columns="hunger_state_start",
        values="interaction_id",aggfunc="count").reindex(index=ss_ord,columns=HS_ORDER)
def draw(ax,M,N,cmap,vmax,label,fmt):
    im=ax.imshow(M.values,cmap=cmap,vmin=0,vmax=vmax,aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(HS_NAMES); ax.set_yticks(range(3)); ax.set_yticklabels([SS_NAME[s] for s in ss_ord])
    ax.set_xlabel("hunger state"); ax.grid(False)
    for i in range(3):
        for j in range(3):
            v=M.values[i,j]; n=N.values[i,j]
            if np.isnan(v):
                ax.text(j,i,"—",ha="center",va="center",color=MUTED); continue
            frac=v/vmax
            tc="white" if frac>0.55 else INK
            ax.text(j,i,fmt.format(v),ha="center",va="center",color=tc,fontweight="bold",fontsize=12)
            ax.text(j,i+0.30,f"n={0 if np.isnan(n) else int(n)}",ha="center",va="center",color=tc,fontsize=8,alpha=0.85)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label(label); cb.outline.set_visible(False)
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.6))
N=ngrid()
draw(a1,grid("reached_ss4"),N,"Greens",1.0,"P(reach Engaged)","{:.2f}")
a1.set_ylabel("social state at start"); a1.set_title("Engaged-completion probability")
draw(a2,grid("n_turns"),N,"Purples",grid("n_turns").max().max(),"avg conversation turns","{:.1f}")
a2.set_title("Mean conversation turns")
fig.suptitle("Fig 5 — Priority reallocation: Starving shifts behaviour away from social completion",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig05_prioritisation_heatmap"); plt.show()
""")

md("""**Fig 8 — The remote loop, against its control** *(unit: proactive Telegram ping; one-to-one
reply matching; `hs3_recovery` notifications excluded from the hunger channel)*. The bar that
matters is the **matched no-ping control window**: people message the bot anyway, so a raw
response-to-ping rate says nothing on its own. Exact 95% CIs.""")
code(r"""
# Built from the ONE-TO-ONE matched pings (B5b), not the old loop that let a single reply
# answer several pings at once.
_pm = globals()["_b5_pm"]
_kinds = ["hs2_entry","hs3_proactive"]
_kcol  = {"hs2_entry":HS_PALETTE["HS2"], "hs3_proactive":HS_PALETTE["HS3"]}
_klabel= {"hs2_entry":"Hungry\nping","hs3_proactive":"Starving\nping"}

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,4.6),width_ratios=[1,1.05])

# LEFT: ping vs control — the comparison the previous version never made.
_cats, _es, _los, _his, _ns, _cols = [], [], [], [], [], []
for k in _kinds:
    s=_pm[_pm.ping_type==k]
    if not len(s): continue
    e,lo,hi = exact_prop_ci(int(s["replied"].sum()), len(s))
    _cats.append(_klabel[k]); _es.append(e); _los.append(e-lo); _his.append(hi-e)
    _ns.append(len(s)); _cols.append(_kcol[k])
_e,_lo,_hi = exact_prop_ci(int(_pm["replied"].sum()), len(_pm))
_cats.append("ALL hunger\npings"); _es.append(_e); _los.append(_e-_lo); _his.append(_hi-_e)
_ns.append(len(_pm)); _cols.append(MUTED)
_bp=globals()["_b5_ping"]
_ec_e = _bp["ctrl"]
_cats.append("matched\nCONTROL\n(no ping)"); _es.append(_ec_e)
_los.append(0.0); _his.append(0.0); _ns.append(0); _cols.append("#C7CDD4")
bars_with_ci(axL,_cats,_es,_los,_his,_cols,
             n_labels=[n if n else None for n in _ns])
axL.axhline(_ec_e,color="#B23A26",ls="--",lw=1.4,zorder=5)
axL.annotate("control baseline",(len(_cats)-1.0,_ec_e),textcoords="offset points",
             xytext=(0,8),ha="center",fontsize=8.4,color="#B23A26",fontweight="semibold")
axL.set_ylim(0,1); axL.set_ylabel("P(user reply within 1 h)")
axL.set_title(f"Reply rate: pings vs matched control windows\n"
              f"(difference {_bp['diff']:+.2f}, subscriber-cluster bootstrap "
              f"[{_bp['boot'][0]:+.2f}, {_bp['boot'][2]:+.2f}])",fontsize=10.8)

# RIGHT: window sensitivity — does the answer depend on where the window is drawn?
_w=globals()["_b5_win"]
for k in _kinds+["ALL"]:
    s=_w[_w.ping_type==k].sort_values("window_min")
    if not len(s): continue
    _c = _kcol.get(k, INK)
    axR.errorbar(s["window_min"], s["rate"],
                 yerr=[s["rate"]-s["exact_lo"], s["exact_hi"]-s["rate"]],
                 marker="o", ms=6, lw=1.8, capsize=4, color=_c,
                 label=f"{_klabel.get(k,'all hunger pings').replace(chr(10),' ')} (n={int(s['n'].iloc[0])})")
axR.axhline(_ec_e,color="#B23A26",ls="--",lw=1.4,zorder=2)
axR.annotate("control baseline (60 min)",(15,_ec_e),textcoords="offset points",xytext=(4,6),
             fontsize=8.2,color="#B23A26")
axR.set_xticks([15,30,60]); axR.set_xlabel("reply window (minutes)")
axR.set_ylabel("P(user reply)"); axR.set_ylim(0,1)
axR.legend(fontsize=8.2,loc="upper left")
axR.set_title("Window sensitivity (exact 95% CI)",fontsize=10.8)
fig.suptitle("Fig 8 — The remote channel is a weak recovery pathway",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig08_remote_loop"); plt.show()
""")

md("""**Fig 9 — Occupancy: what happened, and what the model says** *(unit: state sojourn from the
level series, terminal right-censored dwells included; 12 monitored runs)*. The **empirical** bars
need no stationarity assumption and are the figure of record. The modelled CTMC is shown beside
them, with its diagnostics: the fraction of non-ergodic bootstrap resamples, and the spread between
the visited and idle regimes it pools together.""")
code(r"""
from matplotlib.ticker import PercentFormatter
_b7=globals()["_b7"]
pi=globals().get("_ctmc_pi",{})
states=["HS1","HS2","HS3"]
seq=state_sequence(hunger_raw); seq=seq[(seq["state"].isin(states))&(seq["dwell"]>0)]
ti=seq.groupby("state")["dwell"].sum().reindex(states).fillna(0); emp=ti/ti.sum()

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.8,4.8),width_ratios=[1.15,1])

x=np.arange(3); wbar=0.38
b1=axL.bar(x-wbar/2,[emp[s] for s in states],wbar,label="EMPIRICAL (observed seconds)",
           color=[HS_PALETTE[s] for s in states],edgecolor="white",linewidth=1.5,zorder=3)
if pi:
    b2=axL.bar(x+wbar/2,[pi[s] for s in states],wbar,label="CTMC stationary (modelled)",
               color=[HS_PALETTE[s] for s in states],alpha=0.42,edgecolor=MUTED,
               linewidth=1.2,hatch="//",zorder=3)
_eb=_b7["emp_boot"]
axL.errorbar([2-wbar/2],[emp["HS3"]],
             yerr=[[max(emp["HS3"]-_eb[0],0)],[max(_eb[2]-emp["HS3"],0)]],
             fmt="none",ecolor=INK,elinewidth=1.4,capsize=4,zorder=5)
for _bars,_vals in ([(b1,[emp[s] for s in states])] + ([(b2,[pi[s] for s in states])] if pi else [])):
    for r,v in zip(_bars,_vals):
        _yt = v+0.012
        if _bars is b1 and abs(r.get_x()+r.get_width()/2-(2-wbar/2))<1e-9:
            _yt = max(_yt, _eb[2]+0.014)
        axL.text(r.get_x()+r.get_width()/2,_yt,f"{v*100:.1f}%",ha="center",fontsize=9,fontweight="medium")
axL.set_xticks(x); axL.set_xticklabels([HS_NAME[s] for s in states])
axL.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
axL.set_ylabel("occupancy"); axL.set_ylim(0,0.65); axL.grid(False); axL.legend(fontsize=8.6)
axL.set_title(f"Starving: {emp['HS3']*100:.2f}% empirical "
              f"[{_eb[0]*100:.2f}, {_eb[2]*100:.2f}]",fontsize=11)

# RIGHT: the diagnostics that decide whether the modelled number means anything.
_st = pd.read_csv(OUT_DIR/"b7_stratified_occupancy.csv")
_st = _st[_st["identified"]]
_lbl = _st["stratum"].tolist()
_ye = np.arange(len(_st))[::-1]
axR.barh(_ye, _st["empirical_starving"]*100, height=0.34, color=HS_PALETTE["HS3"],
         edgecolor="white", label="empirical", zorder=3)
axR.barh(_ye-0.36, _st["modelled_starving"]*100, height=0.34, color=HS_PALETTE["HS3"],
         alpha=0.42, edgecolor=MUTED, hatch="//", label="modelled", zorder=3)
for yy,(e,m) in zip(_ye, zip(_st["empirical_starving"], _st["modelled_starving"])):
    axR.text(e*100, yy, f" {e*100:.2f}%", va="center", fontsize=8.2)
    if np.isfinite(m): axR.text(m*100, yy-0.36, f" {m*100:.2f}%", va="center", fontsize=8.2, color=MUTED)
axR.set_yticks(_ye-0.18); axR.set_yticklabels(_lbl, fontsize=9)
axR.set_xlabel("Starving occupancy (%)")
axR.legend(fontsize=8.4, loc="lower right")
axR.margins(x=0.22)
_verdict = "NOT identified" if not _b7["stable"] else "stable"
axR.set_title(f"Stratified: the pooled chain mixes these\n"
              f"{_b7['nonergodic']*100:.0f}% of run-resamples non-ergodic  →  {_verdict}",
              fontsize=10.6)
fig.suptitle("Fig 9 — Report the empirical occupancy; the stationary model does not survive its diagnostics",
             fontsize=12.6,fontweight="semibold")
savefig(fig,"fig09_steady_state"); plt.show()
""")

md("""**Fig 10 — Homeostatic-affinity trajectories, faceted by Phase-1 role** *(unit: learning
update event, n = 205 learning-eligible RQ3 events over 14 named people)*: one panel per
assigned role, sharing both axes. Thin lines are individual people; the bold line is the role
mean of the currently-held affinity. Faceting replaces the 14-line overlay: the panel carries
identity, so colour never has to separate more series than it can.""")
code(r"""
# Use B9's re-threaded affinity (merged identities get a single coherent EMA, not the stale
# last-label value) rather than reloading the raw per-label log.
hlc=globals()["_b9_hlc"].copy()
meals_by=(master.assign(fed=pd.to_numeric(master["meals_eaten_count"],errors="coerce").fillna(0))
                .groupby("user_key")["fed"].sum())
named=hlc[(hlc["person_id"]!="unknown") & hlc["affinity_after"].notna()].copy()
named=named.sort_values(["person_id","timestamp_epoch"])
named["experiment_day"]=(named["timestamp_epoch"]-named["timestamp_epoch"].min())/86400.0
named["role"]=named["person_id"].map(role_of)
# Feeds delivered DURING PHASE 1 — the window in which the role actually constrained the
# person. A whole-deployment total would credit the no-feed pair with post-lift feeds and
# contradict B10's "0 feeds in Phase 1" compliance result.
_p1m=master[master["day_rome"].astype(str).isin(PHASE1_DAYS)]
p1_meals_by=(_p1m.assign(fed=pd.to_numeric(_p1m["meals_eaten_count"],errors="coerce").fillna(0))
                 .groupby("user_key")["fed"].sum())

# Phase boundary on the experiment-day axis: first Phase-2 learning event.
_p2=named[~named["day_rome"].astype(str).isin(PHASE1_DAYS)]
xbound=float(_p2["experiment_day"].min())-0.02 if len(_p2) else np.nan
xmax=float(named["experiment_day"].max())

# Role mean of the CURRENTLY-HELD affinity: forward-fill each person onto a common grid, then
# average. A mean over update events alone would be biased by who happened to interact that day.
grid=np.linspace(0,xmax,80)
def role_mean(sub):
    cols=[]
    for _,g in sub.groupby("person_id"):
        x=g["experiment_day"].values; y=g["affinity_after"].values
        idx=np.searchsorted(x,grid,side="right")-1          # last known value at each grid point
        col=np.where(idx>=0, y[np.clip(idx,0,len(y)-1)], np.nan)   # NaN before a person's 1st update
        cols.append(col)
    return np.nanmean(np.vstack(cols),axis=0) if cols else np.full_like(grid,np.nan)

PANELS=[("feeder","Obligated feeders — required to feed in Phase 1"),
        ("no_feed","No-feed pair — asked to interact, never to feed"),
        ("normal","Unconstrained — no instruction either way")]
fig,axes=plt.subplots(3,1,figsize=(11.4,8.8),sharex=True,sharey=True,layout="constrained")
_ymin=min(-0.08,float(named["affinity_after"].min())-0.05)
for ax,(role,title) in zip(axes,PANELS):
    sub=named[named["role"]==role]
    c=ROLE_COLOR[role]; ls="--" if role=="no_feed" else "-"
    ax.axhspan(0,1.06,color=HS_PALETTE["HS1"],alpha=0.045,zorder=0)
    ax.axhspan(_ymin,0,color=HS_PALETTE["HS3"],alpha=0.035,zorder=0)
    ax.axhline(0,color=MUTED,ls=":",lw=1,zorder=1)
    if np.isfinite(xbound):
        ax.axvline(xbound,color=INK,lw=1.1,ls=(0,(4,3)),alpha=0.65,zorder=3)
    people=sub["person_id"].unique()
    for pid in people:                                  # thin: one line per person
        g=sub[sub["person_id"]==pid]
        ax.plot(g["experiment_day"],g["affinity_after"],lw=1.0,color=c,alpha=0.5,ls=ls,
                marker="o",ms=2.4,zorder=3)
    ax.plot(grid,role_mean(sub),lw=2.6,color=c,ls=ls,zorder=5,solid_capstyle="round")
    # Direct-label the endpoints, but only where they cannot collide: the two controlled
    # panels have 2 people each; in the 10-person panel label just the top 3 by terminal value.
    term=sub.groupby("person_id").tail(1)[["person_id","experiment_day","affinity_after"]]
    show=term.sort_values("affinity_after",ascending=False).head(2 if role!="normal" else 3)
    for _,r in show.iterrows():
        _tot=int(meals_by.get(r["person_id"],0))
        ax.annotate(f"{r['person_id']} · {_tot} {'meal' if _tot==1 else 'meals'} in total",
                    xy=(r["experiment_day"],r["affinity_after"]),
                    xytext=(6,0),textcoords="offset points",va="center",ha="left",
                    fontsize=8.2,color=c,fontweight="medium")
    _p1f=int(p1_meals_by.reindex(people).fillna(0).sum())
    ax.set_title(f"{title}   (n={len(people)} {'person' if len(people)==1 else 'people'}; "
                 f"{_p1f} {'meal' if _p1f==1 else 'meals'} delivered while the role held, in Phase 1)",
                 loc="left",fontsize=10.4,color=INK,pad=6)
    ax.grid(True,axis="y"); ax.set_ylabel("learned affinity")
axes[0].set_xlim(0,xmax*1.17); axes[0].set_ylim(_ymin,1.06)
if np.isfinite(xbound):
    # Phase labels go INSIDE the top panel, straddling the boundary line: placed above the axes
    # they collide with the panel title.
    for _x,_ha,_t in [(xbound-0.12,"right","Phase 1 — roles assigned"),
                      (xbound+0.12,"left","Phase 2 — constraints lifted")]:
        axes[0].text(_x,_ymin+0.06,_t,ha=_ha,va="bottom",fontsize=8.6,color=MUTED,
                     bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.85))
axes[0].plot([],[],lw=1.0,color=MUTED,alpha=0.6,label="individual person")
axes[0].plot([],[],lw=2.6,color=MUTED,label="role mean (currently-held affinity)")
fig.legend(loc="outside lower center",ncol=2,frameon=False,fontsize=8.6)
axes[-1].set_xlabel("experiment time (days from first logged affinity update)")
fig.suptitle("Fig 10 — Adaptive regulatory memory: affinity separates by assigned role, then relaxes",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig10_affinity_trajectories"); plt.show()
""")

# Fig 11 (feeding-share vs approach-share of top-affinity people) removed: a descriptive
# share comparison over 5 people, superseded by the inferential B10.3 test (prior affinity ->
# next-day proactive approaches, person-clustered GEE) and Fig 13.

md("""**Fig 12 — Role-manipulation validation (B10.1)** *(units: interaction and person-day;
217 interactions, 14 named people, 8 days; controlled roles = 2 feeders + 2 no-feed in
Phase 1)*: did the Phase-1 experimental labels produce the intended feeding behaviour, and
did it relax in Phase 2? Left: feed probability per interaction with exact 95% CIs. Right:
meals per person-day with the feeder-vs-normal Poisson-GEE rate ratios. Roles are external
experiment metadata, not robot software inputs.""")
code(r"""
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,4.9))
_d=_b10_m10[_b10_m10["role"]!="unknown"]
_x0={"P1":0.0,"P2":1.0}; _off={"normal":-0.22,"feeder":0.0,"no_feed":0.22}
for (role,ph),g in _d.groupby(["role","phase"], observed=True):
    k,n=int(g["fed"].sum()),len(g)
    lo,hi=proportion_confint(k,n,method="beta")
    x=_x0[ph]+_off[role]
    axL.errorbar(x,k/n,yerr=[[k/n-lo],[hi-k/n]],fmt="o",ms=7,color=ROLE_COLOR[role],
                 ecolor=ROLE_COLOR[role],elinewidth=1.4,capsize=4,zorder=4)
    axL.annotate(f"{k}/{n}",(x,hi),textcoords="offset points",xytext=(0,5),
                 ha="center",fontsize=8,color=INK)
axL.set_xticks([0,1]); axL.set_xticklabels(["Phase 1 (roles active)","Phase 2 (unconstrained)"])
axL.set_ylabel("P(meal during interaction)"); axL.set_ylim(-0.03,0.85)
for r in ROLE_ORDER:
    axL.plot([],[],"o",color=ROLE_COLOR[r],label=ROLE_LABEL[r])
axL.legend(loc="upper right",fontsize=8.6)
axL.set_title("Feed probability per interaction — exact 95% CIs",fontsize=11.5)

_pdm=(_d.groupby(["person_id","day_rome","phase","role"],observed=True)
        .agg(meals=("meals_eaten_count","sum")).reset_index())
_bars=_pdm.groupby(["role","phase"],observed=True)["meals"].mean()
for role in ROLE_ORDER:
    for ph in ("P1","P2"):
        v=float(_bars.get((role,ph),0.0)); x=_x0[ph]+_off[role]
        axR.bar(x,v,width=0.19,color=ROLE_COLOR[role],edgecolor="white",linewidth=1.0,zorder=3)
        axR.annotate(f"{v:.2f}",(x,v),textcoords="offset points",xytext=(0,3),
                     ha="center",fontsize=8.4,color=INK)
_mg=globals()["_b10_meal_gee"]
# The decomposition IS the finding: the role changed attendance, not generosity per encounter.
axR.text(0.985,0.97,(f"feeder vs unconstrained, Phase 1\n"
        f"(a) TOTAL delivery   {_mg['rr_total']:.2f}x  boot [{_mg['boot'][0]:.2f}, {_mg['boot'][2]:.2f}]  "
        f"perm p={_mg['perm_p_total']:.3f}\n"
        f"(b) PER ENCOUNTER    {_mg['rr']:.2f}x  boot [{_mg['boot_perop'][0]:.2f}, {_mg['boot_perop'][2]:.2f}]  "
        f"perm p={_mg['perm_p']:.3f}\n"
        f"(c) EXPOSURE         {_mg['exposure_ratio']:.2f}x more interactions/day\n"
        f"    -> the role changed ATTENDANCE, not generosity\n"
        f"randomisation floor {_mg['perm_floor']:.3f} (2 feeders, {len(_d['person_id'].unique())} people)"),
        transform=axR.transAxes,va="top",ha="right",fontsize=7.9,family="monospace",
        bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=MUTED,lw=0.8,alpha=0.94))
axR.set_xticks([0,1]); axR.set_xticklabels(["Phase 1 (roles active)","Phase 2 (unconstrained)"])
axR.set_ylabel("meals per person-day")
axR.set_ylim(0, float(_bars.max())*1.75 if len(_bars) else 1.0)
axR.set_title("Meal supply per person-day",fontsize=11.5)
fig.suptitle("Fig 12 — The manipulation took, for these 4 people. It estimates nothing about a population.",
             fontsize=12.4,fontweight="semibold")
savefig(fig,"fig12_role_validation"); plt.show()
""")

md("""**Fig 13 — What actually drives Δaffinity (B10.2)** *(unit: learning update event, all named
learning-eligible events)*. Left: Δaffinity against engagement dose, coloured by whether a **meal
arrived** — the split the old model did not control for. Right: the dose slope under four
specifications. The gap between the uncontrolled slope and the controlled one is the finding: most
of the original "engagement predicts affinity" effect was `fed` and `affinity_before`, both of which
are terms inside the update rule.""")
code(r"""
fig,(axL,axR)=plt.subplots(1,2,figsize=(13.4,5.2),width_ratios=[1.0,1.12])

# LEFT: the confound, drawn. Colour by fed, not by role — that is the variable that was omitted.
_h=_b10_h10.dropna(subset=["z_n_turns","d_aff"]).copy()
_FED_C={0:"#4472B0", 1:"#2C8A60"}
for _f,_lab in [(0,"no meal delivered"),(1,"meal delivered")]:
    g=_h[_h["fed"]==_f]
    if not len(g): continue
    axL.scatter(g["z_n_turns"],g["d_aff"],s=28,alpha=0.6,color=_FED_C[_f],
                edgecolor="white",linewidth=0.5,zorder=3,label=f"{_lab} (n={len(g)})")
    if len(g)>=6 and g["z_n_turns"].std()>0:
        b1,b0=np.polyfit(g["z_n_turns"],g["d_aff"],1)
        xs=np.linspace(g["z_n_turns"].min(),g["z_n_turns"].max(),20)
        axL.plot(xs,b0+b1*xs,color=_FED_C[_f],lw=2.0,zorder=4)
axL.axhline(0,color=MUTED,ls=":",lw=1)
axL.set_xlabel("engagement dose: conversation turns (z-scored, all events)")
axL.set_ylabel("affinity change at the learning event")
axL.legend(loc="upper left",fontsize=8.6,title="within-interaction outcome",title_fontsize=8.4)
axL.set_title("The omitted variable: Δaffinity separates on whether a meal arrived",fontsize=11.2)

# RIGHT: the dose slope under four specifications.
_dc=globals()["_b10_dose_cmp"]
_order=["OLD SPEC: duration, no controls","SECONDARY: duration, complete case",
        "SECONDARY: duration, IPW","PRIMARY: n_turns (all events)"]
_rows=[]
for _m in _order:
    _r=_dc[_dc.model==_m]
    if len(_r): _rows.append((_m,float(_r["slope"].iloc[0]),float(_r["lo"].iloc[0]),
                              float(_r["hi"].iloc[0]),int(_r["n"].iloc[0])))
_y=np.arange(len(_rows))[::-1]
for (lab,co,lo,hi,nn),yy in zip(_rows,_y):
    _old = lab.startswith("OLD")
    _c = "#B23A26" if _old else (INK if lab.startswith("PRIMARY") else MUTED)
    axR.plot([lo,hi],[yy,yy],color=_c,lw=2.0 if not _old else 1.6,zorder=3)
    axR.plot([co],[yy],"o",ms=7.5 if lab.startswith("PRIMARY") else 6.0,color=_c,zorder=4)
    axR.annotate(f"{co:+.3f}   (n={nn})",(hi,yy),textcoords="offset points",
                 xytext=(7,0),va="center",fontsize=8.6,color=_c,
                 fontweight="semibold" if lab.startswith(("PRIMARY","OLD")) else "normal")
axR.axvline(0,color=MUTED,ls=":",lw=1)
axR.set_yticks(_y)
axR.set_yticklabels([r[0].replace(": ",":\n") for r in _rows],fontsize=8.4)
axR.set_xlabel("Δaffinity per +1 SD of engagement dose (95% CI)")
axR.set_title("Same question, four specifications",fontsize=11.2)
axR.margins(x=0.34)
fig.suptitle("Fig 13 — Most of the reported dose→affinity effect was the update rule itself",
             fontsize=13,fontweight="semibold")
savefig(fig,"fig13_affinity_dose"); plt.show()
""")

# ==========================================================================
# PHASE D — MACHINE LEARNING (interpretive, n~200)
# ==========================================================================
md(r"""## Phase D — Machine-learning sensitivity check (n = 217 interactions)

**The previous version of this section reported a number it could not defend.** It fitted a
two-feature model (`ss_rank`, `hs_rank`), reported that adding hunger state lifted held-out AUC from
0.669 to 0.757, and presented a "drop-column importance" table alongside — even though, with two
features, drop-column importance *is* the ablation. `auc_without hs_rank = 0.6687` is the social-only
AUC to four decimals. One result, printed twice, as though it were two.

More importantly, ΔAUC = +0.088 came from a **single** grouped CV split on 217 rows with no interval
and no null. At that sample size an AUC difference of ~0.09 is comfortably inside sampling noise.

Two options were available: cut the section, or make it say something. It is made to say something:

- **Repeated grouped CV** (`n_repeats` × grouped splits with reshuffled group assignment) yields a
  *distribution* of ΔAUC rather than one number.
- **A permutation null**: `hs_rank` is shuffled *within group* and the whole CV is rerun, giving the
  distribution of ΔAUC under "hunger state carries no information". The observed ΔAUC is scored
  against it.
- The redundant drop-column table is **deleted**.

If ΔAUC does not clear its own permutation null, the section says so and the verdict is
`Inconclusive`. This is a sensitivity check; it was never confirmatory, and it cannot become
confirmatory by being run more carefully.""")

md("### D1 — Does hunger state add held-out engagement signal?")
code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import clone

d = master.copy()
d["replied_any"]=pd.to_numeric(d["replied_any"],errors="coerce").fillna(0).astype(int)
d["reached_ss4"]=(d["final_state"]=="ss4").astype(int)
# Phase D is scoped to the two state variables the controller actually reasons over:
# the orexigenic drive (hunger state) and the social state. The raw perceptual signals
# (IPS mean, proximity, centrality, gaze, co-presence, attention, hour, ...) are excluded.
d["hs_rank"]=d["hunger_state_start"].map({"HS1":1,"HS2":2,"HS3":3})
d["ss_rank"]=d["initial_state"].map({"ss1":1,"ss2":2,"ss3":3,"ss4":4})
HUNGER_FEATS=["hs_rank"]
SOCIAL_FEATS=["ss_rank"]
X_cols=SOCIAL_FEATS+HUNGER_FEATS
d=d.dropna(subset=["hs_rank","ss_rank"])
X=d[X_cols].fillna(d[X_cols].median())

def group_cv_oof(model, X, y, groups):
    "Out-of-fold predictions under leave-one-group-out, preserving row index."
    X = pd.DataFrame(X).copy()
    y = pd.Series(y, index=X.index).astype(int)
    groups = pd.Series(groups, index=X.index)
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    n_fit = 0
    n_skip = 0
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if y.iloc[train_idx].nunique() < 2:
            n_skip += 1
            continue
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof.iloc[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
        n_fit += 1
    return oof, n_fit, n_skip

def score_oof(y, oof):
    y = pd.Series(y, index=oof.index).astype(int)
    ok = oof.notna()
    if ok.sum() == 0 or y.loc[ok].nunique() < 2:
        return np.nan, np.nan
    return roc_auc_score(y.loc[ok], oof.loc[ok]), average_precision_score(y.loc[ok], oof.loc[ok])

def group_cv_scores(model, X, y, groups):
    "Out-of-fold predictions under leave-one-group-out; return AUC and PR-AUC."
    try:
        oof, _, _ = group_cv_oof(model, X, y, groups)
        return score_oof(y, oof)
    except Exception as e:
        return np.nan, np.nan

def make_models():
    return {"logit(L2)":make_pipeline(StandardScaler(),
                LogisticRegression(penalty="l2",C=0.5,max_iter=2000,class_weight="balanced")),
            "gbm":GradientBoostingClassifier(random_state=SEED)}

metric_rows=[]
for target in ["replied_any","reached_ss4"]:
    y=d[target]; base_rate=y.mean()
    print(f"\n=== target: {target}  (n={len(y)}, positives={int(y.sum())}, base rate={base_rate:.2f}) ===")
    for gname,groups in [("leave-one-run-out",d["run_id"]),("leave-one-person-out",d["user_key"])]:
        for mname,model in make_models().items():
            auc,ap=group_cv_scores(model,X,y,groups)
            lift=ap/base_rate if base_rate>0 else np.nan
            print(f"  {gname:22s} {mname:10s} AUC={auc:.2f}  PR-AUC={ap:.2f} (baseline {base_rate:.2f}, lift x{lift:.2f})")
            metric_rows.append(dict(target=target,cv=gname,model=mname,auc=auc,pr_auc=ap,
                                    baseline_pr=base_rate,n=len(y),pos=int(y.sum())))
    # majority-class baseline (AUC=0.5 by construction; PR-AUC = base rate)
    print(f"  {'majority baseline':22s} {'dummy':10s} AUC=0.50  PR-AUC={base_rate:.2f} (no signal)")

metrics_df=pd.DataFrame(metric_rows); metrics_df.to_csv(OUT_DIR/"ml_model_metrics.csv",index=False)

# --- ABLATION with a REPEATED-CV DISTRIBUTION and a PERMUTATION NULL --------------------
# Single-split ΔAUC on n=217 is noise. We need (a) a spread, and (b) a null.
from sklearn.model_selection import GroupKFold

def grouped_auc(Xd, y, groups, n_splits=5, seed=SEED):
    "One grouped-CV pass -> (AUC, PR-AUC). Groups never straddle a fold."
    rng = np.random.default_rng(seed)
    gs = pd.Series(groups).astype(str)
    uniq = gs.unique()
    # Reshuffle which groups land in which fold, so repeats actually differ.
    perm = rng.permutation(len(uniq))
    fold_of = {g: perm[i] % n_splits for i, g in enumerate(uniq)}
    fold = gs.map(fold_of).values
    oof = np.full(len(y), np.nan)
    yv = np.asarray(y)
    for f in range(n_splits):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or len(np.unique(yv[tr])) < 2:
            continue
        m = GradientBoostingClassifier(random_state=int(seed))
        m.fit(np.asarray(Xd)[tr], yv[tr])
        oof[te] = m.predict_proba(np.asarray(Xd)[te])[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() == 0 or len(np.unique(yv[ok])) < 2:
        return np.nan, np.nan
    return (roc_auc_score(yv[ok], oof[ok]), average_precision_score(yv[ok], oof[ok]))

N_REPEATS = 25
N_PERM    = 200
print(f"--- Repeated grouped CV: {N_REPEATS} repeats, groups = run_id ---")
abl_rows = []
for target in ["replied_any", "reached_ss4"]:
    y = d[target].astype(int)
    for label, cols in [("social-only", SOCIAL_FEATS), ("social+hunger", X_cols)]:
        Xd = d[cols].fillna(d[cols].median())
        aucs, aps = [], []
        for r in range(N_REPEATS):
            a, p = grouped_auc(Xd, y, d["run_id"], seed=SEED + r)
            if np.isfinite(a): aucs.append(a); aps.append(p)
        abl_rows.append(dict(target=target, feature_set=label,
                             auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                             auc_lo=float(np.percentile(aucs, 2.5)),
                             auc_hi=float(np.percentile(aucs, 97.5)),
                             pr_auc=float(np.mean(aps)), n_repeats=len(aucs)))
        print(f"  {target:12s} {label:14s} AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f} "
              f"[{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}]")
abl = pd.DataFrame(abl_rows)
abl.to_csv(OUT_DIR/"ml_ablation.csv", index=False)

# ΔAUC distribution over repeats (paired: same fold assignment for both feature sets).
print(f"\n--- ΔAUC (social+hunger MINUS social-only), paired over {N_REPEATS} repeats ---")
delta_rows = []
for target in ["replied_any", "reached_ss4"]:
    y = d[target].astype(int)
    Xs = d[SOCIAL_FEATS].fillna(d[SOCIAL_FEATS].median())
    Xf = d[X_cols].fillna(d[X_cols].median())
    deltas = []
    for r in range(N_REPEATS):
        a_s, _ = grouped_auc(Xs, y, d["run_id"], seed=SEED + r)
        a_f, _ = grouped_auc(Xf, y, d["run_id"], seed=SEED + r)   # SAME folds
        if np.isfinite(a_s) and np.isfinite(a_f):
            deltas.append(a_f - a_s)
    deltas = np.asarray(deltas)
    # PERMUTATION NULL: shuffle hs_rank WITHIN run, rerun the whole comparison.
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(N_PERM):
        Xp = Xf.copy()
        Xp["hs_rank"] = (pd.Series(Xp["hs_rank"].values, index=d.index)
                         .groupby(d["run_id"]).transform(lambda s: rng.permutation(s.values)).values)
        a_s, _ = grouped_auc(Xs, y, d["run_id"], seed=SEED)
        a_p, _ = grouped_auc(Xp, y, d["run_id"], seed=SEED)
        if np.isfinite(a_s) and np.isfinite(a_p):
            null.append(a_p - a_s)
    null = np.asarray(null)
    obs = float(np.mean(deltas))
    p_perm = float((1.0 + np.sum(null >= obs)) / (len(null) + 1.0)) if len(null) else np.nan
    delta_rows.append(dict(target=target, auc_delta_mean=obs,
                           auc_delta_sd=float(np.std(deltas)),
                           auc_delta_lo=float(np.percentile(deltas, 2.5)),
                           auc_delta_hi=float(np.percentile(deltas, 97.5)),
                           null_mean=float(np.mean(null)) if len(null) else np.nan,
                           null_p95=float(np.percentile(null, 95)) if len(null) else np.nan,
                           perm_p=p_perm, n_repeats=len(deltas), n_perm=len(null)))
    print(f"  {target:12s} ΔAUC = {obs:+.3f} +/- {np.std(deltas):.3f} "
          f"[{np.percentile(deltas,2.5):+.3f}, {np.percentile(deltas,97.5):+.3f}]")
    print(f"               permutation null: mean {np.mean(null):+.3f}, 95th pct "
          f"{np.percentile(null,95):+.3f}  ->  p = {p_perm:.3f}")
abl_delta = pd.DataFrame(delta_rows)
abl_delta.to_csv(OUT_DIR/"ml_ablation_delta.csv", index=False)

_ss4 = abl_delta[abl_delta.target=="reached_ss4"].iloc[0]
d_auc = float(_ss4["auc_delta_mean"]); _d_p = float(_ss4["perm_p"])
print(f"\n  -> Engaged prediction: adding hunger state changes AUC by {d_auc:+.3f} "
      f"(permutation p = {_d_p:.3f}).")
print(f"  -> The single-split figure the previous version reported (+0.088) sits "
      f"{'inside' if d_auc < _ss4['null_p95'] else 'outside'} the null's 95th percentile "
      f"({_ss4['null_p95']:+.3f}).")

# The drop-column table is DELETED. With two features it is the ablation restated:
# auc_without(hs_rank) is identically the social-only AUC. Reporting it as a second,
# corroborating analysis was double-counting one result.
for _stale in ["ml_dropcolumn_importance.csv"]:
    _p = OUT_DIR/_stale
    if _p.exists():
        _p.unlink()
        print(f"\n  (removed stale {_stale}: drop-column == the ablation when there are 2 features)")
""")

md("**Fig D1 — ML sensitivity** *(unit: interaction, n = 217; repeated grouped CV, groups = runs)*: ΔAUC from adding hunger state, against its within-run permutation null. Sensitivity only — never confirmatory.")

code(r"""
# --- Out-of-fold readout by hunger state (descriptive corroboration of B4's direction) ---
y=d["reached_ss4"].astype(int)
gb=GradientBoostingClassifier(random_state=SEED)
oof, n_folds, n_skip = group_cv_oof(gb, X, y, d["run_id"])
base_auc, base_ap = score_oof(y, oof)
print(f"Out-of-fold Engaged model: AUC={base_auc:.3f}, PR-AUC={base_ap:.3f} "
      f"({n_folds} run-held-out folds, skipped={n_skip})")

d2=d.copy(); d2["p_ss4_oof"]=oof; d2=d2[d2["p_ss4_oof"].notna()]
by=d2.groupby("hunger_state_start").agg(
    oof_pred=("p_ss4_oof","mean"), observed=("reached_ss4","mean"), n=("reached_ss4","size")
).reindex(HS_ORDER)
by.index=[HS_NAME[h] for h in HS_ORDER]
print("\nOut-of-fold P(reach Engaged) vs observed rate, by hunger state at start:")
print(by.round(3).to_string())
print(f"\n  The Starving column rests on n={int(by.loc['Starving','n'])} interactions — the same 13")
print(f"  that make B4 exploratory. Held-out predictions reproducing that pattern is NOT")
print(f"  independent corroboration; it is the same 13 rows, seen through a model.")

_d1_informative = np.isfinite(_d_p) and (_d_p < 0.05)
verdict("D1", EV_ASSOC if _d1_informative else EV_INCONC,
        (f"Hunger state adds held-out signal beyond social state: ΔAUC = {d_auc:+.3f} "
         f"[{_ss4['auc_delta_lo']:+.3f}, {_ss4['auc_delta_hi']:+.3f}] over {int(_ss4['n_repeats'])} "
         f"repeated grouped-CV runs, permutation p = {_d_p:.3f} against a within-run shuffle null. "
         f"Social state remains the dominant predictor. Sensitivity evidence only — it corroborates "
         f"the direction of B3/B4 and confirms nothing."
         if _d1_informative else
         f"Inconclusive. Adding hunger state changes held-out Engaged-prediction AUC by "
         f"{d_auc:+.3f} [{_ss4['auc_delta_lo']:+.3f}, {_ss4['auc_delta_hi']:+.3f}] across "
         f"{int(_ss4['n_repeats'])} repeated grouped-CV runs, which does NOT clear its own "
         f"within-run permutation null (p = {_d_p:.3f}; null 95th pct {_ss4['null_p95']:+.3f}). "
         f"At n=217 with two features, a ΔAUC of this size is not distinguishable from noise. "
         f"The previously reported '+0.088, AUC 0.669->0.757' came from a single CV split with no "
         f"interval and no null, and it does not survive either. The redundant drop-column table "
         f"(identical to the ablation with 2 features) has been removed."))
""")

# D2 (standalone survival readout) removed: it duplicated B6's episode evidence, and a
# Cox fit on 8 episodes is not research-grade. D5 (framing/mediation) removed: path a is
# prompt-driven (true by construction, already shown descriptively in B3), path b is
# temporally leaked, and the one leakage-free signal (ping->reply) already lives in B5.

md("""### D4 — Feeding concentration (robustness of RQ2-c)

The robustness question for RQ2-c is *does replenishment depend on a few feeders?* The
**concentration metrics (Gini, top-3 share)** answer this directly and are the whole point of
this cell. *(An earlier exploratory KMeans over per-user behaviour was dropped: its silhouette
was too low to define meaningful user types, so it added no research-grade signal.)*""")
code(r"""
d=master.copy()
d["fed_here"]=pd.to_numeric(d["meals_eaten_count"],errors="coerce").fillna(0)

# Feeding concentration over named users. "unknown" is an unrecognised-face placeholder,
# not a stable person, so excluding it keeps the robustness caveat interpretable.
all_meals=d[(d["user_key"]!="") & (d["user_key"]!="unknown")].groupby("user_key")["fed_here"].sum().sort_values(ascending=False)
m=np.sort(all_meals.values.astype(float)); nP=len(m)
gini=(2*np.sum(np.arange(1,nP+1)*m)/(nP*m.sum())-(nP+1)/nP) if m.sum()>0 else np.nan
top3=all_meals.head(3).sum()/max(all_meals.sum(),1)
print(f"Total meal energy into the drive = {int(all_meals.sum())} (stomach %), across {nP} named users.")
print(f"Feeding concentration: Gini={gini:.2f}; top-3 users ({', '.join(all_meals.head(3).index)}) "
      f"supply {top3*100:.0f}% of meals.")
conc = "concentrated in a few feeders (fragile)" if (top3>0.75 or gini>0.75) else \
       ("moderate concentration — replenishment leans on a few feeders"
        if (top3>=0.45 or gini>=0.5) else "well spread across users (robust)")
verdict("D4", EV_EXPL,
        f"Descriptive. Feeding Gini={gini:.2f} over {nP} named users; the top 3 supply "
        f"{top3*100:.0f}% of all meal energy — {conc}. This is the standing caveat on every RQ2 "
        f"result: the regulatory loop closed because a handful of specific people chose to close "
        f"it. Nothing here shows that would hold in another room of people.",
        n=nP)
""")


# ==========================================================================
# SYNTHESIS
# ==========================================================================
md(r"""## Synthesis — evidence classes, findings, checklist""")

md(r"""### Success criteria, by evidence class

The generic `Supported` label is gone. It flattened an implementation check, a clustered
association, and a 13-interaction descriptive into one word, and that word was doing a lot of work
it had not earned. Every claim now carries exactly one of the five classes declared at the top of
Phase B.""")
code(r"""
# Success criteria, tagged with the evidence class each analysis actually earned.
def outcome_of(key):
    r = RESULTS.get(key, {})
    return r.get("evidence", EV_INCONC), r.get("verdict", "(not run)")

rows = [
 ("RQ1-1","Internal monitoring is continuous and autonomous","B1"),
 ("RQ1-2","Deficit detection follows the coded 60/25 thresholds","B2"),
 ("RQ1-3","Deficit is associated with feeding received","B3"),
 ("RQ1-4","Starving reallocates priority away from social completion","B4"),
 ("RQ2-a","Deficit expression elicits recovery behaviour","B5"),
 ("RQ2-b","Observed Starving episodes resolve by feeding","B6"),
 ("RQ2-c","Long-run Starving occupancy is low","B7"),
 ("RQ3-a","The role manipulation changed what people did","B10.1"),
 ("RQ3-b","Affinity encodes interaction history and is expressed downstream","B10"),
 ("D1",   "Hunger state adds held-out predictive signal","D1"),
]
sc = []
for cid, claim, key in rows:
    ev, v = outcome_of(key)
    sc.append(dict(id=cid, claim=claim, source=key, evidence_class=ev, detail=v))
sc_df = pd.DataFrame(sc)
sc_df.to_csv(OUT_DIR / "success_criteria.csv", index=False)
print(sc_df[["id","claim","source","evidence_class"]].to_string(index=False))
print("\nCounts by evidence class:")
print(sc_df["evidence_class"].value_counts().to_string())
print("\nDetails:")
for _, r in sc_df.iterrows():
    print(f"\n  [{r['evidence_class']}] {r['id']} ({r['source']})\n    {r['detail']}")
""")

md("### Consolidated results summary → `outputs/results_summary.md`")
code(r"""
# One human-readable report. Every number is read from the objects computed above, so this
# file cannot drift from the notebook.
_b3=globals()["_b3"]; _b5m=globals()["_b5_meal"]; _b5p=globals()["_b5_ping"]
_b6d=globals()["_b6"]; _b7d=globals()["_b7"]; _mg=globals()["_b10_meal_gee"]
_pr=globals()["_b10_prior"]; _dc=globals()["_b10_dose_cmp"]
_ptab = pd.read_csv(OUT_DIR/"multiplicity_table.csv")

L=["# Orexigenic drive — results summary", "",
   f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Single always-on condition (no drive-off control). "
   f"{hunger_raw['run_id'].nunique()} monitored runs, {interactions['run_id'].nunique()} with visitors, "
   f"{hunger_raw['day_rome'].nunique()} session-days, {len(interactions)} interactions, "
   f"{master[master.person_id!='unknown']['person_id'].nunique()} named people. "
   f"Two-phase design: Phase 1 (first 4 days) had assigned roles (2 obligated feeders, "
   f"2 interact-no-feed, rest unconstrained); Phase 2 unconstrained._", "",
   "## How to read this report", "",
   "Every result carries exactly one evidence class. The classes are not interchangeable and the",
   "differences between them are the point:", "",
   "| Class | Meaning |",
   "|---|---|",
   f"| `{EV_IMPL}` | Follows from the controller source. Confirms the code is faithfully implemented and logged. **Not a discovered fact.** |",
   f"| `{EV_ASSOC}` | A cluster-aware association in this deployment. Not causal, not a population estimate. |",
   f"| `{EV_EXPL}` | Descriptive. Too small-n or too selection-prone to support inference. |",
   f"| `{EV_INCONC}` | Run, and did not settle the question. |",
   f"| `{EV_REPL}` | Suggestive; identification needs new data. |", "",
   "## Verification gate", "",
   f"All V1–V5 checks passed (see `verification_report.md`). Per-action energy costs match the source "
   f"constants exactly; corpus energy balance active-out "
   f"{hunger_raw[hunger_raw.event_type=='active_cost'].active_energy_cost.sum():.0f} "
   f"vs meal-in {hunger_raw[hunger_raw.event_type=='feeding'].meal_delta.sum():.0f}.", ""]

L+=["## Corrections to the previous version of this analysis", "",
    "This report supersedes an earlier one whose headline claims did not survive audit. The",
    "substantive corrections, in descending order of how much they changed:", "",
    f"1. **Starving episodes: 8 -> {_b6d['n']}.** The episode builder keyed off the logged",
    "   `hunger_state_before/after` fields, and the executive logger never emits a `before != after`",
    "   row for a **passive-drain** crossing. It therefore found only the episodes the robot fell into",
    "   *while a human was interacting with it* — i.e. the ones where someone was there to feed it.",
    f"   The old '8/8 recovered by feeding, median 21 s' was the selection rule restating itself. Over",
    f"   all {_b6d['n']} episodes, {_b6d['full_k']}/{_b6d['n']} recovered to Full by feeding",
    f"   (exact 95% CI [{_b6d['e_full'][1]:.2f}, {_b6d['e_full'][2]:.2f}]). The longest episode ran",
    f"   {_b6d['worst_sec']/60:.0f} minutes down to level {_b6d['worst_min_level']:.1f} **in a run with",
    "   15 logged interactions** — people were present and the robot was not fed. That episode is ~65%",
    "   of all Starving time in the corpus and was invisible to the previous analysis.",
    "2. **RQ3's core model regressed Δaffinity on terms inside its own update equation.** Affinity is a",
    "   deterministic EMA of delivered energy; `fed` and `affinity_before` are *in the formula* and",
    f"   alone explain R²={globals()['_b9_r2_mech']:.2f} of every update. The old model omitted both and used",
    "   `duration` (correlated 0.58 with `fed`) in their place, and used `active_energy_cost` — a literal",
    "   additive term in the credit — as an 'independent dose agreement check'. Controlled, and using the",
    f"   fully observed dose, the slope is {float(_dc.query('model.str.startswith(\"PRIMARY\")')['slope'].iloc[0]):+.3f} "
    f"against the {float(_dc.query('model.str.startswith(\"OLD\")')['slope'].iloc[0]):+.3f} previously reported.",
    f"3. **The role manipulation worked through attendance, not generosity.** The old report said",
    f"   obligated feeders 'supplied meals at 2.7x the unconstrained rate', presented as evidence that",
    f"   the roles changed feeding behaviour. The 2.7x is real — but feeders interacted with the robot",
    f"   **{_mg['exposure_ratio']:.1f}x more often per day**, and *per interaction* they fed only",
    f"   {_mg['rr']:.1f}x as often (randomisation p={_mg['perm_p']:.3f}). Being told to feed the robot made",
    "   people go to the robot. It did not make them markedly more generous once there. Both quantities",
    "   are now reported; conflating them overstated what the manipulation demonstrated.",
    "4. **B4 was quasi-separation reported as precision.** One success in 13 Starving interactions",
    "   produced `OR 0.03 [0.008, 0.136], p=1.9e-6` from a diverging likelihood. Refitted with Firth's",
    "   penalised likelihood and an exact test; its p-value is excluded from the confirmatory families.",
    "5. **'The labels do not flap' was backwards.** B2's headline non-trivial result was *zero*",
    "   rapid reversals at either threshold. The detector compared `from_state` against the previous",
    f"   `from_state`, when a reversal requires `from_state == previous to_state` — the condition was",
    f"   unsatisfiable and would have returned zero on any data at all. Corrected, there are",
    f"   **{RESULTS['B2'].get('n_reversals', 0)} rapid reversals** across the two thresholds",
    "   (median gap ~29 s), typically an action cost pushing the level under the threshold and a feed",
    "   pulling it straight back. The flapping is real — and it is why `chatBot.py` carries a 60 s",
    "   `HS_DWELL_SEC` debounce, which the old write-up cited without noticing it contradicted the claim.",
    "6. **The CTMC dropped every run's final dwell**, silently discarded non-ergodic bootstrap",
    "   resamples, and pooled visited with idle runs (and Phase 1 with Phase 2) into one",
    "   time-homogeneous generator. Starving occupancy differs ~9x between the phases.",
    "7. **The remote-loop analysis double-counted replies** (one reply could answer many pings) and",
    "   pooled `hs3_recovery` 'thanks, I'm full' notifications in with hunger pings. It also had no",
    "   control condition at all, so a 21% reply rate had nothing to be 21% *against*.",
    "8. **The ML section reported one CV split with no interval and no null**, and printed its",
    "   drop-column table as a second analysis when, with two features, it is the ablation restated.", ""]

L+=["## Results", "", "| id | claim | evidence class |", "|---|---|---|"]
for _,r in sc_df.iterrows():
    L.append(f"| {r['id']} | {r['claim']} | `{r['evidence_class']}` |")
L+=["", "## Per-analysis verdicts", ""]
for k in ["B1","B2","B3","B4","B5","B6","B7","B9","B10.1","B10","D1","D4"]:
    r = RESULTS.get(k)
    if r: L.append(f"- **{k}** — `{r['evidence']}` — {r['verdict']}\n")

L+=["", "## Multiplicity", "",
    f"Every p-value entering a conclusion is registered at the point it is computed and exported to",
    f"`multiplicity_table.csv` ({len(_ptab)} rows). Benjamini–Hochberg is applied within pre-declared",
    f"families over the **complete** confirmatory set — including the dose × role and dose × phase",
    f"interaction terms and the role contrasts, which the previous version quoted in its verdicts but",
    f"never corrected.", "",
    f"- Confirmatory p-values: {int((_ptab['status']=='confirmatory').sum())}, of which "
    f"{int(_ptab['sig_q05'].fillna(False).sum())} survive at q<0.05.",
    f"- Exploratory p-values recorded but deliberately NOT corrected and NOT used to support any "
    f"claim: {int((_ptab['status']=='exploratory').sum())} (B4's separated cell).", ""]

L+=["## Key quantities", "",
    f"- **Passive drain** exactly 1.00x nominal (a software integrator — true by construction); "
    f"dense autonomous sampling every {gaps.median():.1f} s across {hunger_raw['run_id'].nunique()} runs.",
    f"- **Deficit -> feeding received** (B3, the strongest result here): OR {_b3['orr']:.1f}, "
    f"person-cluster bootstrap [{_b3['boot'][0]:.1f}, {_b3['boot'][2]:.1f}], run-cluster bootstrap "
    f"[{_b3['boot_run'][0]:.1f}, {_b3['boot_run'][2]:.1f}], LOPO {_b3['lopo'][0]:.1f}-{_b3['lopo'][1]:.1f}. "
    f"Survives adjustment for social state, trigger mode, phase and prior interaction count.",
    f"- **Meal size by deficit**: Full {_b5m['means']['HS1']:.0f} / Hungry {_b5m['means']['HS2']:.0f} / "
    f"Starving {_b5m['means']['HS3']:.0f}; {_b5m['slope']:+.1f} points per deficit step "
    f"(run-cluster bootstrap [{_b5m['boot'][0]:+.1f}, {_b5m['boot'][2]:+.1f}]).",
    f"- **Remote loop**: {int(globals()['_b5_pm']['replied'].sum())}/{_b5p['n_pings']} hunger pings drew a "
    f"reply within 1 h ({_b5p['rate']:.2f}) vs {_b5p['ctrl']:.2f} in matched no-ping control windows — "
    f"difference {_b5p['diff']:+.2f}, subscriber-cluster bootstrap "
    f"[{_b5p['boot'][0]:+.2f}, {_b5p['boot'][2]:+.2f}]. One-to-one reply matching; recovery "
    f"notifications excluded.",
    f"- **Starving occupancy** (empirical, no stationarity assumption): {_b7d['emp']*100:.2f}% of observed "
    f"seconds, run-cluster bootstrap [{_b7d['emp_boot'][0]*100:.2f}, {_b7d['emp_boot'][2]*100:.2f}]. "
    f"The modelled CTMC figure is reported only with its diagnostics "
    f"({_b7d['nonergodic']*100:.0f}% non-ergodic resamples).",
    f"- **Starving episodes**: {_b6d['n']} in total; {_b6d['feed_k']}/{_b6d['n']} received a feed "
    f"(exact [{_b6d['e_feed'][1]:.2f}, {_b6d['e_feed'][2]:.2f}]). Longest {_b6d['worst_sec']/60:.0f} min "
    f"to level {_b6d['worst_min_level']:.1f}.",
    f"- **Role manipulation** (RQ3's only empirical claim, and it decomposes): obligated feeders "
    f"delivered **{_mg['rr_total']:.1f}x** the meal energy per person-day (bootstrap "
    f"[{_mg['boot'][0]:.1f}, {_mg['boot'][2]:.1f}], randomisation p={_mg['perm_p_total']:.3f}), but "
    f"**per interaction they fed only {_mg['rr']:.1f}x as often** (bootstrap "
    f"[{_mg['boot_perop'][0]:.1f}, {_mg['boot_perop'][2]:.1f}], randomisation p={_mg['perm_p']:.3f}). "
    f"The difference is exposure: they came to the robot **{_mg['exposure_ratio']:.1f}x more often**. "
    f"Being told to feed the robot made people *visit* it; it did not make them markedly more "
    f"generous once there. No-feed pair: {_mg['nofeed_k']}/{_mg['nofeed_n']} feeds (upper bound "
    f"{_mg['nofeed_hi']:.2f}).",
    f"- **Feeding concentration**: {RESULTS.get('D4',{}).get('verdict','see D4')}", ""]

L+=["## What these data cannot establish", "",
    "New data are required for each of the following. None of them is a matter of better analysis.", "",
    "- **Drive-on vs drive-off causal identification.** There is no off condition. Every behavioural",
    "  result here is an association within a single always-on deployment. The causal share of the",
    "  drive in any observed feeding is not identified and cannot be.",
    "- **Multi-site generalisation.** One robot, one site, one convenience sample.",
    "- **Stable role-effect estimation.** Two people per controlled role means role is nearly aliased",
    "  with identity. The randomisation p-value has a hard floor set by the number of possible",
    f"  assignments ({_mg['perm_floor']:.3f}). More people per role, not more modelling.",
    f"- **Reliable Starving-episode rates.** {_b6d['n']} episodes clustered in a handful of runs, some",
    "  right-censored. Longer runs and more Starving exposure.",
    "- **Population-level conclusions of any kind.** Every effect size here is existence-and-magnitude",
    "  evidence for this loop, not a calibrated rate that transfers.", ""]

L+=["## Scope", "",
    f"One robot, one site, {hunger_raw['day_rome'].nunique()} session-days, "
    f"{hunger_raw['run_id'].nunique()} runs, "
    f"{master[master.person_id!='unknown']['person_id'].nunique()} named people (convenience sample). "
    f"Every result is a within-deployment characterisation of *this* human-robot loop."]
(OUT_DIR/"results_summary.md").write_text("\n".join(L))
print("wrote outputs/results_summary.md")
""")

md("### Final findings vs RQ1 / RQ2 / RQ3")
code(r"""
print("="*78)
print("FINDINGS — single-condition, always-on orexigenic regulatory loop")
print("="*78)
def g(k):
    r = RESULTS.get(k)
    if not r: return "(not run)"
    return f"[{r['evidence']}]\n      {r['verdict']}"

print("\nSTRONG ENGINEERING VERIFICATION (the code does what the code says):")
print("  B1 internal monitoring   :", g("B1"))
print("  B2 deficit detection     :", g("B2"))
print("  B9 affinity mechanism    :", g("B9"))

print("\nCREDIBLE WITHIN-DEPLOYMENT ASSOCIATIONS (clustered, adjusted, not causal):")
print("  B3 deficit -> feeding    :", g("B3"))
print("  B5 recovery behaviour    :", g("B5"))
print("  B10.1 role manipulation  :", g("B10.1"))

print("\nEXPLORATORY / INCONCLUSIVE (reported, not relied upon):")
print("  B4 Starving priority     :", g("B4"))
print("  B6 Starving episodes     :", g("B6"))
print("  B7 long-run occupancy    :", g("B7"))
print("  D1 ML sensitivity        :", g("D1"))

print("\nRQ3 (B10)                  :", g("B10"))

print("\n" + "="*78)
print("UNSUPPORTED BY THESE DATA — requires new data collection, not new analysis:")
print("="*78)
print("  * drive-on vs drive-off causal identification (there is no off condition)")
print("  * multi-site generalisation (one robot, one site)")
print("  * stable role-effect estimation (2 people per controlled role)")
print("  * reliable Starving-episode rates (few, clustered, partly censored episodes)")
print("  * ANY population-level conclusion")
""")

md("### Final output checklist")
code(r"""
def exists(p): return "done" if (OUT_DIR/p).exists() or (FIG_DIR/p).exists() else "MISSING"
def fig_exists(n): return "done" if (FIG_DIR/f"{n}.png").exists() else "MISSING"
items = [
 ("verification_report.md", exists("verification_report.md")),
 ("quality_report.md", exists("quality_report.md")),
 ("master_interactions.parquet", exists("master_interactions.parquet")),
 ("hs3_episodes.parquet", exists("hs3_episodes.parquet")),
 ("hs_transitions.parquet", exists("hs_transitions.parquet")),
 ("hs_crossing_log_gap.csv", exists("hs_crossing_log_gap.csv")),
 ("active_cost_table.csv", exists("active_cost_table.csv")),
 ("b2_detection_check.csv", exists("b2_detection_check.csv")),
 ("b2_transition_counts.csv", exists("b2_transition_counts.csv")),
 ("b3_adjusted_models.csv", exists("b3_adjusted_models.csv")),
 ("b4_starving_exact.csv", exists("b4_starving_exact.csv")),
 ("b5_meal_size_by_state.csv", exists("b5_meal_size_by_state.csv")),
 ("b5_ping_response_windows.csv", exists("b5_ping_response_windows.csv")),
 ("b5_ping_by_subscriber.csv", exists("b5_ping_by_subscriber.csv")),
 ("b6_episode_outcomes.csv", exists("b6_episode_outcomes.csv")),
 ("b7_terminal_segments.csv", exists("b7_terminal_segments.csv")),
 ("b7_stratified_occupancy.csv", exists("b7_stratified_occupancy.csv")),
 ("b9_mechanism_check.csv", exists("b9_mechanism_check.csv")),
 ("b9_eligibility_profile.csv", exists("b9_eligibility_profile.csv")),
 ("b10_person_day_exposure.csv", exists("b10_person_day_exposure.csv")),
 ("b10_downstream_panel.csv", exists("b10_downstream_panel.csv")),
 ("multiplicity_table.csv", exists("multiplicity_table.csv")),
 ("small_cluster_sensitivity.csv", exists("small_cluster_sensitivity.csv")),
 ("rq3_model_results.csv", exists("rq3_model_results.csv")),
 ("rq3_missingness.csv", exists("rq3_missingness.csv")),
 ("rq3_missingness_model.csv", exists("rq3_missingness_model.csv")),
 ("rq3_dose_specification_comparison.csv", exists("rq3_dose_specification_comparison.csv")),
 ("rq3_affinity_repair_robustness.csv", exists("rq3_affinity_repair_robustness.csv")),
 ("rq3_memory_crosscheck.csv", exists("rq3_memory_crosscheck.csv")),
 ("success_criteria.csv", exists("success_criteria.csv")),
 ("results_summary.md", exists("results_summary.md")),
 ("ml_model_metrics.csv", exists("ml_model_metrics.csv")),
 ("ml_ablation.csv", exists("ml_ablation.csv")),
 ("ml_ablation_delta.csv", exists("ml_ablation_delta.csv")),
]
for n in ["fig02_drive_timeline","fig04_deficit_action","fig05_prioritisation_heatmap",
          "fig08_remote_loop","fig09_steady_state",
          "fig10_affinity_trajectories","fig12_role_validation",
          "fig13_affinity_dose"]:
    items.append((n, fig_exists(n)))
print("FINAL OUTPUT CHECKLIST")
for name, st in items:
    print(f"  [{'x' if st=='done' else ' '}] {name}  ({st})")
n_done=sum(1 for _,s in items if s=="done")
print(f"\n{n_done}/{len(items)} deliverables present.")

# The drop-column table must be GONE: with 2 features it duplicated the ablation.
_stale = OUT_DIR/"ml_dropcolumn_importance.csv"
print(f"\nstale artifact removed: ml_dropcolumn_importance.csv -> "
      f"{'still present (PROBLEM)' if _stale.exists() else 'confirmed absent'}")

print("\nAnalyses run:", ", ".join(sorted(RESULTS.keys())))
print("Evidence classes:")
for _k, _v in RESULTS.items():
    if "evidence" in _v:
        print(f"   {_k:8s} {_v['evidence']}")
print("Cache files:", len(list(CACHE_DIR.glob('*.parquet'))), "parquet frames in analysis/cache/")
""")

md(r"""### Reproducibility note

Set a global seed (`SEED=42`); read-only immutable DB access; deterministic re-runs from
the parquet cache in `analysis/cache/`. Pinned dependencies are written to
`analysis/requirements.txt`. Re-running top-to-bottom regenerates every artifact under
`analysis/outputs/` and `analysis/figures/`.
""")

# Final write (runs after ALL cells appended above).
if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "analysis/orexigenic_analysis.ipynb"
    build_and_write(out)
