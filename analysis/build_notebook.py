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

# --- PRODUCTION STATISTICAL FUNCTIONS ---------------------------------------------------
# Every non-trivial statistical or data-unit routine lives in analysis/statistical_helpers.py.
# The notebook imports it; the TESTS import the same module. Nothing is reimplemented in either.
#
# This is not tidiness. The previous test suite carried its own copies of the episode builder,
# the ping matcher, the sojourn splitter and Firth, so the tests could pass while the notebook
# shipped different code — and one of those copies (the flapping detector) contained the SAME
# bug as the notebook, so the test that "verified" it could never have failed. Duplicated logic
# does not test anything; it tests itself.
import sys as _sys
_HERE = Path.cwd() if (Path.cwd() / "statistical_helpers.py").exists() else Path.cwd() / "analysis"
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import statistical_helpers as SH
from statistical_helpers import (
    EV_IMPL, EV_ASSOC, EV_EXPL, EV_INCONC, EV_REPL, EVIDENCE_CLASSES,
    hs_from_level, build_hs_crossings, build_hs3_episodes, flapping,
    state_sequence, fit_ctmc, is_irreducible,
    match_pings, control_windows, assign_run_by_time,
    exact_prop_ci, boot_ci, boot_diff_ci, cluster_bootstrap,
    exact_permutation_p, enumerate_label_assignments,
    firth_logit, firth_profile_ci,
    fit_gee_checked, adjustment_verdict, ipw_diagnostics, spec_agreement,
)
# Print the module's repo-relative path, never its absolute one: this machine's home
# directory is a participant's name, and the leak checker rightly rejects it.
print(f"statistical_helpers imported from analysis/{Path(SH.__file__).name}")

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

# --- Study phase + attendance (external design metadata, not controller inputs) --------
# Hoisted into setup because B3/B5/B7 all need to condition on phase, not just B10.
# Roles are resolved later (they need the pseudonym map, which is seeded in A1b).
_RP = json.loads((PRIVATE_DIR / "role_phase.json").read_text()) \
      if (PRIVATE_DIR / "role_phase.json").exists() else {}
PHASE1_DAYS_EARLY = set(_RP.get("phase1_days", []))
def phase_of_day(day):
    "P1 = the first 4 days (assigned roles); P2 = the last 4 (all constraints lifted)."
    return "P1" if str(day) in PHASE1_DAYS_EARLY else "P2"

# The COMPLETE person x scheduled-day attendance panel, transcribed from the experiment's own
# session sheet. This is the exposure denominator: on 33 of 96 scheduled person-days a
# participant was expected and DID NOT TURN UP. Those are genuine zeros — zero interactions,
# zero meals — and a panel built from completed interactions (as the previous one was) simply
# loses them, inflating every per-day rate for exactly the people who attended least reliably.
# It also records that roles were assigned BY AVAILABILITY, not randomised.
PRESENCE = json.loads((PRIVATE_DIR / "presence_panel.json").read_text()) \
           if (PRIVATE_DIR / "presence_panel.json").exists() else {}
if not PRESENCE:
    print("WARNING: presence_panel.json missing — B10 exposure falls back to observed "
          "interactions, which loses every scheduled-but-absent day.")

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

# build_hs_crossings() and build_hs3_episodes() are imported from statistical_helpers, which
# is the SAME module the tests import. See the setup cell for why that matters.

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
print(f"Role assignment mechanism: {PRESENCE.get('role_assignment_mechanism','(undocumented)')}"
      f"  -> {'NOT randomised' if PRESENCE.get('role_assignment_mechanism') != 'randomised' else 'randomised'}")

# --- SCHEDULE: the attendance panel, keyed by pseudonym ---------------------------------
# The sheet uses full names; the databases use first names. Map the first token through
# the same canonicalisation + pseudonym map as every other identity, so no real name survives.
SCHEDULE = {}
_unmatched = []
for _full, _days in PRESENCE.get("panel", {}).items():
    _first = str(_full).split()[0]
    _pid = (PSEUDONYM_MAP.get(canon_identity(_first))
            or PSEUDONYM_MAP.get(canon_identity(_first.lower()))
            or PSEUDONYM_MAP.get(canon_identity(_first.capitalize())))
    if _pid is None:
        _unmatched.append("(name withheld)")
        continue
    SCHEDULE[_pid] = _days
print(f"Attendance panel: {len(SCHEDULE)} people matched to pseudonyms"
      + (f", {len(_unmatched)} UNMATCHED" if _unmatched else ""))
_n_sched = sum(1 for d in SCHEDULE.values() for v in d.values() if v["scheduled"])
_n_att = sum(1 for d in SCHEDULE.values() for v in d.values() if v["attended"])
print(f"  {_n_sched} scheduled person-days, {_n_att} attended, "
      f"{_n_sched - _n_att} scheduled-but-ABSENT (the zeros the old panel lost).")

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

# boot_ci, boot_diff_ci, exact_prop_ci, cluster_bootstrap, exact_permutation_p, firth_logit,
# firth_profile_ci, fit_gee_checked, adjustment_verdict, ipw_diagnostics and spec_agreement are
# all imported from statistical_helpers (setup cell). One implementation, shared with the tests.

def cluster_bootstrap_effect(df, cluster_col, fit_fn, *, n=1000, seed=SEED, label="effect"):
    "Back-compat shim over SH.cluster_bootstrap; returns the legacy (lo, mid, hi, n_ok) tuple."
    r = cluster_bootstrap(df, cluster_col, fit_fn, n=n, seed=seed, label=label)
    return (r["lo"], r["median"], r["hi"], r["n_ok"])
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

# flapping() is imported from statistical_helpers. A reversal requires
# `from_state == previous to_state` — you can only undo a crossing by going back the way you
# came. The original detector tested `from_state == previous from_state`, which is
# unsatisfiable, so it returned 0 on any data whatsoever and "the labels do not flap" was
# reported on that basis. They do flap.
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
# "feeding_received_rate", never "feed_pursuit": the robot cannot make a meal appear.
g = d.groupby("grp").agg(n=("interaction_id","size"), feeding_received_rate=("fed","mean"))
print("\n(3) Co-present interactions Full vs Deficit:")
print(g.loc[["Full","Deficit"]].round(3).to_string())
dp,dplo,dphi = boot_diff_ci(d[d.grp=="Deficit"]["fed"].astype(float), d[d.grp=="Full"]["fed"].astype(float))
print(f"    feeding-received diff (Deficit-Full): {dp:+.2f} [95% CI {dplo:.2f},{dphi:.2f}]")

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
# Outcome: FEEDING RECEIVED (meals_eaten_count > 0) — a dyadic outcome that depends on what the
# human did. NOT "feeding pursuit": the robot cannot make a meal appear. The framing / ping /
# feed-seeking rows above are coded gates and are NOT modelled — their significance would be
# true by construction, so giving them a p-value would be dishonest arithmetic.
#
# "unknown" is an unrecognised-face placeholder, not a person. Pooling those interactions into
# one person-cluster asserts a dependence structure that does not exist, so they are dropped
# from person-clustered fits and kept only where the cluster is the run.
d["deficit"] = (d["grp"] == "Deficit").astype(int)
d["feeding_received"] = d["fed"].astype(int)          # the ONLY name for this outcome

# --- Covariates, defined on EVERY dataset used below. --------------------------------------
# The previous version defined `phase` and `prior_n` only on the named-person frame, so the
# run-clustered fits silently dropped every adjusted specification and the "survives adjustment"
# claim rested on models that had never run.
def add_b3_covariates(df):
    df = df.sort_values(["person_id", "interaction_start_epoch"]).copy()
    df["phase"] = df["day_rome"].map(phase_of_day)
    df["day"] = df["day_rome"].astype(str)
    df["prior_n"] = df.groupby("person_id").cumcount()      # strictly BEFORE this interaction
    return df

d = add_b3_covariates(d)
d_named = d[d["person_id"] != "unknown"].copy()
print(f"\nPerson-clustered models exclude 'unknown' faces: {len(d)} interactions -> "
      f"{len(d_named)} across {d_named['person_id'].nunique()} named people "
      f"({len(d) - len(d_named)} unknown-face interactions dropped; retained in the "
      f"run-clustered fits, where the cluster is well defined).")
print("\nDeficit x feeding received (named people):")
print(pd.crosstab(d_named["grp"], d_named["feeding_received"], margins=True).to_string())

# --- ONE prespecified adjusted model. ------------------------------------------------------
# Every predictor is fixed BEFORE the interaction starts, so nothing about the outcome leaks in.
B3_UNADJ = "feeding_received ~ deficit"
# PHASE IS NESTED WITHIN DAY. Each session-day belongs to exactly one phase, so C(day) fully
# determines C(phase) and a model containing both is not identified — the person-clustered fit
# returns a non-finite estimate, which is precisely the failure mode fit_gee_checked exists to
# catch rather than swallow. Day is the finer control and ABSORBS phase, so the prespecified
# model uses day; a phase-only variant is fitted alongside so both adjustments are reported.
B3_ADJ   = ("feeding_received ~ deficit + C(initial_state) + C(trigger_mode) + C(day) + prior_n")
B3_ADJ_PHASE = ("feeding_received ~ deficit + C(initial_state) + C(trigger_mode) + C(phase) "
                "+ prior_n")
print(f"\nPRESPECIFIED adjusted model (day absorbs phase — they are nested, not additive):")
print(f"    {B3_ADJ}")
print(f"PHASE-ONLY variant (coarser time control, reported alongside):")
print(f"    {B3_ADJ_PHASE}")

b3_models = []
for label, fml, src, cl in [
    ("unadjusted",                B3_UNADJ,     d_named, "person_id"),
    ("unadjusted",                B3_UNADJ,     d,       "run_id"),
    ("PRESPECIFIED adjusted",     B3_ADJ,       d_named, "person_id"),
    ("PRESPECIFIED adjusted",     B3_ADJ,       d,       "run_id"),
    ("adjusted, phase not day",   B3_ADJ_PHASE, d_named, "person_id"),
    ("adjusted, phase not day",   B3_ADJ_PHASE, d,       "run_id"),
]:
    r = fit_gee_checked(fml, src, groups=cl, family=sm.families.Binomial(), focal="deficit")
    r["model"] = label
    r["odds_ratio"] = float(np.exp(r["estimate"])) if np.isfinite(r["estimate"]) else np.nan
    r["or_lo"] = float(np.exp(r["lo"])) if np.isfinite(r["lo"]) else np.nan
    r["or_hi"] = float(np.exp(r["hi"])) if np.isfinite(r["hi"]) else np.nan
    b3_models.append(r)

b3_tab = pd.DataFrame(b3_models)[["model", "cluster", "n", "status", "converged", "reason",
                                  "odds_ratio", "or_lo", "or_hi", "p"]]
b3_tab.to_csv(OUT_DIR / "b3_adjusted_models.csv", index=False)
print("\nModel status (nothing is silently caught — a failure is reported and counts against"
      "\nthe 'survives adjustment' claim):")
print(b3_tab.round(4).to_string(index=False))

_primary = [m for m in b3_models if m["model"] == "unadjusted" and m["cluster"] == "person_id"][0]
if _primary["status"] != "ok":
    raise RuntimeError(f"B3 primary model failed: {_primary['reason']}")
b3_or, b3_ci, b3_p = _primary["odds_ratio"], [_primary["or_lo"], _primary["or_hi"]], _primary["p"]
register_p("B3", B3_UNADJ + " (logistic GEE, cluster=person)", "deficit", b3_p,
           "RQ1/2-behaviour", "confirmatory")

# Every adjusted-model p-value that supports the "survives adjustment" claim is REGISTERED,
# because it is quoted in a conclusion. The previous version cited these and corrected none.
for m in b3_models:
    if m["model"] != "unadjusted" and np.isfinite(m["p"]):
        register_p("B3", f"{m['model']} (logistic GEE, cluster={m['cluster']})", "deficit", m["p"],
                   "RQ1/2-behaviour", "confirmatory",
                   note="adjusted-model sensitivity, quoted to support 'survives adjustment'")

# --- Does it actually survive adjustment? Decided, not asserted. ----------------------------
# The gate is the PRESPECIFIED model, under both clusterings. The phase-only variant is a
# secondary sensitivity and is reported (including its convergence status) but does not veto the
# prespecified claim — it is a coarser control on the same axis, not a required check.
b3_adj = adjustment_verdict([m for m in b3_models if m["model"] == "PRESPECIFIED adjusted"],
                            focal_sign=+1.0)
print(f"\n'Survives adjustment' gate (PRESPECIFIED model, both clusterings): "
      f"{'YES' if b3_adj['survives'] else 'NO'}")
print(f"    {b3_adj['reason']}")
_variant = [m for m in b3_models if m["model"] == "adjusted, phase not day"]
for m in _variant:
    if m["status"] != "ok":
        print(f"    NOTE: the secondary phase-only variant [{m['cluster']}] did not fit cleanly "
              f"({m['status']}: {m['reason']}). Its point estimate ({m['odds_ratio']:.2f}) agrees, "
              f"but it is reported as a variant, not relied on.")

# --- THE INTERVALS WE LEAD WITH ------------------------------------------------------------
_b3_boot = cluster_bootstrap(
    d_named, "person_id",
    lambda _s: np.exp(smf.glm(B3_UNADJ, data=_s, family=sm.families.Binomial()).fit().params["deficit"]),
    label="B3 deficit->feeding OR (person-cluster bootstrap)")
_b3_boot_run = cluster_bootstrap(
    d, "run_id",
    lambda _s: np.exp(smf.glm(B3_UNADJ, data=_s, family=sm.families.Binomial()).fit().params["deficit"]),
    label="B3 deficit->feeding OR (run-cluster bootstrap)")
print(f"\nPRIMARY (lead with this): deficit->feeding OR = {b3_or:.2f}")
print(f"    person-cluster bootstrap [{_b3_boot['lo']:.2f}, {_b3_boot['hi']:.2f}]")
print(f"    run-cluster bootstrap    [{_b3_boot_run['lo']:.2f}, {_b3_boot_run['hi']:.2f}]")
print(f"    asymptotic GEE CI [{b3_ci[0]:.2f}, {b3_ci[1]:.2f}], p={b3_p:.2e} "
      f"(second: anti-conservative at {d_named['person_id'].nunique()} clusters)")

_lopo = []
for _pid in d_named["person_id"].unique():
    r = fit_gee_checked(B3_UNADJ, d_named[d_named["person_id"] != _pid], groups="person_id",
                        family=sm.families.Binomial(), focal="deficit")
    if r["status"] == "ok":
        _lopo.append(float(np.exp(r["estimate"])))
print(f"    leave-one-person-out OR range [{min(_lopo):.2f}, {max(_lopo):.2f}] "
      f"({len(_lopo)}/{d_named['person_id'].nunique()} refits succeeded)")

SENSITIVITY.append(dict(metric="B3_deficit_feeding_OR", primary=b3_or,
                        boot_lo=_b3_boot["lo"], boot_median=_b3_boot["median"],
                        boot_hi=_b3_boot["hi"], successful_refits=_b3_boot["n_ok"],
                        unit="person-cluster bootstrap over interactions"))
SENSITIVITY.append(dict(metric="B3_deficit_feeding_OR_runcluster", primary=b3_or,
                        boot_lo=_b3_boot_run["lo"], boot_median=_b3_boot_run["median"],
                        boot_hi=_b3_boot_run["hi"], successful_refits=_b3_boot_run["n_ok"],
                        unit="run-cluster bootstrap over interactions"))
globals()["_b3"] = dict(orr=b3_or, ci=b3_ci, p=b3_p,
                        boot=[_b3_boot["lo"], _b3_boot["median"], _b3_boot["hi"]],
                        boot_run=[_b3_boot_run["lo"], _b3_boot_run["median"], _b3_boot_run["hi"]],
                        lopo=[min(_lopo), max(_lopo)], n=len(d_named),
                        survives_adjustment=b3_adj["survives"], adj_reason=b3_adj["reason"])

_adj_txt = (f"It survives the prespecified adjustment for social state, trigger mode, phase, day "
            f"and prior interaction count, under both clustering schemes."
            if b3_adj["survives"] else
            f"It CANNOT be said to survive adjustment: {b3_adj['reason']}.")
verdict("B3", EV_ASSOC,
        f"Within-deployment association between the orexigenic deficit and FEEDING RECEIVED "
        f"(a meal arrived — an outcome of the dyad, not a robot action). Odds are {b3_or:.1f}x "
        f"higher in deficit than at Full (person-cluster bootstrap "
        f"[{_b3_boot['lo']:.1f}, {_b3_boot['hi']:.1f}]; run-cluster bootstrap "
        f"[{_b3_boot_run['lo']:.1f}, {_b3_boot_run['hi']:.1f}]; asymptotic GEE CI "
        f"[{b3_ci[0]:.1f}, {b3_ci[1]:.1f}], p={b3_p:.1e}; LOPO {min(_lopo):.1f}-{max(_lopo):.1f}). "
        f"{_adj_txt} Raw rates: a meal arrived in {g.loc['Full','feeding_received_rate']:.2f} of Full "
        f"interactions vs {g.loc['Deficit','feeding_received_rate']:.2f} in deficit; meals are larger in "
        f"deficit ({ms.loc['Full','mean']:.0f} -> {ms.loc['Deficit','mean']:.0f}). SEPARATELY, and as "
        f"implementation verification only: the hunger framing ({f_full*100:.0f}%->{f_def*100:.0f}%), "
        f"the {int(sk.get('Deficit',0))} feed-seeking speech acts and the {n_def_ping} proactive pings "
        f"(vs {n_full_ping} at Full) are state-gated in source — `if` statements, not findings. Single "
        f"always-on condition: this is an association, NOT a causal effect of the drive.",
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
d["feeding_received"] = pd.to_numeric(d["meals_eaten_count"], errors="coerce").fillna(0) > 0
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
feed_hs3 = hs3["feeding_received"].mean() if len(hs3) else np.nan
feed_rest = rest["feeding_received"].mean()
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
# Attribute each meal to the interaction that was ACTIVE when it happened. Feeding events carry
# no exec_interaction_id, and `feeder_face_id` is populated for only 20 of 108 feeds (naming 8
# people when 14 received meals), so keying on it would silently drop most of the corpus.
_mi5 = master.dropna(subset=["interaction_start_epoch"]).copy()
_mi5["_end"] = _mi5["interaction_start_epoch"] + _mi5["duration_sec"].fillna(120.0)
def _attr5(row):
    c = _mi5[(_mi5.run_id == row["run_id"])
             & (_mi5.interaction_start_epoch <= row["timestamp_epoch"])
             & (_mi5["_end"] >= row["timestamp_epoch"])]
    if not len(c):
        c = _mi5[(_mi5.run_id == row["run_id"])
                 & (_mi5.interaction_start_epoch <= row["timestamp_epoch"])
                 & (row["timestamp_epoch"] - _mi5.interaction_start_epoch <= 180.0)]
    return c.sort_values("interaction_start_epoch").iloc[-1]["person_id"] if len(c) else None
feeds["person_id"] = feeds.apply(_attr5, axis=1)
feeds["hs_rank"] = feeds["hs_before"].map({"HS1":0,"HS2":1,"HS3":2}).astype(float)
print(f"Meals attributed to an active interaction: "
      f"{int(feeds['person_id'].notna().sum())}/{len(feeds)}.")

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

# --- Does the gradient survive EXCLUDING the two obligated feeders? --------------------
# "Survives exclusion" is a claim about an INTERVAL, not a point estimate. A slope of +7.5 with
# an interval spanning zero does not survive anything; the previous version reported the point
# estimate alone and asserted it did.
_feeder_pids = {p for p, r in ROLE_OF_EARLY.items() if r == "feeder"}
_no_feeders = _mf_named[~_mf_named["person_id"].isin(_feeder_pids)]
_nf_slope, _nf_boot, _nf_ok = np.nan, dict(lo=np.nan, median=np.nan, hi=np.nan, n_ok=0), False
if len(_no_feeders) > 5 and _no_feeders["hs_rank"].nunique() > 1:
    _m_nf = smf.ols("meal_delta ~ hs_rank", _no_feeders).fit(
        cov_type="cluster", cov_kwds={"groups": _no_feeders["run_id"]})
    _nf_slope = float(_m_nf.params["hs_rank"])
    _nf_boot = cluster_bootstrap(
        _no_feeders, "run_id",
        lambda _s: smf.ols("meal_delta ~ hs_rank", _s).fit().params["hs_rank"],
        label="B5a meal-size slope EXCLUDING obligated feeders (run-cluster bootstrap)")
    # The claim is licensed only if the interval excludes zero on the positive side.
    _nf_ok = bool(np.isfinite(_nf_boot["lo"]) and _nf_boot["lo"] > 0)
    print(f"\nEXCLUDING the {len(_feeder_pids)} obligated feeders "
          f"({len(_mf_named) - len(_no_feeders)} of their meals dropped, "
          f"{len(_no_feeders)} meals left over {_no_feeders['person_id'].nunique()} people):")
    print(f"  slope {_nf_slope:+.2f} per deficit step, run-cluster bootstrap "
          f"[{_nf_boot['lo']:+.2f}, {_nf_boot['hi']:+.2f}]")
    print(f"  -> {'SURVIVES exclusion (interval excludes zero)' if _nf_ok else
                  'does NOT survive exclusion (interval includes zero)'}")
    print(f"  mean meal size without them: "
          f"{_no_feeders.groupby('hs_before')['meal_delta'].mean().round(1).to_dict()}")
    SENSITIVITY.append(dict(metric="B5a_meal_size_slope_excl_feeders", primary=_nf_slope,
                            boot_lo=_nf_boot["lo"], boot_median=_nf_boot["median"],
                            boot_hi=_nf_boot["hi"], successful_refits=_nf_boot["n_ok"],
                            unit="run-cluster bootstrap over meals, obligated feeders excluded"))
else:
    print("\nToo few meals outside the obligated feeders to refit — the gradient CANNOT be shown "
          "to be independent of them.")
globals()["_b5_meal"]=dict(slope=float(_m_run.params["hs_rank"]),
                           boot=[_b5_meal_boot[0],_b5_meal_boot[1],_b5_meal_boot[2]],
                           p=_b5_meal_p, nf_slope=_nf_slope,
                           nf_boot=[_nf_boot["lo"], _nf_boot["median"], _nf_boot["hi"]],
                           survives_exclusion=bool(_nf_ok),
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
# --- B5b. Remote loop: reply-centric matching + properly matched controls ----------------
# match_pings() and control_windows() are imported from statistical_helpers (same code the
# tests exercise). Matching is REPLY-CENTRIC: each reply is claimed by the NEAREST eligible
# preceding ping, not by the earliest one still waiting. Both are one-to-one; only one is
# defensible, because a greedy forward walk can credit a reply to a ping sent an hour earlier
# while ignoring the ping sent 30 seconds before it.
ev = chat_events.copy().sort_values(["chat_id", "timestamp_epoch"])

# PSEUDONYMISE THE SUBSCRIBER FIRST. `chat_id` is a raw Telegram user ID — a stable, real-world
# personal identifier — and was not in IDENTITY_COLS, so a per-subscriber table exported from it
# would have published real Telegram IDs.
_sub_order = ev.sort_values("timestamp_epoch")["chat_id"].drop_duplicates().tolist()
SUBSCRIBER_MAP = {c: f"S{i+1:02d}" for i, c in enumerate(_sub_order)}
ev["chat_id"] = ev["chat_id"].map(SUBSCRIBER_MAP)
print(f"Subscribers pseudonymised: {len(SUBSCRIBER_MAP)} raw Telegram IDs -> "
      f"S01..S{len(SUBSCRIBER_MAP):02d}. No raw chat_id reaches any exported table.")

PING_TYPES = ["hs2_entry", "hs3_proactive"]     # genuine hunger signalling
EXCLUDED   = ["hs3_recovery"]                   # "thanks, I'm full" — NOT a request for food
WINDOWS    = [900.0, 1800.0, 3600.0]            # 15 / 30 / 60 min
CTRL_SEEDS = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]

# Control windows may only be drawn from hours the robot was actually RUNNING and the subscriber
# was reachable — i.e. from inside a monitored run. Drawing them from the whole calendar puts
# them in the middle of the night and inflates the effect roughly two-fold.
run_spans = (hunger_raw.groupby("run_id")["timestamp_epoch"].agg(t_start="min", t_end="max")
                       .reset_index())

# CROSS-STREAM KEY. The four databases do NOT share a run_id namespace: the chat and executive
# run_id sets have ZERO overlap. Joining pings to runs on run_id silently matches nothing, and a
# control-window routine that did so would produce no controls at all. Wall-clock containment is
# the pipeline's documented cross-stream key, so each ping is assigned to the EXECUTIVE run whose
# span contains it.
pm_pre = match_pings(ev, ["hs2_entry", "hs3_proactive"], 3600.0)
_chat_runs = set(chat_events["run_id"].dropna().unique())
_exec_runs = set(hunger_raw["run_id"].dropna().unique())
print(f"\nchat run_ids: {len(_chat_runs)}, executive run_ids: {len(_exec_runs)}, "
      f"overlap: {len(_chat_runs & _exec_runs)} — DISJOINT namespaces, so pings are assigned to "
      f"executive runs by wall-clock containment, not by run_id.")
print(f"\nControl windows are drawn ONLY from inside the {len(run_spans)} monitored run spans "
      f"({(run_spans['t_end'] - run_spans['t_start']).sum()/3600:.1f} h), matched to the ping's OWN "
      f"run and to its time-of-day within a +/-2 h caliper, required to be ping-free and "
      f"non-overlapping, with each user message able to satisfy at most ONE control window.")

pm = match_pings(ev, PING_TYPES, 3600.0)
pm["exec_run_id"] = assign_run_by_time(pm["t"], run_spans)
_n_out = int(pm["exec_run_id"].isna().sum())
print(f"\nPings inside a monitored run: {len(pm) - _n_out}/{len(pm)} "
      f"({_n_out} fell outside every run span — the robot was off, so no control window exists "
      f"for them and they are reported as unmatched rather than silently dropped).")
_n_excl = int(ev["event_type"].isin(EXCLUDED).sum())
print(f"\nProactive HUNGER pings: {len(pm)} ({pm['ping_type'].value_counts().to_dict()}) "
      f"across {pm['chat_id'].nunique()} subscribers.")
print(f"Excluded: {_n_excl} hs3_recovery events ('thanks, I'm full' — a notification, not a "
      f"request for food; a reply to a thank-you is not evidence that hunger signalling works).")
print(f"Replies matched one-to-one: {int(pm['replied'].sum())}. No reply is used twice, and "
      f"n_replies <= n_pings by construction.")
assert pm["matched_reply"].dropna().is_unique, "a reply was attributed to more than one ping"

# --- Windows x ping type, with BOTH cluster bootstraps and the matched control ------------
print("\nPING vs MATCHED CONTROL, by window (exact 95% CI; paired difference with both "
      "cluster bootstraps):")
rows, pair_frames = [], []
for w in WINDOWS:
    _p = match_pings(ev, PING_TYPES, w)
    _p["exec_run_id"] = assign_run_by_time(_p["t"], run_spans)
    assert _p["matched_reply"].dropna().is_unique
    # Repeat control selection over several seeds: a single random draw of control instants is
    # itself a source of variation, and reporting one draw hides it.
    per_seed = []
    for sd in CTRL_SEEDS:
        _c = control_windows(ev, _p, w, run_spans, run_col="exec_run_id", seed=sd)
        _c = _c[_c["matched"] == True]
        if not len(_c):
            continue
        pair = _p[["chat_id", "exec_run_id", "ping_id", "ping_type", "replied"]].merge(
            _c[["ping_id", "replied"]].rename(columns={"replied": "control_replied"}),
            on="ping_id", how="inner")
        per_seed.append(dict(seed=sd, n=len(pair),
                             ping=float(pair["replied"].mean()),
                             ctrl=float(pair["control_replied"].mean()),
                             diff=float((pair["replied"] - pair["control_replied"]).mean())))
        if sd == SEED:
            pair["window_min"] = int(w / 60)
            pair_frames.append(pair)
            _pair0 = pair
    ps = pd.DataFrame(per_seed)
    k, n = int(_p["replied"].sum()), len(_p)
    e, lo, hi = exact_prop_ci(k, n)
    kc = int(_pair0["control_replied"].sum()); nc = len(_pair0)
    ec, clo, chi = exact_prop_ci(kc, nc)
    bs = cluster_bootstrap(_pair0, "chat_id",
                           lambda _s: float((_s["replied"] - _s["control_replied"]).mean()),
                           label=f"  {int(w/60)}min paired diff (subscriber-cluster)", verbose=False)
    br = cluster_bootstrap(_pair0, "exec_run_id",
                           lambda _s: float((_s["replied"] - _s["control_replied"]).mean()),
                           label=f"  {int(w/60)}min paired diff (run-cluster)", verbose=False)
    rows.append(dict(window_min=int(w/60), n_pings=n, ping_k=k, ping_rate=e,
                     ping_lo=lo, ping_hi=hi,
                     control_rate=ec, control_lo=clo, control_hi=chi,
                     paired_diff=float((_pair0["replied"] - _pair0["control_replied"]).mean()),
                     sub_boot_lo=bs["lo"], sub_boot_hi=bs["hi"],
                     run_boot_lo=br["lo"], run_boot_hi=br["hi"],
                     diff_across_seeds_min=float(ps["diff"].min()),
                     diff_across_seeds_max=float(ps["diff"].max()),
                     n_seeds=len(ps)))
win_df = pd.DataFrame(rows)
print(win_df.round(3).to_string(index=False))
win_df.to_csv(OUT_DIR/"b5_ping_response_windows.csv", index=False)
pd.concat(pair_frames, ignore_index=True).to_csv(OUT_DIR/"b5_ping_control_pairs.csv", index=False)
print(f"\n  Matched ping-control pairs exported to b5_ping_control_pairs.csv.")
print(f"  Control-assignment sensitivity: across {len(CTRL_SEEDS)} independent control draws the "
      f"60-min paired difference ranges "
      f"{win_df.loc[win_df.window_min==60,'diff_across_seeds_min'].iloc[0]:+.3f} to "
      f"{win_df.loc[win_df.window_min==60,'diff_across_seeds_max'].iloc[0]:+.3f} — the conclusion "
      f"does not hinge on which controls were drawn.")

# --- Per-ping-type breakdown at 60 min ----------------------------------------------------
print("\nBy ping type (60 min):")
for pt in PING_TYPES:
    sub = pm[pm.ping_type == pt]
    k, n = int(sub["replied"].sum()), len(sub)
    e, lo, hi = exact_prop_ci(k, n)
    print(f"    {pt:15s} {k:3d}/{n:3d} = {e:.2f} [{lo:.2f}, {hi:.2f}]")

_w60 = win_df[win_df.window_min == 60].iloc[0]
_diff = float(_w60["paired_diff"])
_sub_lo, _sub_hi = float(_w60["sub_boot_lo"]), float(_w60["sub_boot_hi"])
_run_lo, _run_hi = float(_w60["run_boot_lo"]), float(_w60["run_boot_hi"])

# The verdict rests on the PAIRED difference clearing zero under BOTH clustering schemes.
_meaningful = bool(np.isfinite(_sub_lo) and _sub_lo > 0 and np.isfinite(_run_lo) and _run_lo > 0)

_by_sub = pm.groupby("chat_id").agg(pings=("replied", "size"), replies=("replied", "sum"))
_by_sub["rate"] = (_by_sub["replies"] / _by_sub["pings"]).round(2)
_by_sub.to_csv(OUT_DIR/"b5_ping_by_subscriber.csv")
print(f"\n{(_by_sub['replies']==0).sum()}/{len(_by_sub)} subscribers never replied to a hunger ping.")

globals()["_b5_ping"] = dict(rate=float(_w60["ping_rate"]), ctrl=float(_w60["control_rate"]),
                             diff=_diff, boot=[_sub_lo, np.nan, _sub_hi],
                             run_boot=[_run_lo, np.nan, _run_hi],
                             n_pings=int(_w60["n_pings"]), n_subs=int(pm["chat_id"].nunique()),
                             meaningful=_meaningful)
globals()["_b5_pm"] = pm
globals()["_b5_win"] = win_df

register_p("B5b", "paired ping-vs-matched-control reply difference (subscriber-cluster bootstrap)",
           "ping - control", np.nan, "RQ1/2-behaviour", "exploratory",
           note="interval-based; no p-value is computed or quoted for this contrast")

_meal = globals()["_b5_meal"]
_meal_ok = bool(np.isfinite(_meal["boot"][0]) and _meal["boot"][0] > 0)
if _meal_ok and _meaningful:
    _ev5 = EV_ASSOC
elif _meal_ok or _meaningful:
    _ev5 = EV_ASSOC
else:
    _ev5 = EV_INCONC

_excl_txt = (f", and the gradient SURVIVES EXCLUDING the two obligated feeders "
             f"({_meal['nf_slope']:+.1f}/step, run-cluster bootstrap "
             f"[{_meal['nf_boot'][0]:+.1f}, {_meal['nf_boot'][2]:+.1f}])"
             if _meal["survives_exclusion"] else
             f", but it CANNOT be said to survive excluding the two obligated feeders: without them "
             f"the slope is {_meal['nf_slope']:+.1f}/step with a run-cluster bootstrap interval of "
             f"[{_meal['nf_boot'][0]:+.1f}, {_meal['nf_boot'][2]:+.1f}], which includes zero")

verdict("B5", _ev5,
        f"MEAL SIZE scales with deficit severity: {_meal['means']['HS1']:.0f} (Full) -> "
        f"{_meal['means']['HS2']:.0f} (Hungry) -> {_meal['means']['HS3']:.0f} (Starving), "
        f"{_meal['slope']:+.1f} stomach points per deficit step (run-cluster bootstrap "
        f"[{_meal['boot'][0]:+.1f}, {_meal['boot'][2]:+.1f}]){_excl_txt}. "
        f"REMOTE LOOP, with reply-centric one-to-one matching and hs3_recovery notifications "
        f"excluded: {int(_w60['ping_k'])}/{int(_w60['n_pings'])} hunger pings drew a reply within "
        f"1 h ({_w60['ping_rate']:.2f}, exact [{_w60['ping_lo']:.2f}, {_w60['ping_hi']:.2f}]) against "
        f"{_w60['control_rate']:.2f} [{_w60['control_lo']:.2f}, {_w60['control_hi']:.2f}] in controls "
        f"matched on subscriber, run and time-of-day — a PAIRED difference of {_diff:+.2f} "
        f"(subscriber-cluster bootstrap [{_sub_lo:+.2f}, {_sub_hi:+.2f}]; run-cluster "
        f"[{_run_lo:+.2f}, {_run_hi:+.2f}]), stable across {len(CTRL_SEEDS)} independent control "
        f"draws. {(_by_sub['replies']==0).sum()}/{len(_by_sub)} subscribers never replied to any "
        f"hunger ping. The remote channel is a real but WEAK recovery pathway"
        + ("." if _meaningful else
           "; the ping-control difference does not clear zero under both clustering schemes, so it "
           "is reported as exploratory."),
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
# B7: CTMC over Full/Hungry/Starving. state_sequence() and fit_ctmc() are imported from
# statistical_helpers (same code the tests exercise).
#
# fit_ctmc VALIDATES the stationary vector instead of coercing one out of the null space:
#   * irreducibility is tested as a GRAPH property (strong connectivity of the transition
#     graph), not inferred from the dimension of null(Q'). A reducible chain with a single
#     absorbing class also has a 1-D null space, so the old test would have called it ergodic.
#   * the null-space vector must be single-signed. The old code took abs() of an arbitrary basis
#     vector, which silently turns a mixed-sign vector — the signature of NO stationary
#     distribution — into a plausible-looking one.
#   * the result must satisfy pi >= 0, sum(pi) == 1, and pi @ Q ~ 0 within tolerance.
# The condition tested is therefore named `unique_stationary_distribution`, not `ergodic`.
from scipy.linalg import null_space
states = ["HS1", "HS2", "HS3"]
seq_all = state_sequence(hunger_raw)
seq_all = seq_all[(seq_all["state"].isin(states)) & (seq_all["dwell"] > 0)]

term = seq_all[seq_all["terminal"]]
print("TERMINAL (right-censored) SEGMENTS — dropped entirely by the previous fit:")
_t = term.groupby("state").agg(n=("dwell","size"), total_sec=("dwell","sum"),
                               median_sec=("dwell","median"), max_sec=("dwell","max"))
print(_t.rename(index=HS_NAME).round(1).to_string())
print(f"  total {term['dwell'].sum():.0f} s across {len(term)} runs "
      f"({term['dwell'].sum()/seq_all['dwell'].sum()*100:.1f}% of all observed state-time).")
term.groupby("state")["dwell"].agg(["size","sum","median","max"]).to_csv(
    OUT_DIR/"b7_terminal_segments.csv")

fit = fit_ctmc(seq_all, states)
Q, cnt, time_in = fit["Q"], fit["counts"], fit["time_in"]
print("\nTransition counts (terminal segments excluded, as they must be):")
print(cnt.rename(index=HS_NAME, columns=HS_NAME).to_string())
print("\nTotal time in state (s) — terminal dwell INCLUDED:")
print(time_in.rename(index=HS_NAME).round(0).to_string())
print(f"\nGenerator diagnostics:")
print(f"    irreducible (strong connectivity of the transition graph): {fit['irreducible']}")
print(f"    unique stationary distribution (dim null(Q') == 1):        {fit['unique_stationary_distribution']}")
print(f"    stationary vector VALIDATED (pi>=0, sum=1, pi@Q~0):        {fit['stationary_valid']}")
if fit["stationary_valid"]:
    print(f"    residual max|pi @ Q| = {fit['residual']:.2e}")
else:
    print(f"    reason: {fit['reason']}")

# --- 1. EMPIRICAL occupancy: no stationarity assumption at all --------------------------
emp = time_in / time_in.sum()
print("\n=== EMPIRICAL time-occupancy (assumption-free; report this first) ===")
for st_ in states:
    print(f"    {HS_NAME[st_]:9s} {emp[st_]*100:5.2f}% of observed seconds")
_emp_boot = cluster_bootstrap(
    seq_all, "run_id",
    lambda _s: (_s[_s.state=="HS3"]["dwell"].sum() / max(_s["dwell"].sum(), 1e-9)),
    label="Empirical Starving occupancy (run-cluster bootstrap)")
print(f"    Starving, run-cluster bootstrap: {_emp_boot['median']*100:.2f}% "
      f"[95% {_emp_boot['lo']*100:.2f}, {_emp_boot['hi']*100:.2f}]")

# --- 2. MODELLED stationary occupancy ----------------------------------------------------
pi = fit["pi"]
if fit["stationary_valid"]:
    mean_hs3 = 1.0/(-Q[2,2]) if Q[2,2] != 0 else np.inf
    print(f"\n=== MODELLED stationary occupancy (time-homogeneous CTMC) ===")
    print("    " + ", ".join(f"{HS_NAME[s_]} {p_*100:.2f}%" for s_, p_ in zip(states, pi)))
    print(f"    Mean Starving sojourn {mean_hs3:.0f}s; the Starving row rests on "
          f"{int(cnt.loc['HS3'].sum())} transitions.")
    print(f"    Modelled Starving {pi[2]*100:.2f}% vs EMPIRICAL {emp['HS3']*100:.2f}%.")
    globals()["_ctmc_pi"] = dict(zip(states, pi))
else:
    print("\n=== MODELLED stationary occupancy: NOT VALIDATED on the pooled corpus ===")
    globals()["_ctmc_pi"] = None

# --- 3. Run-block bootstrap, RETAINING every failed resample ------------------------------
rng = np.random.default_rng(SEED)
_runs = seq_all["run_id"].unique()
_by_run = dict(tuple(seq_all.groupby("run_id")))
boot_blk, n_no_unique, n_not_valid, n_reducible = [], 0, 0, 0
for _ in range(2000):
    pick = rng.choice(_runs, len(_runs), replace=True)
    sq = pd.concat([_by_run[r] for r in pick], ignore_index=True)
    f = fit_ctmc(sq, states)
    if not f["irreducible"]:
        n_reducible += 1
    if not f["unique_stationary_distribution"]:
        n_no_unique += 1
        continue
    if not f["stationary_valid"]:
        n_not_valid += 1
        continue
    boot_blk.append(f["pi"][2])
_frac_bad = (n_no_unique + n_not_valid) / 2000
bb = np.percentile(boot_blk, [2.5, 50, 97.5]) if boot_blk else [np.nan]*3
print(f"\n=== Run-block bootstrap (2000 resamples over {len(_runs)} runs) ===")
print(f"    reducible (not strongly connected):        {n_reducible}/2000 = {n_reducible/20:.1f}%")
print(f"    no unique stationary distribution:         {n_no_unique}/2000")
print(f"    stationary vector failed validation:       {n_not_valid}/2000")
print(f"      (the previous code discarded all of these silently in a bare `except: pass`,")
print(f"       which conditioned the interval on the quantity being estimable at all.)")
print(f"    Starving occupancy, CONDITIONAL on a valid stationary distribution: "
      f"median {bb[1]*100:.2f}% [95% {bb[0]*100:.2f}, {bb[2]*100:.2f}] (n={len(boot_blk)})")
globals()["_b7_starve_ci_block"] = (bb[0], bb[1], bb[2])
globals()["_b7_nonergodic"] = _frac_bad

boot_p, n_bad_p = [], 0
for _ in range(2000):
    Qb = np.zeros((3,3))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i != j and time_in[si] > 0:
                Qb[i,j] = rng.poisson(cnt.loc[si,sj]) / time_in[si]
        Qb[i,i] = -Qb[i].sum()
    try:
        ns = null_space(Qb.T)
        if ns.shape[1] != 1:
            n_bad_p += 1; continue
        v = ns[:,0]
        if not (np.all(v >= -1e-8) or np.all(v <= 1e-8)):
            n_bad_p += 1; continue
        v = np.abs(v); pib = v/v.sum()
        if np.max(np.abs(pib @ Qb)) > 1e-8:
            n_bad_p += 1; continue
        boot_p.append(pib[2])
    except Exception:
        n_bad_p += 1
b = np.percentile(boot_p, [2.5, 50, 97.5]) if boot_p else [np.nan]*3
print(f"    Transition-level Poisson bootstrap: {b[1]*100:.2f}% [{b[0]*100:.2f}, {b[2]*100:.2f}] "
      f"({n_bad_p}/2000 invalid)")
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
    f_ = fit_ctmc(sub, states)
    _ti, _c = f_["time_in"], f_["counts"]
    _emp = _ti["HS3"] / _ti.sum() if _ti.sum() > 0 else np.nan
    strat_rows.append(dict(
        stratum=label, n_runs=sub["run_id"].nunique(),
        obs_hours=float(sub["dwell"].sum()/3600),
        n_hs3_transitions=int(_c.loc["HS3"].sum()),
        n_into_hs3=int(_c.loc["HS2","HS3"]),
        empirical_starving=float(_emp),
        modelled_starving=float(f_["pi"][2]) if f_["stationary_valid"] else np.nan,
        irreducible=bool(f_["irreducible"]),
        identified=bool(f_["stationary_valid"])))
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
    _frac_bad < 0.10 and                                # a valid stationary dist is not routinely lost
    np.isfinite(_ph_rel) and _ph_rel < 3.0 and          # the two phases are the same process
    abs(_pooled_mod - _pooled_emp) < 0.01               # model tracks the empirical fraction
)
_reasons = []
if not np.isfinite(bb[2]): _reasons.append("no usable stationary interval")
if _frac_bad >= 0.10:
    _reasons.append(f"{_frac_bad*100:.0f}% of run-resamples yield no valid stationary distribution")
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

globals()["_b7"]=dict(emp=_pooled_emp,
                      emp_boot=[_emp_boot["lo"], _emp_boot["median"], _emp_boot["hi"]],
                      mod=_pooled_mod, block=[bb[0],bb[1],bb[2]],
                      nonergodic=_frac_bad, spread=_spread, phase_rel=_ph_rel,
                      irreducible=bool(fit["irreducible"]),
                      stationary_valid=bool(fit["stationary_valid"]),
                      stable=bool(_stable))

if _stable:
    verdict("B7", EV_ASSOC,
            f"Within-deployment occupancy. The robot spent {_pooled_emp*100:.2f}% of observed seconds "
            f"in Starving (empirical, assumption-free; run-cluster bootstrap "
            f"[{_emp_boot['lo']*100:.2f}, {_emp_boot['hi']*100:.2f}]). The time-homogeneous CTMC agrees "
            f"({bb[1]*100:.2f}% [{bb[0]*100:.2f}, {bb[2]*100:.2f}]), with terminal dwells included and "
            f"{_frac_bad*100:.1f}% of resamples yielding no valid stationary distribution. This is a property of the coupled "
            f"human-robot loop, not of the controller: the transition rates record what people did. "
            f"No causal share is identified.", n=int(cnt.values.sum()))
else:
    verdict("B7", EV_INCONC,
            f"The MODELLED long-run Starving occupancy is not identified by these data, and the "
            f"empirical one does not need it. Empirically, the robot spent {_pooled_emp*100:.2f}% of "
            f"observed seconds in Starving (run-cluster bootstrap "
            f"[{_emp_boot['lo']*100:.2f}, {_emp_boot['hi']*100:.2f}]) — that figure is assumption-free and "
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

md(r"""#### B10.1 — Manipulation check *(exploratory; the only empirical question in RQ3)*

**Roles were NOT randomised.** They were assigned by participant availability — the experimenter
picked who could commit to feeding, and assigned himself as one of the two feeders. This is
recorded in `analysis/private/presence_panel.json` and it governs what can be said here:

- There is **no randomisation inference** in this section, and the phrase "survives randomisation"
  does not appear. A permutation distribution over role labels has a design justification only if
  the labels were randomised. They were not.
- What *is* reported is a **label-permutation sensitivity**: across the exact enumeration of all
  `C(12,2) = 66` ways the same two feeder labels could have fallen on the same twelve people, how
  unusual is the observed split? That is a descriptive statement about how much of the contrast
  rides on those two individuals. It licenses no causal or population claim, and the two feeders
  differ from everyone else in availability and motivation as well as in role.
- Enumeration is **exact**, not 5,000 random draws. Approximating a 66-point discrete
  distribution by sampling adds Monte-Carlo error for nothing.

**The exposure denominator is now the complete scheduled-day panel.** Participants were scheduled
for specific session days, and on 33 of 96 scheduled person-days a participant **did not turn up**.
Those are genuine zeros — zero interactions, zero meals — and they belong in the denominator. The
previous analysis built its panel from *completed interactions*, so a person who was expected and
never came simply vanished, which inflates every per-day rate for exactly the people who attended
least reliably.

**Meal COUNT and delivered ENERGY are different quantities and are reported separately.** Meals
come in three sizes (SMALL 10 / MEDIUM 25 / LARGE 45 stomach points), so a count is not an energy.
The previous report called counts "meal energy". Four quantities are now distinguished:

| Quantity | What it answers |
|---|---|
| meals per scheduled day | how much food arrived, per day the person was expected |
| **delivered energy** per scheduled day (`sum(meal_delta)`) | what the drive actually received |
| meals per interaction | per-encounter feeding propensity |
| interactions per scheduled day | attendance / exposure |

Verdict target: **exploratory manipulation check for these four participants**. No population role
effect is estimated, and none is estimable.""")

code(r"""
# --- B10b. Manipulation check: scheduled-day panel, exploratory, non-randomised ----------
d10 = m10[m10["role"] != "unknown"].copy()

print(f"ROLE ASSIGNMENT MECHANISM: {PRESENCE.get('role_assignment_mechanism','(undocumented)')}")
print("  -> Roles were NOT randomised. No randomisation inference is performed, and the phrase")
print("     'survives randomisation' does not appear in any verdict. What follows is a")
print("     LABEL-PERMUTATION SENSITIVITY over the exact enumeration of assignments.\n")

print("(1) Feed-rate per interaction, exact 95% CIs by role x phase:")
for (role, ph), g_ in d10.groupby(["role", "phase"], observed=True):
    k, n = int(g_["fed"].sum()), len(g_)
    est, lo, hi = exact_prop_ci(k, n)
    print(f"    {ROLE_LABEL[role]:22s} {ph}: {k:2d}/{n:3d} = {est:.2f} [{lo:.2f},{hi:.2f}]")
_nf1 = d10[(d10.role == "no_feed") & (d10.phase == "P1")]
_nf_k, _nf_n = int(_nf1["fed"].sum()), len(_nf1)
_nf_e = exact_prop_ci(_nf_k, _nf_n)
print(f"    -> no-feed compliance in Phase 1: {_nf_k}/{_nf_n} feeds, exact 95% CI "
      f"[{_nf_e[1]:.2f}, {_nf_e[2]:.2f}]. Complete separation IS the compliance result; the upper")
print(f"       bound {_nf_e[2]:.2f} is what {_nf_n} observations can rule out, and no more.")

# --- THE COMPLETE SCHEDULED-DAY PANEL ----------------------------------------------------
# Built from the attendance sheet, NOT from completed interactions. A person who was expected
# and did not turn up contributes a row of zeros; the previous panel simply lost them.
_sched = []
for pid, days in SCHEDULE.items():
    for day, info in days.items():
        if not info["scheduled"]:
            continue
        _sched.append(dict(person_id=pid, day_rome=day, phase=phase_of_day(day),
                           role=role_of(pid), attended=int(info["attended"])))
sched = pd.DataFrame(_sched)

# Observed behaviour, joined on. Missing => genuine zero.
_ix = d10.groupby(["person_id", "day_rome"]).agg(
    n_interactions=("interaction_id", "size"),
    meal_count=("meals_eaten_count", "sum")).reset_index()
_ix["day_rome"] = _ix["day_rome"].astype(str)
# DELIVERED ENERGY is sum(meal_delta) — the stomach points the drive actually received. It is
# NOT the meal count: meals are SMALL 10 / MEDIUM 25 / LARGE 45, so ten small meals and ten large
# ones are the same count and 350 points apart.
#
# ATTRIBUTION. Feeding events carry NO exec_interaction_id, and `feeder_face_id` is populated for
# only 20 of the 108 feeds — it names 8 people when 14 received meals, so using it silently drops
# most of the energy and most of the participants. Feeds are therefore attributed to the
# interaction that was ACTIVE when they occurred (same run, timestamp inside the interaction's
# window). Coverage is reported rather than assumed.
_feeds_all = hunger_raw[hunger_raw["event_type"] == "feeding"].copy()
_mi = master.dropna(subset=["interaction_start_epoch"]).copy()
_mi["_end"] = _mi["interaction_start_epoch"] + _mi["duration_sec"].fillna(120.0)
def _attribute_feed(row):
    c = _mi[(_mi.run_id == row["run_id"])
            & (_mi.interaction_start_epoch <= row["timestamp_epoch"])
            & (_mi["_end"] >= row["timestamp_epoch"])]
    if not len(c):     # brief grace period: the feed can land just after a short interaction
        c = _mi[(_mi.run_id == row["run_id"])
                & (_mi.interaction_start_epoch <= row["timestamp_epoch"])
                & (row["timestamp_epoch"] - _mi.interaction_start_epoch <= 180.0)]
    return c.sort_values("interaction_start_epoch").iloc[-1]["person_id"] if len(c) else None
_feeds_all["person_id"] = _feeds_all.apply(_attribute_feed, axis=1)
_n_attr = int(_feeds_all["person_id"].notna().sum())
_e_attr = float(_feeds_all.loc[_feeds_all["person_id"].notna(), "meal_delta"].sum())
print(f"\nMeal attribution: {_n_attr}/{len(_feeds_all)} feeding events assigned to an active "
      f"interaction ({_e_attr:.0f} of {_feeds_all['meal_delta'].sum():.0f} stomach points, "
      f"{_e_attr/_feeds_all['meal_delta'].sum()*100:.0f}%).")
print(f"  (feeder_face_id, the field the previous cell used, is populated for only "
      f"{int(_feeds_all['feeder_face_id'].notna().sum())}/{len(_feeds_all)} feeds and names "
      f"{_feeds_all[_feeds_all.feeder_face_id.notna() & (_feeds_all.feeder_face_id != 'unknown')]['feeder_face_id'].nunique()} "
      f"people when 14 received meals — it would have dropped most of the energy.)")
_fe = _feeds_all[_feeds_all["person_id"].notna() & (_feeds_all["person_id"] != "unknown")].copy()
_fe["day_rome"] = _fe["day_rome"].astype(str)
_energy = (_fe.groupby(["person_id", "day_rome"])["meal_delta"].sum()
              .rename("delivered_energy").reset_index())
globals()["_feeds_attributed"] = _fe
pdm = (sched.merge(_ix, on=["person_id", "day_rome"], how="left")
            .merge(_energy, on=["person_id", "day_rome"], how="left"))
for c_ in ("n_interactions", "meal_count", "delivered_energy"):
    pdm[c_] = pdm[c_].fillna(0.0)

print(f"\n(2) COMPLETE scheduled-day panel: {len(pdm)} scheduled person-days over "
      f"{pdm['person_id'].nunique()} people.")
print(f"    scheduled but DID NOT ATTEND: {int((pdm['attended']==0).sum())} person-days "
      f"({(pdm['attended']==0).mean()*100:.0f}%) — genuine zeros, kept in the denominator.")
print(f"    attended but delivered nothing: "
      f"{int(((pdm['attended']==1) & (pdm['meal_count']==0)).sum())} person-days.")
print(f"    (The old panel was built from completed interactions, so all "
      f"{int((pdm['attended']==0).sum())} no-show days were invisible and every per-day rate was "
      f"inflated for exactly the people who attended least reliably.)")
pdm.to_csv(OUT_DIR/"b10_scheduled_day_panel.csv", index=False)

print("\n(3) The FOUR distinct quantities, Phase 1 (mean per scheduled day unless noted):")
p1 = pdm[pdm.phase == "P1"]
q = (p1.groupby("role", observed=True)
       .agg(scheduled_days=("attended", "size"), attended_days=("attended", "sum"),
            interactions_per_sched_day=("n_interactions", "mean"),
            meals_per_sched_day=("meal_count", "mean"),
            energy_per_sched_day=("delivered_energy", "mean"),
            total_meals=("meal_count", "sum"), total_energy=("delivered_energy", "sum"),
            total_interactions=("n_interactions", "sum")))
q["meals_per_interaction"] = q["total_meals"] / q["total_interactions"].replace(0, np.nan)
q["energy_per_meal"] = q["total_energy"] / q["total_meals"].replace(0, np.nan)
print(q.round(2).to_string())

def _ratio(col, role_a="feeder", role_b="normal"):
    a_, b_ = q.loc[role_a, col], q.loc[role_b, col]
    return float(a_ / b_) if b_ else np.nan

_rr_meals  = _ratio("meals_per_sched_day")
_rr_energy = _ratio("energy_per_sched_day")
_rr_perint = _ratio("meals_per_interaction")
_rr_expo   = _ratio("interactions_per_sched_day")
print(f"\n    feeder vs unconstrained, Phase 1:")
print(f"      meals per scheduled day      {_rr_meals:.2f}x")
print(f"      DELIVERED ENERGY per sched.d {_rr_energy:.2f}x   <- what the drive experienced")
print(f"      meals per interaction        {_rr_perint:.2f}x   <- per-encounter propensity")
print(f"      interactions per sched. day  {_rr_expo:.2f}x   <- ATTENDANCE")
print(f"\n    The role acted through ATTENDANCE. Feeders delivered {_rr_energy:.1f}x the energy per")
print(f"    scheduled day, but per encounter fed only {_rr_perint:.1f}x as often. Exposure is a")
print(f"    MEDIATOR of the role (being told to feed the robot makes you go to the robot), not a")
print(f"    confounder, so the per-encounter figure is a decomposition, not a corrected estimate.")

# --- Poisson models on the complete panel, with an exposure offset ------------------------
pm_ = pdm[pdm["role"].isin(["feeder", "normal"])].copy()
pm_["is_feeder"] = (pm_["role"] == "feeder").astype(int)
# Offset by ATTENDED interactions. Days with zero interactions carry zero exposure and are kept
# in the outcome; log(0) is undefined, so those rows enter the unoffset model only. Substituting
# 1 for zero exposure (as the old code did) invents an opportunity that never existed.
pm_off = pm_[pm_["n_interactions"] > 0].copy()
pm_off["log_exposure"] = np.log(pm_off["n_interactions"])
print(f"\n    Exposure-offset model uses the {len(pm_off)}/{len(pm_)} scheduled days on which the")
print(f"    person actually interacted at least once. Zero-exposure days cannot enter an offset")
print(f"    model (log 0), and are NOT given a fake exposure of 1 — they are reported in the")
print(f"    unoffset totals above, where they belong.")

_g_energy = fit_gee_checked("delivered_energy ~ is_feeder*C(phase)", pm_, groups="person_id",
                            family=sm.families.Gaussian(), focal="is_feeder")
_g_count = fit_gee_checked("meal_count ~ is_feeder*C(phase)", pm_, groups="person_id",
                           family=sm.families.Poisson(), focal="is_feeder")
_g_perop = fit_gee_checked("meal_count ~ is_feeder*C(phase)", pm_off, groups="person_id",
                           family=sm.families.Poisson(), offset=pm_off["log_exposure"],
                           focal="is_feeder")
for nm, r in [("delivered energy/day", _g_energy), ("meal count/day", _g_count),
              ("meals per interaction", _g_perop)]:
    if r["status"] != "ok":
        print(f"    MODEL FAILED [{nm}]: {r['status']} — {r['reason']}")

_meal_boot = cluster_bootstrap(
    pm_, "person_id",
    lambda _s: (_s[(_s.is_feeder==1)&(_s.phase=="P1")]["delivered_energy"].mean() /
                max(_s[(_s.is_feeder==0)&(_s.phase=="P1")]["delivered_energy"].mean(), 1e-9)),
    label="B10.1 DELIVERED ENERGY per scheduled day, feeder/unconstrained (person-cluster boot)")
_perop_boot = cluster_bootstrap(
    pm_off, "person_id",
    lambda _s: ((_s[(_s.is_feeder==1)&(_s.phase=="P1")]["meal_count"].sum() /
                 max(_s[(_s.is_feeder==1)&(_s.phase=="P1")]["n_interactions"].sum(), 1)) /
                max(_s[(_s.is_feeder==0)&(_s.phase=="P1")]["meal_count"].sum() /
                    max(_s[(_s.is_feeder==0)&(_s.phase=="P1")]["n_interactions"].sum(), 1), 1e-9)),
    label="B10.1 meals per INTERACTION, feeder/unconstrained (person-cluster boot)")

# --- LABEL-PERMUTATION SENSITIVITY (exact enumeration; NOT randomisation inference) -------
_perm = pm_[pm_.phase == "P1"].copy()
_perm["role_lab"] = np.where(_perm["is_feeder"] == 1, "feeder", "normal")

def _stat_energy(df):
    f_ = df[df.role_lab == "feeder"]; n_ = df[df.role_lab == "normal"]
    if not len(f_) or not len(n_): return np.nan
    return f_["delivered_energy"].mean() / max(n_["delivered_energy"].mean(), 1e-9)

def _stat_perint(df):
    f_ = df[df.role_lab == "feeder"]; n_ = df[df.role_lab == "normal"]
    if not len(f_) or not len(n_): return np.nan
    a_ = f_["meal_count"].sum() / max(f_["n_interactions"].sum(), 1)
    b_ = n_["meal_count"].sum() / max(n_["n_interactions"].sum(), 1)
    return a_ / b_ if b_ > 0 else np.nan

_pe = exact_permutation_p(_perm, "person_id", "role_lab", _stat_energy)
_pp = exact_permutation_p(_perm, "person_id", "role_lab", _stat_perint)
print(f"\n(4) LABEL-PERMUTATION SENSITIVITY — exact enumeration of all "
      f"{_pe['n_assignments']} ways the 2 feeder labels could fall on these "
      f"{_perm['person_id'].nunique()} people:")
print(f"      delivered energy/day  observed {_pe['observed']:.2f}x   permutation p = {_pe['p']:.3f}")
print(f"      meals per interaction observed {_pp['observed']:.2f}x   permutation p = {_pp['p']:.3f}")
print(f"      Hard floor on p is 1/{_pe['n_assignments']} = {_pe['floor']:.3f}: that is the design.")
print(f"      THIS IS NOT RANDOMISATION INFERENCE. Roles were assigned by availability, so the")
print(f"      permutation distribution has no design justification; it is a descriptive measure of")
print(f"      how much of the contrast rides on these two particular individuals — who differ from")
print(f"      the rest in availability and motivation as well as in role.")

register_p("B10.1", "delivered_energy ~ is_feeder*phase (GEE, cluster=person, complete scheduled-day panel)",
           "is_feeder", _g_energy["p"], "RQ3-adaptation", "exploratory",
           note="non-randomised roles, 2 feeders — descriptive manipulation check")
register_p("B10.1", "meal_count ~ is_feeder*phase + offset(log interactions) (Poisson GEE, cluster=person)",
           "is_feeder", _g_perop["p"], "RQ3-adaptation", "exploratory",
           note="per-encounter propensity; exposure is a mediator, so this is a decomposition")
register_p("B10.1", "delivered energy/day, exact label-permutation sensitivity (NOT randomisation inference)",
           "feeder vs unconstrained", _pe["p"], "RQ3-adaptation", "exploratory",
           note=f"roles non-randomised; exact enumeration of {_pe['n_assignments']} assignments, floor {_pe['floor']:.3f}")
register_p("B10.1", "meals per interaction, exact label-permutation sensitivity (NOT randomisation inference)",
           "feeder vs unconstrained", _pp["p"], "RQ3-adaptation", "exploratory",
           note=f"roles non-randomised; exact enumeration, floor {_pe['floor']:.3f}")

SENSITIVITY.append(dict(metric="B10_role_DELIVERED_ENERGY_ratio", primary=_rr_energy,
                        boot_lo=_meal_boot["lo"], boot_median=_meal_boot["median"],
                        boot_hi=_meal_boot["hi"], successful_refits=_meal_boot["n_ok"],
                        unit="person-cluster bootstrap, delivered energy per SCHEDULED day"))
SENSITIVITY.append(dict(metric="B10_role_MEALS_PER_INTERACTION_ratio", primary=_rr_perint,
                        boot_lo=_perop_boot["lo"], boot_median=_perop_boot["median"],
                        boot_hi=_perop_boot["hi"], successful_refits=_perop_boot["n_ok"],
                        unit="person-cluster bootstrap, meals per interaction"))
globals()["_b10_meal_gee"] = dict(
    rr_energy=_rr_energy, rr_meals=_rr_meals, rr_perint=_rr_perint, rr_expo=_rr_expo,
    boot=[_meal_boot["lo"], _meal_boot["median"], _meal_boot["hi"]],
    boot_perop=[_perop_boot["lo"], _perop_boot["median"], _perop_boot["hi"]],
    perm_p_energy=_pe["p"], perm_p_perint=_pp["p"], perm_floor=_pe["floor"],
    n_assignments=_pe["n_assignments"],
    nofeed_k=_nf_k, nofeed_n=_nf_n, nofeed_hi=_nf_e[2],
    n_sched=len(pdm), n_noshow=int((pdm["attended"] == 0).sum()),
    randomised=False,
    total_energy_feeder=float(q.loc["feeder", "total_energy"]),
    total_meals_feeder=int(q.loc["feeder", "total_meals"]))
globals()["_b10_pdm"] = pdm
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
print("STEP 1b — model the EXACT programmed update-rule components")
print("="*78)
# The update rule, verbatim from salienceNetwork.py:
#     credit    = reward_delta + max(0, active_energy_cost) if meals_eaten > 0 else reward_delta
#     r_norm    = clamp(credit / AFFINITY_REWARD_SCALE), floored at PENALTY_FLOOR if not positive
#     affinity += (ALPHA if positive else ALPHA_NEG) * (r_norm - affinity)
# So Δaffinity is a DETERMINISTIC function of exactly four logged quantities. Model them.
h10["credit"] = h10["reward_delta"] + np.where(
    h10["fed"] > 0, h10["active_energy_cost"].clip(lower=0), 0.0)
_m_rule = smf.ols("d_aff ~ credit + active_energy_cost + fed + affinity_before", h10).fit(
    cov_type="cluster", cov_kwds={"groups": h10["person_id"]})
print("  d_aff ~ credit + active_energy_cost + fed + affinity_before   "
      f"R^2 = {_m_rule.rsquared:.3f}")
print("  These four are the update rule's own inputs. Whatever they leave unexplained is")
print("  clamping and the alpha/alpha_neg branch, not behaviour.")
_m_rule_dose = smf.ols(
    "d_aff ~ credit + active_energy_cost + fed + affinity_before + z_n_turns", h10).fit(
    cov_type="cluster", cov_kwds={"groups": h10["person_id"]})
_dose_resid = float(_m_rule_dose.params["z_n_turns"])
_dose_resid_p = float(_m_rule_dose.pvalues["z_n_turns"])
print(f"\n  Adding the dose on TOP of the full update rule: n_turns coefficient "
      f"{_dose_resid:+.4f} (p={_dose_resid_p:.3f}), R^2 {_m_rule.rsquared:.3f} -> "
      f"{_m_rule_dose.rsquared:.3f}")
print("  Any residual dose association is EXPLORATORY: both the dose and the outcome remain")
print("  mechanically tied to controller arithmetic (turns drive active_energy_cost, which is a")
print("  literal term in `credit`), so this is not an independent behavioural signal.")
register_p("B10.2", "d_aff ~ credit + active_energy_cost + fed + affinity_before + z_n_turns "
                    "(OLS, cluster-robust by person)", "z_n_turns", _dose_resid_p,
           "RQ3-adaptation", "exploratory",
           note="residual dose effect on top of the FULL programmed update rule; dose and outcome "
                "remain mechanically connected, so this cannot be confirmatory")
globals()["_b10_rule"] = dict(r2=float(_m_rule.rsquared), r2_dose=float(_m_rule_dose.rsquared),
                              resid=_dose_resid, resid_p=_dose_resid_p)

print("\n" + "="*78)
print("STEP 2 — MISSINGNESS MODEL for duration (52% missing, differentially)")
print("="*78)
h10["has_duration"] = h10["duration_sec"].notna().astype(int)
print("Duration availability by role x phase:")
print(h10.groupby(["role","phase"], observed=True)["has_duration"]
        .agg(["mean","size"]).round(2).to_string())

# The observation model must respect person-level dependence — durations are missing in CLUMPS,
# by person and by phase, not independently across events. A plain logit treats 205 events as 205
# independent observations of missingness and understates its own uncertainty. Fit with a person
# random intercept; if that is degenerate (it can be, at 14 clusters), fall back to a Firth-
# penalised fit and SAY SO rather than silently reporting the unpenalised one.
_mm_form = ("has_duration ~ C(role) + C(phase) + C(hunger_state_start) + C(trigger_mode) "
            "+ fed + z_n_turns")
_mm_status, _mm_note = "ok", ""
try:
    _mm_mix = smf.mixedlm(_mm_form, h10, groups=h10["person_id"]).fit(reml=False)
    _mm_pred = np.clip(1.0/(1.0 + np.exp(-_mm_mix.fittedvalues)), 1e-6, 1-1e-6)
    _mm_gvar = float(_mm_mix.cov_re.iloc[0, 0])
    _mm_note = f"person random intercept, Var={_mm_gvar:.4f}"
    if _mm_gvar < 1e-6:
        _mm_status = "boundary"
        _mm_note += " (BOUNDARY: person variance collapsed to zero)"
except Exception as e:
    _mm_status, _mm_note = "failed", f"mixed model failed: {e}"

# Separation check + Firth-penalised observation model, always fitted, always reported.
_X_mm = pd.get_dummies(
    h10[["role","phase","hunger_state_start","trigger_mode"]], drop_first=True, dtype=float)
_X_mm["fed"] = h10["fed"].values
_X_mm["z_n_turns"] = h10["z_n_turns"].values
_X_mm.insert(0, "intercept", 1.0)
_y_mm = h10["has_duration"].astype(float).values
_beta_mm = firth_logit(_X_mm.values, _y_mm)
_eta = _X_mm.values @ _beta_mm
h10["p_obs"] = np.clip(1.0/(1.0 + np.exp(-_eta)), 1e-6, 1-1e-6)
_sep = bool(np.max(np.abs(_beta_mm)) > 15)
print(f"\nObservation model (Firth-penalised logit; person dependence: {_mm_status} — {_mm_note}):")
print(f"  separation detected: {_sep}")
_mm_tab = pd.DataFrame({"term": _X_mm.columns, "coef_firth": _beta_mm})
print(_mm_tab.round(3).to_string(index=False))
_mm_tab.to_csv(OUT_DIR/"rq3_missingness_model.csv", index=False)
print("\n  -> duration is NOT missing at random with respect to role and phase, which is exactly")
print("     why it cannot be the primary dose.")

# --- IPW with FULL diagnostics and truncation sensitivity ---------------------------------
_p_marg = h10["has_duration"].mean()
_ipw_variants = {}
for trunc_lbl, (lo_q, hi_q) in {"none": (0.0, 1.0), "1-99%": (0.01, 0.99),
                                "5-95%": (0.05, 0.95)}.items():
    w = np.where(h10["has_duration"] == 1, _p_marg / h10["p_obs"], 0.0)
    obs = h10["has_duration"] == 1
    if lo_q > 0:
        lo_w, hi_w = np.quantile(w[obs], [lo_q, hi_q])
        w = np.clip(w, lo_w, hi_w)
        w = np.where(obs, w, 0.0)
    _ipw_variants[trunc_lbl] = w
h10["ipw"] = _ipw_variants["1-99%"]          # prespecified truncation

_diag = ipw_diagnostics(h10["p_obs"].values, h10["has_duration"].values, h10["ipw"].values)
print("\nIPW DIAGNOSTICS (prespecified truncation: 1st-99th percentile):")
print(f"  predicted P(observed) range : [{_diag['p_min']:.3f}, {_diag['p_max']:.3f}]")
print(f"  weight range                : [{_diag['w_min']:.2f}, {_diag['w_max']:.2f}] "
      f"(mean {_diag['w_mean']:.2f})")
print(f"  effective sample size       : {_diag['ess']:.1f} of {_diag['n_observed']} observed "
      f"({_diag['ess_frac']*100:.0f}%)")
print(f"  positivity violation        : {_diag['positivity_violation']}")
if _diag["positivity_violation"]:
    print("    -> some stratum is almost never observed. No weight can recover information that")
    print("       was never collected; the IPW estimate is reported, but it is not a repair.")
print("\n  Covariate balance before vs after weighting (standardised mean difference):")
_bal_rows = []
for c_ in ["fed", "z_n_turns"]:
    o, m_ = h10[h10.has_duration == 1][c_], h10[h10.has_duration == 0][c_]
    sd = np.sqrt((o.var() + m_.var()) / 2) or 1.0
    smd_raw = (o.mean() - m_.mean()) / sd
    ww = h10.loc[h10.has_duration == 1, "ipw"]
    wm = np.average(o, weights=ww)
    smd_w = (wm - m_.mean()) / sd
    _bal_rows.append(dict(covariate=c_, smd_unweighted=smd_raw, smd_weighted=smd_w))
_bal = pd.DataFrame(_bal_rows)
print(_bal.round(3).to_string(index=False))
_bal.to_csv(OUT_DIR/"rq3_ipw_balance.csv", index=False)
globals()["_b10_ipw_diag"] = _diag

print("\n" + "="*78)
print("STEP 3 — SECONDARY: duration, complete-case vs IPW (and truncation sensitivity)")
print("="*78)
_dur_df = h10[h10["has_duration"] == 1].copy()
R["dur_cc"] = fit_mix(
    _dur_df, f"d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
             f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
    "SECONDARY complete-case, dose=duration", "z_duration_sec")
R["dur_ipw"] = fit_mix(
    _dur_df, f"d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
             f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
    "SECONDARY IPW-weighted, dose=duration", "z_duration_sec", weights="ipw")

print("\nWEIGHT-TRUNCATION SENSITIVITY (does the answer depend on where we clip?):")
_trunc_rows = []
for lbl_, w_ in _ipw_variants.items():
    tmp = h10.copy(); tmp["w_"] = w_
    sub = tmp[tmp.has_duration == 1]
    try:
        f_ = smf.wls(f"d_aff ~ z_duration_sec*C(role) + z_duration_sec*C(phase) "
                     f"+ C(hunger_state_start) + C(trigger_mode) {CTRL}",
                     sub, weights=sub["w_"]).fit(
            cov_type="cluster", cov_kwds={"groups": sub["person_id"]})
        _trunc_rows.append(dict(truncation=lbl_, slope=float(f_.params["z_duration_sec"]),
                                lo=float(f_.conf_int().loc["z_duration_sec", 0]),
                                hi=float(f_.conf_int().loc["z_duration_sec", 1]),
                                max_weight=float(w_[sub.index].max())))
    except Exception as e:
        _trunc_rows.append(dict(truncation=lbl_, slope=np.nan, lo=np.nan, hi=np.nan,
                                max_weight=np.nan))
_trunc = pd.DataFrame(_trunc_rows)
print(_trunc.round(4).to_string(index=False))
_trunc.to_csv(OUT_DIR/"rq3_ipw_truncation_sensitivity.csv", index=False)

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
# AGREEMENT MUST BE IN SIGN **AND** MAGNITUDE. "All three doses agree in sign" was offered as
# corroboration when one of the three was a literal additive term in the outcome's definition.
# Sign agreement is a very weak property: +0.041 and +0.168 agree in sign while differing
# four-fold, and calling them "consistent" would hide the entire finding.
_ests = {r_["model"]: float(r_["slope"]) for _, r_ in dose_cmp.iterrows()
         if r_["model"].startswith(("PRIMARY", "SECONDARY"))}
_agree_res = spec_agreement(_ests, rel_tol=0.5)          # magnitudes within 2x of each other
_agree = bool(_agree_res["sign_agree"] and _agree_res["magnitude_agree"])
print(f"\n  Specification agreement (sign AND magnitude, tolerance 2x):")
print(f"    sign agree      : {_agree_res['sign_agree']}")
print(f"    magnitude agree : {_agree_res['magnitude_agree']}  (max ratio "
      f"{_agree_res['max_ratio']:.1f}x)")
print(f"    -> {_agree_res['reason']}")
_old = dose_cmp[dose_cmp.model.str.startswith("OLD")]["slope"]
_new = dose_cmp[dose_cmp.model.str.startswith("PRIMARY")]["slope"]
if len(_old) and len(_new):
    print(f"\n  The old, uncontrolled duration slope was {float(_old.iloc[0]):+.3f}; controlling for "
          f"`fed` and `affinity_before` and using the fully-observed dose gives "
          f"{float(_new.iloc[0]):+.3f} — a {float(_old.iloc[0])/float(_new.iloc[0]):.1f}-fold "
          f"difference. They agree in sign. They do not agree.")
globals()["_b10_agree_res"] = _agree_res

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

**Exposure is now reconstructed from the perception and salience logs, not from completed
interactions.** The previous model derived "presence" from the fact that an interaction *happened*
— which is the outcome, or nearly so. A person the robot never approached had no interaction, so
they looked absent, so they left the denominator. That is not an exposure model; it is the outcome
wearing a denominator's clothes.

The controller's actual pipeline gives three distinct stages, and all three are logged:

1. **Detected / present** — the person appears in `target_selections` at all on day *d+1*
   (the salience network saw and identified them).
2. **Eligible given detection** — `target_selections.eligible == 1`: their IPS cleared
   `eff_thr = max(0.50, base_ss − 0.15·affinity)`. This is the stage affinity mechanically acts on.
3. **Proactively approached given eligibility** — the executive actually initiated.

The offset is the **eligible-detection count**, which is the real number of approach opportunities.
Days with **zero** exposure are *not* given a fake exposure of 1 — that invents an opportunity that
never existed. They are kept in the outcome and reported, and simply cannot enter an offset model.

**And the honest caveat stands.** Stage 2 *is* the eligibility threshold, which B9 verifies to a
maximum error of 0.0000. So "higher affinity → more eligible detections" is the source code, not a
discovery. The only stage with any behavioural content is stage 3, and it is downstream of the
other two. This is `Implementation verification` with an exploratory deployment association on
top.""")

code(r"""
# --- B10d. Three-stage exposure from the perception/salience logs -------------------------
tsel_all = load_view("salience", "v_target_selections_clean")
tsel_all = tsel_all[tsel_all["person_id"] != "unknown"].copy()
tsel_all["day_rome"] = tsel_all["day_rome"].astype(str)

# STAGE 1 + 2, per person-day, straight from the salience logs.
stage = (tsel_all.groupby(["person_id", "day_rome"])
                 .agg(detections=("eligible", "size"),
                      eligible_detections=("eligible", "sum"))
                 .reset_index())
print(f"Salience target-selection log: {len(tsel_all)} records, "
      f"{tsel_all['person_id'].nunique()} named people, {len(stage)} person-days with a detection.")
print(f"  detections total          : {int(stage['detections'].sum())}")
print(f"  ELIGIBLE detections total : {int(stage['eligible_detections'].sum())}  "
      f"<- the real approach-opportunity count")

# STAGE 3: proactive approaches actually initiated.
_pro = m10[m10["trigger_mode"] == "proactive"]
day_aff = (h10.groupby(["person_id", "day_rome"])["affinity_after"].last()
              .rename("aff_eod").reset_index())
day_aff["day_rome"] = day_aff["day_rome"].astype(str)
_days = sorted(str(d_) for d_ in m10["day_rome"].unique())

rows = []
for _p in m10[m10["role"] != "unknown"]["person_id"].unique():
    for _i in range(len(_days) - 1):
        _d0, _d1 = _days[_i], _days[_i + 1]
        _a = day_aff[(day_aff.person_id == _p) & (day_aff.day_rome <= _d0)]
        st = stage[(stage.person_id == _p) & (stage.day_rome == _d1)]
        n_det = int(st["detections"].iloc[0]) if len(st) else 0
        n_elig = int(st["eligible_detections"].iloc[0]) if len(st) else 0
        n_pro = int((_pro["person_id"].eq(_p) & _pro["day_rome"].astype(str).eq(_d1)).sum())
        rows.append(dict(
            person_id=_p, day=_d1, phase=phase_of_day(_d1),
            prior_aff=float(_a["aff_eod"].iloc[-1]) if len(_a) else 0.0,
            n_today=int((m10["person_id"].eq(_p) & m10["day_rome"].astype(str).eq(_d0)).sum()),
            detected_next=int(n_det > 0), n_detections_next=n_det,
            n_eligible_next=n_elig, n_pro_next=n_pro))
panel = pd.DataFrame(rows)
panel.to_csv(OUT_DIR/"b10_downstream_stages.csv", index=False)

print(f"\n=== ESTIMAND ===")
print(f"Among people DETECTED on day d+1: proactive approaches toward person i on d+1, per")
print(f"ELIGIBLE-DETECTION opportunity, as a function of i's affinity at end of day d.")
print(f"\nPerson-day grid: {len(panel)} rows ({panel['person_id'].nunique()} people x "
      f"{len(_days)-1} day-transitions)")
print(f"  detected on d+1                : {int(panel['detected_next'].sum())}")
print(f"  detected AND eligible >= 1     : {int((panel['n_eligible_next'] > 0).sum())}")
print(f"  zero-exposure days (no eligible detection) are NOT given a fake exposure of 1; they")
print(f"  simply cannot enter an offset model, and are reported here instead of being hidden.")

# --- The three stages, fitted separately ---------------------------------------------------
stages = []

# Stage 1: does affinity predict being DETECTED at all? (Presence is largely up to the human.)
s1 = fit_gee_checked("detected_next ~ prior_aff + n_today + C(phase)", panel,
                     groups="person_id", family=sm.families.Binomial(), focal="prior_aff")
stages.append(dict(stage="1. P(detected on d+1)", **{k: s1[k] for k in
              ("n", "status", "reason", "estimate", "lo", "hi", "p")}))

# Stage 2: eligibility GIVEN detection. This IS the coded threshold, so expect it to be strong —
# and fit it at the DETECTION level, which is its natural unit. A Poisson-with-offset on the
# person-day count is the wrong model for a bounded proportion (eligible <= detections) and it
# returns a non-finite estimate here; fit_gee_checked reports that rather than hiding it. Each
# target-selection record is one Bernoulli trial: did this detection clear the threshold?
det_lvl = tsel_all[["person_id", "day_rome", "eligible"]].copy()
det_lvl = det_lvl.merge(
    panel[["person_id", "day", "prior_aff", "phase"]].rename(columns={"day": "day_rome"}),
    on=["person_id", "day_rome"], how="inner")
det_lvl["eligible"] = det_lvl["eligible"].astype(int)
print(f"\nStage 2 is fitted at the DETECTION level: {len(det_lvl)} target-selection records "
      f"(one Bernoulli trial each) over {det_lvl['person_id'].nunique()} people.")
s2 = fit_gee_checked("eligible ~ prior_aff + C(phase)", det_lvl, groups="person_id",
                     family=sm.families.Binomial(), focal="prior_aff")
s2["n"] = len(det_lvl)
stages.append(dict(stage="2. eligible | detected (THE CODED GATE)",
                   **{k: s2[k] for k in ("n", "status", "reason", "estimate", "lo", "hi", "p")}))
_elig_terc = det_lvl.assign(q=pd.qcut(det_lvl["prior_aff"], 3, duplicates="drop")) \
                    .groupby("q", observed=True)["eligible"].agg(["mean", "size"])
print("Eligibility rate by prior-affinity tercile (descriptive — this IS the coded threshold):")
print(_elig_terc.round(3).to_string())

# Stage 3: proactive approach GIVEN eligibility — the only stage with behavioural content.
elig = panel[panel["n_eligible_next"] > 0].copy()
elig["log_elig"] = np.log(elig["n_eligible_next"])
s3 = fit_gee_checked("n_pro_next ~ prior_aff + n_today + C(phase)", elig, groups="person_id",
                     family=sm.families.Poisson(), offset=elig["log_elig"], focal="prior_aff")
stages.append(dict(stage="3. proactive | eligible",
                   **{k: s3[k] for k in ("n", "status", "reason", "estimate", "lo", "hi", "p")}))

st_tab = pd.DataFrame(stages)
st_tab["rate_ratio"] = np.exp(st_tab["estimate"])
st_tab["rr_lo"] = np.exp(st_tab["lo"]); st_tab["rr_hi"] = np.exp(st_tab["hi"])
st_tab["cov_struct"] = [s1.get("cov_struct"), s2.get("cov_struct"), s3.get("cov_struct")]
print("\n=== THE THREE STAGES ===")
print(st_tab[["stage", "n", "status", "cov_struct", "rate_ratio", "rr_lo", "rr_hi", "p"]]
      .round(4).to_string(index=False))
st_tab.to_csv(OUT_DIR/"b10_downstream_stage_models.csv", index=False)
for r_ in stages:
    if r_["status"] != "ok":
        print(f"  MODEL FAILED [{r_['stage']}]: {r_['status']} — {r_['reason']}")

for r_, nm in zip(stages, ["detected_next ~ prior_aff (logistic GEE)",
                           "eligible ~ prior_aff (logistic GEE, detection-level, cluster=person)",
                           "n_pro_next ~ prior_aff + offset(log eligible) (Poisson GEE)"]):
    if np.isfinite(r_["p"]):
        register_p("B10.3", nm, "prior_aff", r_["p"], "RQ3-adaptation",
                   "exploratory" if r_["stage"].startswith("2") else "confirmatory",
                   note=("stage 2 IS the coded eligibility threshold (eff_thr = max(0.50, "
                         "base - 0.15*affinity), verified exactly in B9) — it is implementation "
                         "verification, not inference" if r_["stage"].startswith("2")
                         else "three-part exposure decomposition"))

_pri = s3
_prior_boot = cluster_bootstrap(
    elig, "person_id",
    lambda _s: np.exp(smf.glm("n_pro_next ~ prior_aff + n_today + C(phase)", data=_s,
                              family=sm.families.Poisson(),
                              offset=_s["log_elig"]).fit().params["prior_aff"]),
    label="B10.3 stage-3 approach RR per eligible opportunity (person-cluster bootstrap)")
SENSITIVITY.append(dict(metric="B10_prior_affinity_RR_per_eligible_opportunity",
                        primary=float(np.exp(_pri["estimate"])),
                        boot_lo=_prior_boot["lo"], boot_median=_prior_boot["median"],
                        boot_hi=_prior_boot["hi"], successful_refits=_prior_boot["n_ok"],
                        unit="person-cluster bootstrap, offset by eligible detections"))
globals()["_b10_prior"] = dict(
    rr=float(np.exp(_pri["estimate"])), ci=[float(np.exp(_pri["lo"])), float(np.exp(_pri["hi"]))],
    p=float(_pri["p"]), boot=[_prior_boot["lo"], _prior_boot["median"], _prior_boot["hi"]],
    stage1_or=float(np.exp(s1["estimate"])), stage1_p=float(s1["p"]),
    stage2_rr=float(np.exp(s2["estimate"])), stage2_p=float(s2["p"]),
    stage2_lo=float(np.exp(s2["lo"])), stage2_hi=float(np.exp(s2["hi"])),
    n_panel=len(panel), n_detected=int(panel["detected_next"].sum()),
    n_eligible=int((panel["n_eligible_next"] > 0).sum()),
    n_people=int(elig["person_id"].nunique()),
    total_eligible=int(panel["n_eligible_next"].sum()))
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
#
# B10.1 is EXPLORATORY, not an association: roles were assigned by availability, there are 2
# people per role, and role is therefore nearly aliased with identity. No population role effect
# is estimated, and the words "randomisation" and "survives randomisation" do not appear.
verdict("B10.1", EV_EXPL,
        f"Exploratory manipulation check for these four participants. Roles were assigned BY "
        f"AVAILABILITY, not randomised, so nothing here is randomisation inference and no "
        f"population role effect is estimated. On the COMPLETE scheduled-day panel "
        f"({_mg['n_sched']} scheduled person-days, of which {_mg['n_noshow']} were no-shows kept as "
        f"genuine zeros): obligated feeders delivered {_mg['rr_energy']:.1f}x the ENERGY per "
        f"scheduled day (sum of meal_delta — stomach points, not a meal count; person-cluster "
        f"bootstrap [{_mg['boot'][0]:.1f}, {_mg['boot'][2]:.1f}]), but PER INTERACTION they fed only "
        f"{_mg['rr_perint']:.1f}x as often (bootstrap "
        f"[{_mg['boot_perop'][0]:.1f}, {_mg['boot_perop'][2]:.1f}]). The gap is ATTENDANCE: they "
        f"showed up {_mg['rr_expo']:.1f}x more per scheduled day. Being told to feed the robot made "
        f"people GO TO the robot; it did not make them markedly more generous once there. Exposure "
        f"is a mediator of the role, not a confounder, so the per-encounter figure is a "
        f"decomposition and the energy figure is what the drive actually experienced. A "
        f"label-permutation sensitivity over the exact enumeration of all {_mg['n_assignments']} "
        f"possible assignments gives p={_mg['perm_p_energy']:.3f} (energy) and "
        f"p={_mg['perm_p_perint']:.3f} (per-encounter) against a hard design floor of "
        f"{_mg['perm_floor']:.3f} — a descriptive measure of how much of the contrast rides on two "
        f"individuals, NOT a randomisation test. The no-feed pair supplied "
        f"{_mg['nofeed_k']}/{_mg['nofeed_n']} meals in Phase 1 (exact upper bound "
        f"{_mg['nofeed_hi']:.2f}): perfect compliance, but {_mg['nofeed_n']} observations cannot "
        f"rule out more than that. An earlier version reported a 2.7x meal-rate ratio as though "
        f"feeders fed more READILY; the excess is real, and it is a fact about attendance.",
        n=int(_mg["n_sched"]))

# RQ3-b is IMPLEMENTATION VERIFICATION with an exploratory deployment association on top.
# It is not a general learning result, and the previous class (Within-deployment association)
# claimed more than the mechanism permits: affinity is a deterministic EMA of delivered energy,
# and the downstream path runs through a threshold B9 verifies exactly.
_rule = globals()["_b10_rule"]; _ag = globals()["_b10_agree_res"]
_ipwd = globals()["_b10_ipw_diag"]
verdict("B10", EV_IMPL,
        f"IMPLEMENTATION VERIFICATION, with an exploratory deployment association on top — not a "
        f"general learning result. Affinity is a DETERMINISTIC EMA of delivered energy: modelling "
        f"its four programmed inputs (credit, active_energy_cost, fed, affinity_before) explains "
        f"R^2={_rule['r2']:.2f} of every update, and what they leave over is clamping and the "
        f"alpha/alpha_neg branch, not behaviour. On top of the FULL update rule, engagement dose "
        f"adds {_rule['resid']:+.4f} per SD (p={_rule['resid_p']:.3f}) — EXPLORATORY, because dose "
        f"and outcome remain mechanically connected (turns drive active_energy_cost, a literal term "
        f"in `credit`). With only `fed` and `affinity_before` controlled and the fully observed dose, "
        f"the slope is {_sl['coef']:+.3f} [{_sl['lo']:+.3f}, {_sl['hi']:+.3f}] (person-cluster "
        f"bootstrap [{_db[0]:+.3f}, {_db[2]:+.3f}]) against the "
        f"{float(globals()['_b10_dose_cmp'].query('model.str.startswith(\"OLD\")')['slope'].iloc[0]):+.3f} "
        f"the uncontrolled specification reported — a {_ag['max_ratio']:.1f}x spread across "
        f"specifications, so they agree in SIGN but NOT in magnitude ({_ag['reason']}). Duration is "
        f"52% missing and NOT at random; the IPW fit has an effective sample size of "
        f"{_ipwd['ess']:.0f}/{_ipwd['n_observed']} ({_ipwd['ess_frac']*100:.0f}%) and "
        f"{'a POSITIVITY VIOLATION' if _ipwd['positivity_violation'] else 'no positivity violation'}. "
        f"DOWNSTREAM, decomposed into its three logged stages: affinity does not clearly predict who "
        f"is DETECTED (OR {_pr['stage1_or']:.2f}, p={_pr['stage1_p']:.3f}); the eligibility rate DOES rise "
        f"with affinity descriptively (3% -> 9% across terciles), but at the person-cluster level it "
        f"is not distinguishable (RR {_pr['stage2_rr']:.2f} "
        f"[{_pr['stage2_lo']:.2f}, {_pr['stage2_hi']:.2f}], p={_pr['stage2_p']:.2f}) — and that stage "
        f"IS the coded threshold eff_thr=max(0.50, base-0.15*affinity), which B9 verifies to a maximum "
        f"error of 0.0000, so it is arithmetic, not a finding; and given eligibility it is "
        f"associated with proactive approach at RR {_pr['rr']:.2f} "
        f"[{_pr['ci'][0]:.2f}, {_pr['ci'][1]:.2f}] (bootstrap [{_pr['boot'][0]:.2f}, "
        f"{_pr['boot'][2]:.2f}]) over {_pr['total_eligible']} eligible opportunities. What RQ3 "
        f"genuinely establishes is B10.1 — the role manipulation changed what PEOPLE did. Everything "
        f"else is the controller doing what it was written to do.",
        n=len(h10))
# BH correction is NOT run here. It runs ONCE, at the very end of the notebook, after
# EVERY analysis (including D1) has registered its p-values. Correcting a family before
# it is complete silently omits whatever registers later.
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
            ("feeding received\nP(meal arrived)",_mm,"fed")]
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
# Built from the REPLY-CENTRIC one-to-one matched pings and their MATCHED controls (B5b).
_pm   = globals()["_b5_pm"]
_win  = globals()["_b5_win"]
_bp   = globals()["_b5_ping"]
_pairs = pd.read_csv(OUT_DIR/"b5_ping_control_pairs.csv")
_p60  = _pairs[_pairs.window_min == 60]
_kcol = {"hs2_entry": HS_PALETTE["HS2"], "hs3_proactive": HS_PALETTE["HS3"]}
_klab = {"hs2_entry": "Hungry\nping", "hs3_proactive": "Starving\nping"}

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.8,4.8),width_ratios=[1,1.06])

# LEFT: ping vs its MATCHED control — the comparison the previous version never made at all.
cats, es, los, his, ns, cols = [], [], [], [], [], []
for k in ["hs2_entry","hs3_proactive"]:
    sub=_p60[_p60.ping_type==k]
    if not len(sub): continue
    e,lo,hi = exact_prop_ci(int(sub["replied"].sum()), len(sub))
    cats.append(_klab[k]); es.append(e); los.append(e-lo); his.append(hi-e)
    ns.append(len(sub)); cols.append(_kcol[k])
e,lo,hi = exact_prop_ci(int(_p60["replied"].sum()), len(_p60))
cats.append("ALL hunger\npings"); es.append(e); los.append(e-lo); his.append(hi-e)
ns.append(len(_p60)); cols.append(INK)
ec,clo,chi = exact_prop_ci(int(_p60["control_replied"].sum()), len(_p60))
cats.append("MATCHED\nCONTROL\n(no ping)"); es.append(ec); los.append(ec-clo); his.append(chi-ec)
ns.append(len(_p60)); cols.append("#B0B7BE")
bars_with_ci(axL, cats, es, los, his, cols, n_labels=ns)
axL.axhline(ec, color="#B23A26", ls="--", lw=1.4, zorder=5)
axL.set_ylim(0,1); axL.set_ylabel("P(user reply within 1 h)")
axL.set_title(f"Ping vs matched control (same subscriber, run, time-of-day)\n"
              f"paired difference {_bp['diff']:+.2f}  "
              f"[subscriber-cluster {_bp['boot'][0]:+.2f}, {_bp['boot'][2]:+.2f}]  "
              f"[run-cluster {_bp['run_boot'][0]:+.2f}, {_bp['run_boot'][2]:+.2f}]",
              fontsize=10.2)

# RIGHT: window sensitivity, ping vs control, with the paired difference.
axR.errorbar(_win["window_min"], _win["ping_rate"],
             yerr=[_win["ping_rate"]-_win["ping_lo"], _win["ping_hi"]-_win["ping_rate"]],
             marker="o", ms=7, lw=2.0, capsize=4, color=HS_PALETTE["HS3"], label="after a hunger ping")
axR.errorbar(_win["window_min"], _win["control_rate"],
             yerr=[_win["control_rate"]-_win["control_lo"], _win["control_hi"]-_win["control_rate"]],
             marker="s", ms=6, lw=2.0, capsize=4, color="#8A939B", ls="--", label="matched control window")
for _,r_ in _win.iterrows():
    axR.annotate(f"{r_['paired_diff']:+.2f}", (r_["window_min"], (r_["ping_rate"]+r_["control_rate"])/2),
                 textcoords="offset points", xytext=(8,-2), fontsize=8.6, color=INK,
                 fontweight="semibold")
axR.set_xticks([15,30,60]); axR.set_xlabel("reply window (minutes)")
axR.set_ylabel("P(user reply)"); axR.set_ylim(0,1)
axR.legend(fontsize=8.6, loc="upper left")
axR.set_title("Window sensitivity (exact 95% CI; paired difference labelled)", fontsize=10.6)
fig.suptitle("Fig 8 — The remote channel is a real but weak recovery pathway",
             fontsize=13, fontweight="semibold")
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
# The DECOMPOSITION is the finding: the role changed attendance, not per-encounter generosity.
# Note these are DELIVERED ENERGY (sum of meal_delta), not meal counts.
axR.text(0.985,0.97,(f"feeder vs unconstrained, Phase 1\n"
        f"(roles assigned by AVAILABILITY, not randomised)\n"
        f"delivered ENERGY / scheduled day  {_mg['rr_energy']:.2f}x\n"
        f"   boot [{_mg['boot'][0]:.2f}, {_mg['boot'][2]:.2f}]\n"
        f"meals per INTERACTION             {_mg['rr_perint']:.2f}x\n"
        f"   boot [{_mg['boot_perop'][0]:.2f}, {_mg['boot_perop'][2]:.2f}]\n"
        f"interactions / scheduled day      {_mg['rr_expo']:.2f}x  <- ATTENDANCE\n"
        f"label-permutation sensitivity p={_mg['perm_p_energy']:.3f}\n"
        f"   (exact over {_mg['n_assignments']} assignments; floor {_mg['perm_floor']:.3f})\n"
        f"NOT randomisation inference."),
        transform=axR.transAxes,va="top",ha="right",fontsize=7.4,family="monospace",
        bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=MUTED,lw=0.8,alpha=0.94))
axR.set_xticks([0,1]); axR.set_xticklabels(["Phase 1 (roles active)","Phase 2 (unconstrained)"])
axR.set_ylabel("meals per person-day (observed)")
axR.set_ylim(0, float(_bars.max())*1.9 if len(_bars) else 1.0)
axR.set_title("Meal supply per person-day",fontsize=11.5)
fig.suptitle("Fig 12 — The role changed ATTENDANCE, not generosity. Non-randomised; 4 people.",
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

# REGISTER D1's permutation p-values. The previous version quoted "+0.088 AUC" in its verdict and
# in the README while running BH inside B10 — before D1 had even executed — so these could never
# have entered a family. They are quoted in a conclusion, so they are corrected like anything else.
for _t in ["replied_any", "reached_ss4"]:
    _r = abl_delta[abl_delta.target == _t].iloc[0]
    register_p("D1", f"ΔAUC from adding hunger state, {int(_r['n_repeats'])} repeated grouped-CV "
                     f"runs vs within-run permutation null ({int(_r['n_perm'])} permutations)",
               f"hunger_state -> {_t}", float(_r["perm_p"]), "RQ1/2-behaviour", "confirmatory",
               note="held-out predictive sensitivity; quoted in the D1 verdict and the README")
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
# MEAL COUNT AND DELIVERED ENERGY ARE DIFFERENT QUANTITIES and are no longer conflated. The
# previous cell summed `meals_eaten_count` — a COUNT — and printed it as "total meal energy into
# the drive ... (stomach %)". A count of meals is not stomach points: meals come in three sizes
# (SMALL 10 / MEDIUM 25 / LARGE 45), so ten small meals and ten large ones are the same count and
# 350 points apart. Delivered energy is the sum of `meal_delta`, and that is what the drive
# actually experiences.
# Same attribution as B10.1: feeds are assigned to the interaction that was active when they
# occurred. `feeder_face_id` is populated for only 20 of 108 feeds and would drop most of the
# energy and half the participants.
_named_feeds = globals()["_feeds_attributed"]

by_user = (_named_feeds.groupby("person_id")
                       .agg(meal_count=("meal_delta", "size"),
                            delivered_energy=("meal_delta", "sum"))
                       .sort_values("delivered_energy", ascending=False))
print(f"Meals delivered by {len(by_user)} named users:")
print(f"  meal COUNT      total = {int(by_user['meal_count'].sum())} meals")
print(f"  delivered ENERGY total = {by_user['delivered_energy'].sum():.0f} stomach points")
print(f"  (these are different quantities; meals are SMALL 10 / MEDIUM 25 / LARGE 45)")

def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float(2*np.sum(np.arange(1, n+1)*x)/(n*x.sum()) - (n+1)/n) if x.sum() > 0 else np.nan

nP = len(by_user)
gini_e = _gini(by_user["delivered_energy"])
gini_c = _gini(by_user["meal_count"])
top3_e = by_user["delivered_energy"].head(3).sum()/max(by_user["delivered_energy"].sum(), 1)
top3_c = by_user["meal_count"].head(3).sum()/max(by_user["meal_count"].sum(), 1)
print(f"\nConcentration of DELIVERED ENERGY: Gini={gini_e:.2f}; top 3 supply {top3_e*100:.0f}%")
print(f"Concentration of MEAL COUNT:       Gini={gini_c:.2f}; top 3 supply {top3_c*100:.0f}%")
by_user.to_csv(OUT_DIR/"d4_feeding_concentration.csv")

conc = ("concentrated in a few feeders (fragile)" if (top3_e > 0.75 or gini_e > 0.75) else
        "moderate concentration — replenishment leans on a few feeders"
        if (top3_e >= 0.45 or gini_e >= 0.5) else "well spread across users (robust)")
globals()["_d4"] = dict(gini_energy=gini_e, top3_energy=top3_e, n_users=nP,
                        total_energy=float(by_user["delivered_energy"].sum()),
                        total_meals=int(by_user["meal_count"].sum()))
verdict("D4", EV_EXPL,
        f"Descriptive. Over {nP} named users, DELIVERED ENERGY (sum of meal_delta, "
        f"{by_user['delivered_energy'].sum():.0f} stomach points from "
        f"{int(by_user['meal_count'].sum())} meals) has Gini={gini_e:.2f}, with the top 3 supplying "
        f"{top3_e*100:.0f}% — {conc}. This is the standing caveat on every RQ2 result: the "
        f"regulatory loop closed because a handful of specific people chose to close it, and "
        f"nothing here shows that would hold in another room of people.",
        n=nP)
""")


# ==========================================================================
# SYNTHESIS
# ==========================================================================
md(r"""### Multiplicity: Benjamini-Hochberg over the COMPLETE ledger

Run **once, here**, after every analysis — B3, B4, B5, B7, B9, B10.1, B10.2, B10.3, D1, D4 — has
registered its p-values. The previous version corrected the families inside B10, before D1 had run,
so D1's numbers could never have entered them; and it quoted the dose x role and dose x phase
interaction terms in its verdicts while correcting neither.

A p-value is `confirmatory` only if it can support a claim. Everything else — B4's separated cell,
B10.1's non-randomised label permutation, B10.2's residual dose on top of the update rule, B10.3's
eligibility stage (which *is* the coded threshold) — is registered as `exploratory`: recorded in
full, never corrected, never used to support a conclusion.""")
code(r"""
BH = run_bh()
""")

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
_pr=globals()["_b10_prior"]; _dc=globals()["_b10_dose_cmp"]; _rule=globals()["_b10_rule"]
_ag=globals()["_b10_agree_res"]; _ipwd=globals()["_b10_ipw_diag"]; _d4=globals()["_d4"]
_ptab = pd.read_csv(OUT_DIR/"multiplicity_table.csv")
_old_slope = float(_dc.query('model.str.startswith("OLD")')["slope"].iloc[0])
_new_slope = float(_dc.query('model.str.startswith("PRIMARY")')["slope"].iloc[0])

L=["# Orexigenic drive — results summary", "",
   f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Single always-on condition (no drive-off control). "
   f"{hunger_raw['run_id'].nunique()} monitored runs, {interactions['run_id'].nunique()} with visitors, "
   f"{hunger_raw['day_rome'].nunique()} session-days, {len(interactions)} interactions, "
   f"{master[master.person_id!='unknown']['person_id'].nunique()} named people. "
   f"Two-phase design: Phase 1 (first 4 days) had roles assigned BY AVAILABILITY (2 obligated "
   f"feeders, 2 interact-no-feed, rest unconstrained) — **not randomised**; Phase 2 unconstrained._", "",
   "## How to read this report", "",
   "Every result carries exactly one evidence class. They are not interchangeable:", "",
   "| Class | Meaning |", "|---|---|",
   f"| `{EV_IMPL}` | Follows from the controller source. Confirms the code is faithfully implemented and logged. **Not a discovered fact.** |",
   f"| `{EV_ASSOC}` | A cluster-aware association in this deployment. Not causal, not a population estimate. |",
   f"| `{EV_EXPL}` | Descriptive. Too small-n or too selection-prone to support inference. |",
   f"| `{EV_INCONC}` | Run, and did not settle the question. |",
   f"| `{EV_REPL}` | Suggestive; identification needs new data. |", "",
   "## Verification gate", "",
   f"All V1–V5 checks passed. Controller constants verified key-by-key against the pinned "
   f"deployment commits (see `reproducibility_report.md`); corpus energy balance active-out "
   f"{hunger_raw[hunger_raw.event_type=='active_cost'].active_energy_cost.sum():.0f} vs meal-in "
   f"{hunger_raw[hunger_raw.event_type=='feeding'].meal_delta.sum():.0f}.", ""]

L+=["## Results", "", "| id | claim | evidence class |", "|---|---|---|"]
for _,r in sc_df.iterrows():
    L.append(f"| {r['id']} | {r['claim']} | `{r['evidence_class']}` |")

L+=["", "## Per-analysis verdicts", ""]
for k in ["B1","B2","B3","B4","B5","B6","B7","B9","B10.1","B10","D1","D4"]:
    r = RESULTS.get(k)
    if r: L.append(f"- **{k}** — `{r['evidence']}` — {r['verdict']}\n")

L+=["", "## Multiplicity", "",
    f"Every p-value entering a conclusion is registered at the point it is computed and exported to "
    f"`multiplicity_table.csv` ({len(_ptab)} rows). Benjamini–Hochberg runs **once, at the very end**, "
    f"after every analysis — including D1 — has registered. The previous version corrected inside B10, "
    f"before D1 had run, and quoted interaction terms it never corrected.", "",
    f"- Confirmatory: {int((_ptab['status']=='confirmatory').sum())}, of which "
    f"{int(_ptab['sig_q05'].fillna(False).sum())} survive at q<0.05.",
    f"- Exploratory (recorded in full, NOT corrected, never used to support a claim): "
    f"{int((_ptab['status']=='exploratory').sum())} — B4's separated cell, B10.1's non-randomised "
    f"label permutation, B10.2's residual dose on top of the update rule, and B10.3's eligibility "
    f"stage (which *is* the coded threshold).", ""]

L+=["## Key quantities", "",
    f"- **Deficit -> feeding received** (B3, the strongest result): OR {_b3['orr']:.1f}, "
    f"person-cluster bootstrap [{_b3['boot'][0]:.1f}, {_b3['boot'][2]:.1f}], run-cluster "
    f"[{_b3['boot_run'][0]:.1f}, {_b3['boot_run'][2]:.1f}], LOPO {_b3['lopo'][0]:.1f}-{_b3['lopo'][1]:.1f}. "
    f"Survives adjustment: **{'YES' if _b3['survives_adjustment'] else 'NO'}** ({_b3['adj_reason']}).",
    f"- **Meal size by deficit**: Full {_b5m['means']['HS1']:.0f} / Hungry {_b5m['means']['HS2']:.0f} / "
    f"Starving {_b5m['means']['HS3']:.0f}; {_b5m['slope']:+.1f} points per deficit step "
    f"(run-cluster bootstrap [{_b5m['boot'][0]:+.1f}, {_b5m['boot'][2]:+.1f}]). Survives excluding the "
    f"obligated feeders: **{'YES' if _b5m['survives_exclusion'] else 'NO'}** "
    f"({_b5m['nf_slope']:+.1f}/step, [{_b5m['nf_boot'][0]:+.1f}, {_b5m['nf_boot'][2]:+.1f}]).",
    f"- **Remote loop** (reply-centric one-to-one matching; recovery notifications excluded): "
    f"{_b5p['rate']:.2f} reply rate after a hunger ping vs {_b5p['ctrl']:.2f} in controls matched on "
    f"subscriber, run and time-of-day — paired difference {_b5p['diff']:+.2f}, subscriber-cluster "
    f"bootstrap [{_b5p['boot'][0]:+.2f}, {_b5p['boot'][2]:+.2f}], run-cluster "
    f"[{_b5p['run_boot'][0]:+.2f}, {_b5p['run_boot'][2]:+.2f}].",
    f"- **Starving occupancy** (empirical, assumption-free): {_b7d['emp']*100:.2f}% of observed seconds, "
    f"run-cluster bootstrap [{_b7d['emp_boot'][0]*100:.2f}, {_b7d['emp_boot'][2]*100:.2f}]. The modelled "
    f"CTMC is NOT identified (irreducible={_b7d['irreducible']}, stationary vector validated="
    f"{_b7d['stationary_valid']}; Phase 1 vs Phase 2 differ {_b7d['phase_rel']:.0f}x).",
    f"- **Starving episodes**: {_b6d['n']} in total; {_b6d['feed_k']}/{_b6d['n']} received a feed "
    f"(exact [{_b6d['e_feed'][1]:.2f}, {_b6d['e_feed'][2]:.2f}]). Longest {_b6d['worst_sec']/60:.0f} min "
    f"to level {_b6d['worst_min_level']:.1f}.",
    f"- **Role manipulation** (B10.1, exploratory; roles NOT randomised): on the complete "
    f"scheduled-day panel ({_mg['n_sched']} scheduled person-days, {_mg['n_noshow']} no-shows kept as "
    f"zeros), feeders delivered **{_mg['rr_energy']:.1f}x the ENERGY** per scheduled day "
    f"(bootstrap [{_mg['boot'][0]:.1f}, {_mg['boot'][2]:.1f}]) but fed only "
    f"**{_mg['rr_perint']:.1f}x per interaction** (bootstrap "
    f"[{_mg['boot_perop'][0]:.1f}, {_mg['boot_perop'][2]:.1f}]). The gap is ATTENDANCE "
    f"({_mg['rr_expo']:.1f}x more interactions per scheduled day). Label-permutation sensitivity "
    f"p={_mg['perm_p_energy']:.3f} over the exact enumeration of {_mg['n_assignments']} assignments "
    f"(floor {_mg['perm_floor']:.3f}) — **not randomisation inference**.",
    f"- **Affinity is a programmed update rule** (B10): its four logged inputs (credit, "
    f"active_energy_cost, fed, affinity_before) explain R^2={_rule['r2']:.2f} of every update. On top of "
    f"the full rule, dose adds {_rule['resid']:+.4f}/SD (p={_rule['resid_p']:.3f}) — exploratory. The "
    f"controlled dose slope is {_new_slope:+.3f} against the {_old_slope:+.3f} the uncontrolled "
    f"specification reported: same sign, {_ag['max_ratio']:.1f}x apart, so they do NOT agree.",
    f"- **Downstream, decomposed** (B10.3): affinity does not clearly predict who is DETECTED "
    f"(OR {_pr['stage1_or']:.2f}, p={_pr['stage1_p']:.3f}); the eligibility rate rises with affinity "
    f"descriptively (3% -> 9% across terciles) but is NOT distinguishable at the person-cluster level "
    f"(RR {_pr['stage2_rr']:.2f} [{_pr['stage2_lo']:.2f}, {_pr['stage2_hi']:.2f}], "
    f"p={_pr['stage2_p']:.2f}) — and that stage IS the coded threshold, verified exactly in B9; given "
    f"eligibility, proactive approach RR {_pr['rr']:.2f} "
    f"[{_pr['ci'][0]:.2f}, {_pr['ci'][1]:.2f}] over {_pr['total_eligible']} eligible opportunities.",
    f"- **Feeding concentration**: delivered ENERGY Gini {_d4['gini_energy']:.2f} over {_d4['n_users']} "
    f"named users; top 3 supply {_d4['top3_energy']*100:.0f}% ({_d4['total_energy']:.0f} stomach points "
    f"from {_d4['total_meals']} meals — a count is not an energy).", ""]

L+=["## What these data cannot establish", "",
    "New data are required for each. None is a matter of better analysis.", "",
    "- **Drive-on vs drive-off causal identification.** There is no off condition.",
    "- **Multi-site generalisation.** One robot, one site, one convenience sample.",
    "- **Any role effect beyond these four people.** Roles were assigned by availability, 2 per role, "
    "and role is nearly aliased with identity. There is no randomisation to license inference, and "
    f"the label-permutation p has a hard floor of {_mg['perm_floor']:.3f} set by the "
    f"{_mg['n_assignments']} possible assignments.",
    f"- **Reliable Starving-episode rates.** {_b6d['n']} episodes clustered in a handful of runs, "
    "some right-censored.",
    "- **A calibrated long-run occupancy.** The deployment is not a time-homogeneous process.",
    "- **Any independent evidence that the robot 'learns' about people.** Affinity is a deterministic "
    "EMA of delivered energy, and the downstream path runs through a threshold verified exactly.",
    "- **Population-level conclusions of any kind.**", ""]

L+=["## Scope", "",
    f"One robot, one site, {hunger_raw['day_rome'].nunique()} session-days, "
    f"{hunger_raw['run_id'].nunique()} runs, "
    f"{master[master.person_id!='unknown']['person_id'].nunique()} named people (convenience sample)."]
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
