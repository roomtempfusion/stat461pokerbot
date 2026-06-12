"""
poker_agent_v5.py — Claude Poker Agent: v5 Full System
Based on llm_integration_v5.ipynb, styled like poker_agent_ltm.py.

New in v5 vs v4:
  • Up to 7 players (configurable)
  • Button rotation per hand
  • ev_based_strategy — Monte Carlo equity bot
  • random_raiser — 10% chance raise bot
  • humanize_action_history — readable logs
  • extract_player_summary — per-player eval snippets
  • Smarter LTM prompt (trim, summary-only for new players)
  • run_game() encapsulation
  • Stack chart (replaces external plotter)

Run:
    pip install streamlit anthropic pokerkit python-dotenv pandas
    streamlit run poker_agent_v5.py
"""

import os, random, json, re, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pokerkit import (
    NoLimitTexasHoldem, Automation,
    Card, Deck, StandardHighHand,
    HoleDealing, CardBurning, ChipsPushing,
    HoleCardsShowingOrMucking,
    parse_range, calculate_hand_strength,
)
from anthropic import Anthropic, beta_tool

load_dotenv()


# ══════════════════════════════════════════════════════════════════
#  MediumTermMemory
# ══════════════════════════════════════════════════════════════════
class MediumTermMemory:
    def __init__(self, buffer_dir="./memory/medium_term"):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self._game_dir = None
        self.game_id = ""
        self.hands_played = 0

    def new_game(self, game_id, players):
        self.game_id = game_id
        self.hands_played = 0
        self._game_dir = self.buffer_dir / game_id
        self._game_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("metadata.json", {
            "game_id": game_id,
            "started_at": datetime.now().isoformat(),
            "players": players,
        })
        self._write_json("stats.json", {
            "hands_played": 0,
            "players": {p: self._empty_player_stats() for p in players},
        })
        self._append_to("raw_log.txt",
            f"=== GAME {game_id} STARTED | {datetime.now().isoformat()} ===\n"
            f"Players: {', '.join(players)}\n\n")
        self._append_to("trends.txt", f"=== TREND LOG | GAME {game_id} ===\n\n")

    def ingest_hand(self, action_history, reasoning, chip_changes):
        self.hands_played += 1
        for item in action_history:
            self._append_to("raw_log.txt", f"{item}\n")
        self._append_to("raw_log.txt", "\nReasonings:\n")
        for r in reasoning:
            self._append_to("raw_log.txt", f"{r}\n")
        stats = self._read_json("stats.json")
        stats["hands_played"] = self.hands_played
        stats["chip_changes"] = chip_changes
        self._write_json("stats.json", stats)

    def log_trend(self, observation, hand_number=None):
        ref = f"[After hand #{hand_number or self.hands_played}]"
        self._append_to("trends.txt", f"{ref}\n{observation.strip()}\n\n")

    def read_digest(self, recent_trends=5):
        if not self._game_dir:
            return "[Medium term memory is empty — no active game]"
        stats = self._read_json("stats.json")
        n = max(stats.get("hands_played", 0), 1)
        lines = [f"=== TABLE DIGEST | Game {self.game_id} | {n} hands ===\n"]
        for player, b in stats.get("players", {}).items():
            lines.append(
                f"  {player}: VPIP={100*b.get('vpip',0)/n:.0f}% | "
                f"PFR={100*b.get('preflop_raise',0)/n:.0f}% | "
                f"Net={b.get('net_chips',0):+d}")
        trends_text = self._read_text("trends.txt")
        entries = [e.strip() for e in trends_text.split("\n\n")
                   if e.strip() and not e.strip().startswith("=== TREND")]
        lines.append("")
        if entries:
            lines.append(f"--- Recent Observations (last {recent_trends}) ---")
            for e in entries[-recent_trends:]:
                lines.append(e)
        else:
            lines.append("--- No trend observations yet ---")
        return "\n".join(lines)

    def close_game(self, final_stacks, game_critique=""):
        stacks_str = " | ".join(f"{p}: {c}" for p, c in final_stacks.items())
        self._append_to("raw_log.txt",
            f"\n=== GAME {self.game_id} CLOSED | {datetime.now().isoformat()} ===\n"
            f"Final stacks: {stacks_str}\nHands: {self.hands_played}\n"
            f"Critique: {game_critique or 'pending'}\n{'='*60}\n")

    @staticmethod
    def _empty_player_stats():
        return {"vpip":0,"preflop_raise":0,"preflop_3bet":0,
                "continuation_bet":0,"went_to_showdown":0,"won_hand":0,"net_chips":0}

    def _path(self, f):
        assert self._game_dir, "Call new_game() first."
        return self._game_dir / f

    def _append_to(self, f, text):
        with open(self._path(f), "a", encoding="utf-8") as fp:
            fp.write(text)

    def _read_text(self, f):
        p = self._path(f)
        return p.read_text("utf-8") if p.exists() else ""

    def _read_json(self, f):
        p = self._path(f)
        if not p.exists(): return {}
        try: return json.loads(p.read_text("utf-8"))
        except: return {}

    def _write_json(self, f, data):
        with open(self._path(f), "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)


# ══════════════════════════════════════════════════════════════════
#  LongTermMemory  (v5 full version with .txt profiles + index)
# ══════════════════════════════════════════════════════════════════
class LongTermMemory:
    INDEX_FILENAME  = "index.json"
    PLAYERS_DIRNAME = "players"
    SELF_FILENAME   = "self.txt"
    _NO_PROFILE = "No prior profile for this player."
    _SELF_SCAFFOLD = (
        "# Self Profile\n## Last updated: never\n## Games played: 0\n\n"
        "## Current strategy\n(no strategy notes yet)\n\n"
        "## Known leaks\n(none identified yet)\n\n"
        "## Lessons learned\n(none yet)\n\n"
        "## Open questions\n(none yet)\n"
    )
    _HEADER_RE  = re.compile(r"^##\s*([^:]+?)\s*:\s*(.*?)\s*$")
    _SECTION_RE = re.compile(r"^##\s+([^:]+?)\s*$")

    def __init__(self, root="./memory/long_term"):
        self.root = Path(root)
        self.players_dir = self.root / self.PLAYERS_DIRNAME
        self.index_path  = self.root / self.INDEX_FILENAME
        self.self_path   = self.root / self.SELF_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.players_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict = self._load_index()
        self._session_active = False

    def new_session(self, game_id):
        self._index = self._load_index()
        self._session_active = True
        self._session_game_id = game_id

    def close_session(self):
        self.rebuild_index()
        self._session_active = False

    def lookup_player(self, player_id):
        path = self._player_path(player_id)
        if not path.exists(): return self._NO_PROFILE
        return path.read_text("utf-8")

    def read_self_profile(self):
        if not self.self_path.exists():
            self.self_path.write_text(self._SELF_SCAFFOLD, "utf-8")
        return self.self_path.read_text("utf-8")

    def list_known_players(self, filter_tags=None):
        wanted = None if filter_tags is None else {t.strip().lower() for t in filter_tags if t.strip()}
        results = []
        for pid, entry in self._index.items():
            if wanted is not None:
                ptags = {t.lower() for t in entry.get("tags", [])}
                if not (wanted & ptags): continue
            results.append({"id": pid, **entry})
        results.sort(key=lambda r: (r.get("last_seen",""), r["id"]), reverse=True)
        return results

    def log_player_update(self, player_id, observation, tags=None, summary=None):
        path = self._player_path(player_id)
        timestamp = datetime.now().isoformat()
        game_ref = getattr(self, "_session_game_id", "unknown")
        if not path.exists():
            header = (
                f"# Player Profile: {player_id}\n"
                f"## Tags: {', '.join(tags or [])}\n"
                f"## Last seen: {timestamp}\n"
                f"## Games observed: 1\n\n"
                f"## Summary\n(no summary yet)\n\n"
                f"## Observations\n"
            )
            self._write(path, header, "w")
            games_observed = 1
        else:
            games_observed = self._index.get(player_id, {}).get("games_observed", 0) + 1
        entry = f"[{timestamp} | {game_ref}]\n{observation.strip()}\n\n"
        self._write(path, entry, "a")
        self._patch_headers(path, {
            "last seen": timestamp,
            "games observed": str(games_observed),
            "tags": ", ".join(self._merge_tags(self._index.get(player_id,{}).get("tags",[]), tags or [])),
        })
        if summary is not None:
            self._patch_section(path, "Summary", summary)
        self._index[player_id] = self._parse_profile(path.read_text("utf-8"))
        self._save_index()

    def log_self_update(self, observation, section="Lessons learned"):
        _ = self.read_self_profile()
        timestamp = datetime.now().isoformat()
        game_ref = getattr(self, "_session_game_id", "unknown")
        entry = f"[{timestamp} | {game_ref}] {observation.strip()}\n"
        text = self.self_path.read_text("utf-8")
        target = f"## {section}"
        if target not in text:
            text += f"\n## {section}\n{entry}"
        else:
            lines = text.splitlines(keepends=True)
            out = []
            for line in lines:
                out.append(line)
                if line.strip() == target:
                    out.append(entry)
            text = "".join(out)
        self.self_path.write_text(text, "utf-8")
        self._patch_headers(self.self_path, {"last updated": timestamp})

    def rebuild_index(self):
        new_index = {}
        for txt_path in self.players_dir.glob("*.txt"):
            pid = txt_path.stem
            try:
                text = txt_path.read_text("utf-8")
            except OSError:
                continue
            new_index[pid] = self._parse_profile(text)
        self._index = new_index
        self._save_index()

    # ── helpers ───────────────────────────────────────────────────
    def _player_path(self, pid):
        if "/" in pid or "\\" in pid or ".." in pid:
            raise ValueError(f"Invalid player_id: {pid!r}")
        return self.players_dir / f"{pid}.txt"

    def _write(self, path, text, mode="a"):
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)

    def _load_index(self):
        if not self.index_path.exists(): return {}
        try:
            raw = json.loads(self.index_path.read_text("utf-8"))
            return raw if isinstance(raw, dict) else {}
        except: return {}

    def _save_index(self):
        self.index_path.write_text(json.dumps(self._index, indent=2, sort_keys=True), "utf-8")

    def _parse_profile(self, text):
        entry = {"tags":[], "last_seen":"", "games_observed":0, "summary":""}
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = self._HEADER_RE.match(lines[i])
            if m:
                key, val = m.group(1).strip().lower(), m.group(2).strip()
                if key == "tags": entry["tags"] = [t.strip() for t in val.split(",") if t.strip()]
                elif key == "last seen": entry["last_seen"] = val
                elif key == "games observed":
                    try: entry["games_observed"] = int(val)
                    except: pass
                i += 1; continue
            m = self._SECTION_RE.match(lines[i])
            if m and m.group(1).strip().lower() == "summary":
                j = i + 1; buf = []
                while j < len(lines) and not lines[j].lstrip().startswith("##"):
                    buf.append(lines[j]); j += 1
                entry["summary"] = " ".join(l.strip() for l in buf if l.strip())
                i = j; continue
            i += 1
        return entry

    def _patch_headers(self, path, updates):
        text = path.read_text("utf-8")
        lines = text.splitlines(keepends=True)
        out = []
        for line in lines:
            m = self._HEADER_RE.match(line.rstrip("\n"))
            if m and m.group(1).strip().lower() in updates:
                line = f"## {m.group(1)}: {updates[m.group(1).strip().lower()]}\n"
            out.append(line)
        path.write_text("".join(out), "utf-8")

    def _patch_section(self, path, section, body):
        text = path.read_text("utf-8")
        lines = text.splitlines(keepends=True)
        out = []; target = f"## {section}"; i = 0
        while i < len(lines):
            out.append(lines[i])
            if lines[i].rstrip("\n").strip() == target:
                out.append(f"{body.strip()}\n\n")
                i += 1
                while i < len(lines) and not lines[i].lstrip().startswith("##"):
                    i += 1
                continue
            i += 1
        path.write_text("".join(out), "utf-8")

    @staticmethod
    def _merge_tags(existing, new):
        seen = set(); merged = []
        for tag in existing + new:
            if tag.lower() not in seen:
                seen.add(tag.lower()); merged.append(tag)
        return merged


# ══════════════════════════════════════════════════════════════════
#  Page config & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Claude Poker Agent — v5", page_icon="♠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Cinzel+Decorative:wght@700&family=IBM+Plex+Mono:wght@400;600&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

:root {
  --gold:      #d4a843;
  --gold-dim:  #8a6a20;
  --gold-glow: rgba(212,168,67,.35);
  --green:     #1db954;
  --green-dim: #0d5c28;
  --green-glow:rgba(29,185,84,.3);
  --red:       #e8344a;
  --blue:      #4fa3e0;
  --purple:    #9b72cf;
  --felt:      #0b2a1a;
  --bg:        #06080f;
  --surface:   #0d1117;
  --surface2:  #161c27;
  --border:    rgba(212,168,67,.12);
  --border-dim:rgba(255,255,255,.05);
  --text:      #c8bfa8;
  --text-dim:  #5a5040;
}

html, body, [data-testid="stApp"] { background: var(--bg) !important; color: var(--text); }
* { box-sizing: border-box; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; }

[data-testid="stMain"] {
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, #0e3320 0%, transparent 65%),
    repeating-linear-gradient(60deg, transparent, transparent 2px, rgba(255,255,255,.008) 2px, rgba(255,255,255,.008) 4px),
    repeating-linear-gradient(-60deg, transparent, transparent 2px, rgba(255,255,255,.006) 2px, rgba(255,255,255,.006) 4px),
    var(--bg) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080c14 0%, #06080f 100%) !important;
  border-right: 1px solid var(--border) !important;
}

h1 {
  font-family: 'Cinzel Decorative', serif !important;
  font-size: 2.1rem !important; font-weight: 700 !important;
  letter-spacing: .08em !important; color: var(--gold) !important;
  text-shadow: 0 0 20px var(--gold-glow), 0 0 60px rgba(212,168,67,.15), 0 2px 4px rgba(0,0,0,.8);
}
h2, h3 { font-family: 'Cinzel', serif !important; color: #9a8860 !important; letter-spacing: .06em !important; }
p, li, span, label, div { font-family: 'Crimson Pro', Georgia, serif; font-size: 1.05rem; }

[data-testid="stWidgetLabel"] p,
.stTextInput label, .stNumberInput label, .stSlider label, .stCheckbox label, .stSelectbox label {
  font-family: 'Cinzel', serif !important; font-size: .72rem !important;
  letter-spacing: .1em !important; color: var(--gold-dim) !important; text-transform: uppercase !important;
}

input, [data-baseweb="input"] input {
  background: var(--surface2) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important; border-radius: 6px !important;
}

.stButton > button {
  font-family: 'Cinzel', serif !important; font-size: .78rem !important;
  letter-spacing: .12em !important; text-transform: uppercase !important; border-radius: 6px !important; transition: all .2s !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #b8862e 0%, #d4a843 50%, #b8862e 100%) !important;
  border: none !important; color: #06080f !important; font-weight: 700 !important;
  box-shadow: 0 0 20px var(--gold-glow), 0 2px 8px rgba(0,0,0,.6) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 35px var(--gold-glow), 0 4px 16px rgba(0,0,0,.8) !important; transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
  background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text) !important;
}

[data-testid="stTabs"] [role="tab"] {
  font-family: 'Cinzel', serif !important; font-size: .72rem !important;
  letter-spacing: .1em !important; text-transform: uppercase !important; color: var(--text-dim) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important;
  text-shadow: 0 0 12px var(--gold-glow) !important;
}

[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--green-dim), var(--green)) !important;
  box-shadow: 0 0 8px var(--green-glow) !important;
}

/* ── Cards ── */
.card {
  display: inline-flex; align-items: center; justify-content: center;
  width: 52px; height: 72px; margin: 4px; border-radius: 10px;
  font-family: 'Cinzel', serif; font-weight: 700; font-size: 1.1rem;
  background: linear-gradient(145deg, #fefefe 0%, #f0ebe0 100%);
  color: #1a1a1a; border: 1px solid rgba(0,0,0,.15);
  box-shadow: 0 6px 20px rgba(0,0,0,.7), 0 2px 4px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.9);
  transition: transform .15s ease; vertical-align: middle;
}
.card:hover { transform: translateY(-4px); }
.card-red { color: #b91c1c; }
.card-back {
  background: linear-gradient(145deg, #1a3a6e 0%, #0f2347 100%) !important;
  color: rgba(255,255,255,.1) !important; border-color: #2d4f8a !important;
  font-size: 1.5rem !important;
  background-image: repeating-linear-gradient(45deg, rgba(255,255,255,.03) 0px, rgba(255,255,255,.03) 2px, transparent 2px, transparent 8px) !important;
}

/* ── Player boxes ── */
.player-box {
  position: relative;
  background: linear-gradient(160deg, var(--surface2) 0%, var(--surface) 100%);
  border-radius: 14px; padding: 16px 12px; text-align: center;
  border: 1px solid var(--border-dim);
  box-shadow: 0 4px 24px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.04);
  overflow: hidden;
}
.player-box::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(212,168,67,.2), transparent);
}
.player-box.agent {
  border-color: rgba(29,185,84,.35);
  box-shadow: 0 4px 24px rgba(0,0,0,.5), 0 0 30px rgba(29,185,84,.08), inset 0 1px 0 rgba(29,185,84,.1);
}
.player-box.agent::before { background: linear-gradient(90deg, transparent, rgba(29,185,84,.4), transparent); }
.player-box.button-seat { border-color: rgba(212,168,67,.4); }
.player-name { font-family: 'Cinzel', serif; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #5a5040; margin-bottom: 10px; }
.player-name.agent-name { color: var(--green); text-shadow: 0 0 10px var(--green-glow); }
.chip-count { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: var(--gold); margin-top: 10px; text-shadow: 0 0 8px var(--gold-glow); }
.dealer-btn { display: inline-block; background: var(--gold); color: #06080f; border-radius: 50%; width: 18px; height: 18px; font-size: .65rem; font-weight: 700; line-height: 18px; text-align: center; margin-left: 4px; font-family: 'Cinzel', serif; }

/* ── Pot box ── */
.pot-box {
  position: relative;
  background: linear-gradient(160deg, #071a0e 0%, #060d16 100%);
  border: 1px solid rgba(29,185,84,.3); border-radius: 18px; padding: 20px; text-align: center;
  box-shadow: 0 0 40px rgba(29,185,84,.08), 0 8px 32px rgba(0,0,0,.6), inset 0 1px 0 rgba(29,185,84,.15);
}
.pot-box::after {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 0%, rgba(29,185,84,.06) 0%, transparent 70%); pointer-events: none;
}
.pot-label { font-family: 'Cinzel', serif; font-size: .65rem; letter-spacing: .2em; color: var(--green); text-transform: uppercase; opacity: .7; }
.pot-amount { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 600; color: #fff; line-height: 1.1; margin: 4px 0; }
.pot-chips { font-size: 1.3rem; margin-bottom: 4px; }
.street-pill {
  display: inline-block; padding: 4px 18px; border-radius: 20px;
  background: rgba(212,168,67,.07); border: 1px solid rgba(212,168,67,.25);
  color: var(--gold); font-family: 'Cinzel', serif; font-size: .66rem; letter-spacing: .18em; text-transform: uppercase;
}

/* ── Badges ── */
.badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-family: 'Cinzel', serif; font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; }
.badge-fold  { background: rgba(232,52,74,.12); color: #f07080; border: 1px solid rgba(232,52,74,.3); }
.badge-call  { background: rgba(79,163,224,.12); color: #7ec8f5; border: 1px solid rgba(79,163,224,.3); }
.badge-check { background: rgba(29,185,84,.12); color: #5de898; border: 1px solid rgba(29,185,84,.3); }
.badge-raise { background: rgba(212,168,67,.12); color: var(--gold); border: 1px solid rgba(212,168,67,.35); box-shadow: 0 0 8px rgba(212,168,67,.15); }

/* ── Info boxes ── */
.memory-box {
  background: linear-gradient(160deg, #0d0d1f 0%, #0a0a18 100%);
  border: 1px solid rgba(155,114,207,.2); border-radius: 10px; padding: 14px 16px;
  font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #b09ed4;
  line-height: 1.7; white-space: pre-wrap; max-height: 260px; overflow-y: auto;
}
.ltm-box {
  background: linear-gradient(160deg, #100d00 0%, #0a0800 100%);
  border: 1px solid rgba(212,168,67,.18); border-radius: 10px; padding: 14px 16px;
  font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #c8a84a;
  line-height: 1.7; white-space: pre-wrap; max-height: 220px; overflow-y: auto;
}
.judge-box {
  background: linear-gradient(160deg, #040d18 0%, #030a12 100%);
  border-left: 3px solid var(--blue); border-radius: 0 10px 10px 0; padding: 14px 18px;
  font-family: 'Crimson Pro', Georgia, serif; font-style: italic; font-size: 1rem; color: #7ab8e8; line-height: 1.7;
}
.reasoning-box {
  background: linear-gradient(160deg, #051208 0%, #030d05 100%);
  border-left: 3px solid var(--green); border-radius: 0 10px 10px 0; padding: 14px 18px;
  font-family: 'Crimson Pro', Georgia, serif; font-size: 1rem; color: #6ed48a; line-height: 1.7;
}
.equity-box {
  background: linear-gradient(160deg, #0e0814 0%, #090510 100%);
  border-left: 3px solid var(--purple); border-radius: 0 10px 10px 0; padding: 10px 16px;
  font-family: 'IBM Plex Mono', monospace; font-size: .82rem; color: #b998e8; line-height: 1.5;
}

b { font-family: 'Cinzel', serif; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--gold-dim); }

.divider {
  border: none; height: 1px; margin: 24px 0;
  background: linear-gradient(90deg, transparent 0%, rgba(212,168,67,.08) 20%, rgba(212,168,67,.2) 50%, rgba(212,168,67,.08) 80%, transparent 100%);
}

[data-testid="stExpander"] { background: var(--surface2) !important; border: 1px solid var(--border-dim) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-family: 'Cinzel', serif !important; font-size: .78rem !important; color: #7a6a4a !important; }

[data-testid="stMetric"] { background: var(--surface2); border: 1px solid var(--border-dim); border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricLabel"] p { font-family: 'Cinzel', serif !important; font-size: .65rem !important; letter-spacing: .14em !important; color: var(--text-dim) !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; color: var(--gold) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 3px; }

@keyframes float {
  0%, 100% { transform: translateY(0px); opacity: .15; }
  50%       { transform: translateY(-10px); opacity: .28; }
}
.suit-float { display: inline-block; animation: float 4s ease-in-out infinite; }
.suit-float:nth-child(2) { animation-delay: 1s; }
.suit-float:nth-child(3) { animation-delay: 2s; }
.suit-float:nth-child(4) { animation-delay: 3s; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Constants & helpers
# ══════════════════════════════════════════════════════════════════
MAX_PLAYERS = 7
ALL_PLAYER_LABELS = ["Hero (Claude 🤖)"] + [f"P{i+1}" for i in range(1, MAX_PLAYERS)]
PLAYER_LABEL_MAP  = {i: f"P{i+1}" for i in range(1, MAX_PLAYERS)}

STRATEGY_NAMES = ["Pocket Pair 🃏", "Punter 🎲", "Tight Passive 🧊", "Random Raiser 🎰", "EV-Based 🧠", "EV-Conservative 📐"]


def to_short(c):
    r = repr(c)
    if len(r) == 2: return r
    s = str(c)
    if "(" in s and ")" in s: return s[s.index("(")+1:s.index(")")]
    return s[:2] if len(s) >= 2 else s


def card_html(c):
    s = to_short(c)
    suit = s[-1].lower() if s else ""
    cls = "card-red" if suit in ("h", "d") else ""
    return f'<span class="card {cls}">{s}</span>'


def cards_html(cards, hidden=False):
    if hidden or not cards:
        return '<span class="card card-back">🂠</span>'
    return "".join(card_html(c) for c in cards)


def street_name(idx):
    return ["Preflop", "Flop", "Turn", "River"][idx] if idx is not None else "—"


def badge(action):
    a = action.lower()
    return f'<span class="badge badge-{a}">{a.upper()}</span>'


def position_name(seat):
    return ["SB", "BB"][seat] if seat < 2 else "BTN" if seat == 0 else "other"


def get_visible_ops(state):
    try:
        winners = set()
        for op in state.operations:
            if isinstance(op, ChipsPushing):
                winners = {i for i, amt in enumerate(op.amounts) if amt > 0}
        showdown = sum(1 for p in state.statuses if p) > 1
        out = []
        for op in state.operations:
            if isinstance(op, (HoleDealing, CardBurning)): continue
            if isinstance(op, HoleCardsShowingOrMucking):
                if showdown or op.player_index in winners: out.append(op)
            else: out.append(op)
        return out
    except Exception:
        return [op for op in state.operations if not isinstance(op, (HoleDealing, CardBurning))]


def humanize_action_history(ops, active_indices, players):
    history = []
    for op in ops:
        s = str(op)
        def repl(m):
            seat = int(m.group(1))
            if seat < len(active_indices):
                gi = active_indices[seat]
                return f'player={players[gi] if gi < len(players) else f"P{gi+1}"}'
            return m.group(0)
        s = re.sub(r'player_index=(\d+)', repl, s)
        history.append(s)
    return history


# ══════════════════════════════════════════════════════════════════
#  LTM helpers  (v5)
# ══════════════════════════════════════════════════════════════════
def extract_tags_from_reasoning(reasoning_texts):
    tag_map = {}
    for text in reasoning_texts:
        if not text: continue
        words = text.split()
        for i, word in enumerate(words):
            label = word.strip(",:").upper()
            if not (label.startswith("P") and label[1:].isdigit()): continue
            if label[1] not in "123456789": continue
            for chunk in words[i+1:i+11]:
                if "/10" not in chunk: continue
                sp = chunk.split("/10")[0].strip()
                if not sp.isdigit(): break
                score = int(sp)
                tag = "aggressive" if score >= 7 else ("passive" if score <= 3 else "balanced")
                tag_map.setdefault(label, []).append(tag)
                break
    return {pid: [Counter(tags).most_common(1)[0][0]] for pid, tags in tag_map.items()}


def extract_player_summary(eval_text, pid):
    paragraphs = [p.strip() for p in eval_text.split("\n\n") if p.strip()]
    for para in paragraphs:
        header = para.split(":", 1)[0]
        if pid in re.findall(r"P\d", header):
            return para
    return eval_text


def run_post_hand_reflection(ltm, eval_text, active_player_ids, reasonings, payoffs):
    if not eval_text: return
    reasoning_texts = [r[2] for r in reasonings if len(r) >= 3]
    tag_map = extract_tags_from_reasoning(reasoning_texts)
    for pid in active_player_ids:
        tags = tag_map.get(pid, [])
        ltm.log_player_update(
            player_id=pid, observation=eval_text, tags=tags,
            summary=extract_player_summary(eval_text, pid))
    hero_payoff = payoffs[0] if payoffs else 0
    outcome = "won" if hero_payoff > 0 else ("lost" if hero_payoff < 0 else "broke even")
    hero_rsn = " | ".join(f"{r[0]} {r[1] if r[1] else ''}: {r[2]}" for r in reasonings if len(r) >= 3)
    ltm.log_self_update(
        observation=(f"Hand outcome: {outcome} ({hero_payoff:+,} chips). "
                     f"Opponent eval: {eval_text[:300]}. "
                     f"Hero reasoning: {hero_rsn[:400]}."),
        section="Lessons learned")


# ══════════════════════════════════════════════════════════════════
#  Anthropic tools
# ══════════════════════════════════════════════════════════════════
@beta_tool
def take_action(action: str, reasoning: str, amount: float = 0) -> str:
    """Submit your poker action and explain in max 3 sentences.

    Args:
        action: One of fold, check, call, raise
        amount: Raise amount in chips (required if action is raise).
        reasoning: Why you took this action. Rate each opponent's aggression 1-10 if past preflop.
    """
    return json.dumps({"status":"accepted","action":action,"amount":amount,"reasoning":reasoning})


@beta_tool
def get_equity_strength(hole_cards: str, board_cards: str, active_players: int) -> str:
    """Calculate win probability via Monte Carlo simulation.

    Args:
        hole_cards: Your two hole cards (e.g., 'AsKc').
        board_cards: Board cards (e.g., 'Td9h2c'). Empty string for preflop.
        active_players: Total players in the simulation.
    """
    board_input = tuple(Card.parse(board_cards)) if board_cards else ()
    equity = calculate_hand_strength(
        active_players, parse_range(hole_cards), board_input,
        active_players, 5, Deck.STANDARD, (StandardHighHand,), sample_count=500)
    win_pct = round(equity * 100, 1)
    return json.dumps({"status":"success","equity":equity,"message":f"You have a {win_pct}% chance of winning."})


@beta_tool
def generate_observation(evaluation: str) -> str:
    """Evaluate the action history of the previous poker hand.

    Args:
        evaluation: Specific analysis of non-Hero players — mistakes, tendencies, bluffs.
    """
    return json.dumps({"status":"accepted","evaluation":evaluation})


# ══════════════════════════════════════════════════════════════════
#  LLM calls (v5)
# ══════════════════════════════════════════════════════════════════
JUDGE_SYSTEM_PROMPT = (
    "You are an evaluating agent for Texas Hold'em. Evaluate non-Hero players "
    "(P2 and up) to help Hero defeat them. Index 0 is Hero, index 1 is P2, etc. "
    "Use the generate_observation tool to submit."
)


def build_system_prompt(ltm, active_player_ids):
    """v5: smarter LTM injection — trim self-notes, summary-only for new opponents."""
    base = (
        f"You are a poker agent playing {len(active_player_ids)+1}-handed Texas Hold'em No-Limit.\n"
        "Blinds 1000/2000. You are Hero. Think about pot odds, position, hand strength, stack depth.\n"
        "Use the take_action tool to act. Use get_equity_strength only for tough spots.\n"
        "Use self-notes and opponent notes to inform your decisions."
    )
    # Self-notes: last 20 lines only
    full_self = ltm.read_self_profile()
    self_lines = full_self.strip().splitlines()
    snippet = "\n".join(self_lines[-20:])
    ltm_section = f"\n\n--- SELF NOTES (recent) ---\n{snippet}\n---"

    known = {p["id"]: p for p in ltm.list_known_players()}
    for pid in active_player_ids:
        if pid in known:
            entry = known[pid]
            games = entry.get("games_observed", 0)
            if games >= 2:
                profile = ltm.lookup_player(pid)
                lines = profile.strip().splitlines()
                trimmed = "\n".join(lines[:8] + ["..."] + lines[-15:]) if len(lines) > 25 else profile
                ltm_section += f"\n--- {pid} (seen {games}x) ---\n{trimmed}\n---"
            else:
                tags = ", ".join(entry.get("tags", [])) or "unknown"
                summary = entry.get("summary", "")[:120]
                ltm_section += f"\n--- {pid}: tags=[{tags}] summary={summary} ---"
        else:
            ltm_section += f"\n--- {pid}: no prior data ---"
    return base + ltm_section


def run_turn(obs_json, api_key, ltm, active_player_ids):
    client = Anthropic(api_key=api_key)
    runner = client.beta.messages.tool_runner(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=[take_action, get_equity_strength],
        system=build_system_prompt(ltm, active_player_ids),
        messages=[{"role":"user","content":(
            "It's your turn to act. Here is the current game state:\n\n"
            f"```json\n{obs_json}\n```\n\nDecide your action."
        )}],
    )
    agent_action = None
    usage = None
    equity_calls = []
    for message in runner:
        usage = message.usage
        for block in message.content:
            if block.type == "tool_use":
                if block.name == "take_action":
                    agent_action = block.input
                elif block.name == "get_equity_strength":
                    try:
                        r = json.loads(get_equity_strength(
                            block.input.get("hole_cards",""),
                            block.input.get("board_cards",""),
                            block.input.get("active_players",2)))
                        equity_calls.append(r.get("message",""))
                    except Exception: pass
    if agent_action is None:
        agent_action = {"action":"check","amount":0,"reasoning":"Defaulted to check."}
    return agent_action, usage, equity_calls


def run_judge(action_history, api_key):
    client = Anthropic(api_key=api_key)
    trimmed = action_history[-30:]
    runner = client.beta.messages.tool_runner(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        tools=[generate_observation],
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role":"user","content":(
            "Here is the action history of the game.\n\n"
            f"```json\n{json.dumps(trimmed)}\n```\n\nGenerate your observations."
        )}],
    )
    eval_text = ""
    usage = None
    for message in runner:
        usage = message.usage
        for block in message.content:
            if block.type == "tool_use" and block.name == "generate_observation":
                eval_text = block.input.get("evaluation", "")
    return eval_text, usage


# ══════════════════════════════════════════════════════════════════
#  Opponent strategies  (v5: adds ev_based + random_raiser)
# ══════════════════════════════════════════════════════════════════
def pocket_pair_strategy(state, player_index):
    hole = state.hole_cards[player_index]
    is_pair = len(hole) >= 2 and hole[0].rank == hole[1].rank
    strong = {"A","K","Q","J","T","9","8","7"}
    if is_pair and hole[0].rank in strong and state.can_complete_bet_or_raise_to():
        mn = state.min_completion_betting_or_raising_to_amount
        mx = state.max_completion_betting_or_raising_to_amount
        state.complete_bet_or_raise_to(min(mn*2, mx))
    elif is_pair and state.can_check_or_call(): state.check_or_call()
    elif state.can_fold(): state.fold()
    else: state.check_or_call()


def punter_strategy(state, player_index):
    if random.random() >= 0.1:
        if state.can_fold():
            state.fold()
        else:
            state.check_or_call()
        return
    if state.can_complete_bet_or_raise_to():
        mn = state.min_completion_betting_or_raising_to_amount
        mx = state.max_completion_betting_or_raising_to_amount
        state.complete_bet_or_raise_to(min(mn*2, mx))
    elif state.can_check_or_call(): state.check_or_call()
    elif state.can_fold(): state.fold()


def tight_passive_strategy(state, player_index):
    r = random.random()
    if r < 0.9 and state.can_fold(): state.fold()
    elif state.can_check_or_call(): state.check_or_call()
    elif state.can_fold(): state.fold()


def random_raiser(state, player_index):
    r = random.random()
    if r < 0.1 and state.can_complete_bet_or_raise_to():
        mn = state.min_completion_betting_or_raising_to_amount
        mx = state.max_completion_betting_or_raising_to_amount
        state.complete_bet_or_raise_to(min(mn*2, mx))
    elif r < 0.2 and state.can_fold(): state.fold()
    else: state.check_or_call()


def ev_based_strategy(state, player_index, ev_pot_threshold=0.5, num_simulations=300):
    """Bets/raises when equity > threshold, calls on pot odds, else folds."""
    hole = state.hole_cards[player_index]
    if len(hole) < 2:
        if state.can_check_or_call():
            state.check_or_call()
        elif state.can_fold():
            state.fold()
        return
    active_players = sum(1 for s in state.statuses if s)
    board_input = tuple(state.board_cards) if state.board_cards else ()
    try:
        equity = calculate_hand_strength(
            active_players, {frozenset(hole)}, board_input,
            2, 5, Deck.STANDARD, (StandardHighHand,), sample_count=num_simulations)
    except Exception:
        equity = 0.4
    pot = state.total_pot_amount or 1
    if equity > ev_pot_threshold and state.can_complete_bet_or_raise_to():
        lo = state.min_completion_betting_or_raising_to_amount
        hi = state.max_completion_betting_or_raising_to_amount
        target = int(lo + min(equity,1.0)*(hi-lo))
        state.complete_bet_or_raise_to(max(lo, min(target, hi)))
    elif state.can_check_or_call():
        call_amount = state.checking_or_calling_amount or 0
        if call_amount == 0 or equity > call_amount/(pot+call_amount):
            state.check_or_call()
        elif state.can_fold(): state.fold()
        else: state.check_or_call()
    elif state.can_fold(): state.fold()
    else: state.check_or_call()


def ev_conservative_strategy(state, player_index):
    ev_based_strategy(state, player_index, ev_pot_threshold=0.7)


STRATEGY_FN_MAP = {
    "Pocket Pair 🃏":    pocket_pair_strategy,
    "Punter 🎲":         punter_strategy,
    "Tight Passive 🧊":  tight_passive_strategy,
    "Random Raiser 🎰":  random_raiser,
    "EV-Based 🧠":       ev_based_strategy,
    "EV-Conservative 📐":ev_conservative_strategy,
}


# ══════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════
def init():
    defaults = {
        "game_log":[], "reasonings":[], "judge_log":[],
        "player_stacks":[], "hand_count":0, "phase":"idle",
        "board_cards":[], "hole_cards":[], "pot":0, "street_idx":0,
        "token_usage":[], "mem_digest":"", "hand_summaries":[],
        "ltm_display":{}, "equity_log":[], "button":0,
        "num_players":4,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()
ss = st.session_state


# ══════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ♠ Claude Poker — v5")
    st.caption("LTM · Equity Tool · EV Bots · Up to 7 Players")

    api_key = st.text_input("Anthropic API Key", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY",""))
    st.markdown("---")

    game_id    = st.text_input("Game ID", value="game_001")
    num_players = st.slider("Number of Players", 2, MAX_PLAYERS, 4)
    max_hands  = st.slider("Max hands per game", 5, 50, 30)

    st.markdown("---")
    st.markdown("**💰 Starting Stacks**")
    _all_stacks = []
    for i in range(MAX_PLAYERS):
        label = "Hero (Claude)" if i == 0 else f"P{i+1}"
        val = st.number_input(label, 1000, 500000, 20000, 1000,
                              key=f"stack_{i}", disabled=(i >= num_players))
        _all_stacks.append(val)
    starting_stacks = _all_stacks[:num_players]

    st.markdown("---")
    st.markdown("**🤖 Opponent Strategies**")
    _all_strats = []
    for i in range(1, MAX_PLAYERS):
        val = st.selectbox(f"P{i+1}", STRATEGY_NAMES,
                           index=min(i-1, len(STRATEGY_NAMES)-1),
                           key=f"strat_{i}", disabled=(i >= num_players))
        _all_strats.append(val)
    opp_strats = _all_strats[:num_players-1]

    st.markdown("---")
    show_memory = st.checkbox("Show MTM digest", True)
    show_ltm    = st.checkbox("Show LTM profiles", True)

    col1, col2 = st.columns(2)
    start = col1.button("▶ Run", use_container_width=True, type="primary")
    reset = col2.button("↺ Reset", use_container_width=True)

    if reset:
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

    if ss.mem_digest and show_memory:
        st.markdown("---")
        st.markdown("**🧠 MTM Digest**")
        st.markdown(f'<div class="memory-box">{ss.mem_digest}</div>', unsafe_allow_html=True)

    if ss.ltm_display and show_ltm:
        st.markdown("---")
        st.markdown("**🏅 LTM Profiles**")
        for pid, profile in ss.ltm_display.items():
            with st.expander(pid):
                st.markdown(f'<div class="ltm-box">{profile}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════
st.markdown('<h1>♠ Claude Poker Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family:Cinzel,serif;color:#3d3220;letter-spacing:.2em;'
    'font-size:.65rem;text-transform:uppercase;margin-top:-.4rem">'
    'v5 &nbsp;·&nbsp; LTM &nbsp;·&nbsp; EV Bots &nbsp;·&nbsp; Button Rotation &nbsp;·&nbsp; Up to 7 Players'
    '</p>', unsafe_allow_html=True)

# ── live display placeholders ──────────────────────────────────────
col_board, col_pot = st.columns([3, 1])
pot_placeholder   = col_pot.empty()
board_placeholder = col_board.empty()
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Dynamic player grid (up to 7 — show up to 4 per row)
_np = max(num_players, len(ss.player_stacks) if ss.player_stacks else num_players)
if _np <= 4:
    player_placeholders = [col.empty() for col in st.columns(_np)]
else:
    row1 = st.columns(4)
    row2 = st.columns(_np - 4)
    player_placeholders = [col.empty() for col in row1] + [col.empty() for col in row2]

st.markdown('<hr class="divider">', unsafe_allow_html=True)


def render_table(num_p=None):
    np_ = num_p or max(num_players, len(ss.player_stacks) if ss.player_stacks else num_players)
    pot_placeholder.markdown(f"""
    <div class="pot-box">
        <div class="pot-chips">🪙</div>
        <div class="pot-label">Total Pot</div>
        <div class="pot-amount">{ss.pot:,}</div>
        <div style="margin-top:10px"><span class="street-pill">{street_name(ss.street_idx)}</span></div>
        <div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;color:#2a2010;font-size:.7rem;letter-spacing:.08em">HAND #{ss.hand_count}</div>
    </div>""", unsafe_allow_html=True)

    with col_board:
        board_placeholder.markdown(
            '<b>Community Cards</b><br>' + (
                cards_html(ss.board_cards) if ss.board_cards
                else '<span style="font-family:Cinzel,serif;color:#1e1a12;font-size:.72rem;letter-spacing:.1em">— Waiting for deal —</span>'
            ), unsafe_allow_html=True)

    stacks = ss.player_stacks if ss.player_stacks else starting_stacks
    for i, ph in enumerate(player_placeholders[:np_]):
        stack  = stacks[i] if i < len(stacks) else 0
        hole   = ss.hole_cards[i] if ss.hole_cards and i < len(ss.hole_cards) else []
        is_agent = (i == 0)
        is_btn   = (i == ss.button)
        cls = "agent" if is_agent else ("button-seat" if is_btn else "")
        cards_disp = cards_html(hole) if hole else (
            '<span class="card card-back">🂠</span><span class="card card-back">🂠</span>')
        raw_label = ALL_PLAYER_LABELS[i] if i < len(ALL_PLAYER_LABELS) else f"P{i+1}"
        btn_badge = '<span class="dealer-btn">D</span>' if is_btn else ""
        name_cls = "player-name agent-name" if is_agent else "player-name"
        ph.markdown(f"""
        <div class="player-box {cls}">
            <div class="{name_cls}">{raw_label}{btn_badge}</div>
            <div style="margin-bottom:10px;min-height:76px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap">{cards_disp}</div>
            <div class="chip-count">{'💎' if is_agent else '🪙'} {stack:,}</div>
        </div>""", unsafe_allow_html=True)


render_table()


# ══════════════════════════════════════════════════════════════════
#  Run game
# ══════════════════════════════════════════════════════════════════
if start:
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
    else:
        ss.player_stacks = list(starting_stacks)
        ss.num_players   = num_players
        ss.hand_count = ss.pot = ss.street_idx = ss.button = 0
        ss.board_cards = []; ss.hole_cards = []
        ss.game_log=[]; ss.reasonings=[]; ss.judge_log=[]
        ss.token_usage=[]; ss.hand_summaries=[]
        ss.mem_digest=""; ss.ltm_display={}; ss.equity_log=[]
        ss.phase = "running"

        players = ["Hero"] + [f"P{i+1}" for i in range(1, num_players)]
        opponent_strategies = {i: STRATEGY_FN_MAP[opp_strats[i-1]] for i in range(1, num_players)}

        med = MediumTermMemory()
        med.new_game(game_id, players)
        ltm = LongTermMemory()
        ltm.new_session(game_id=game_id)

        prog   = st.progress(0, "Starting…")
        status = st.empty()

        hand_count = 0
        player_stacks = list(starting_stacks)
        button = len(player_stacks) - 1

        while (len([s for s in player_stacks if s > 0]) > 1
               and hand_count < max_hands
               and player_stacks[0] > 0):

            active = [i for i, s in enumerate(player_stacks) if s > 0]
            if len(active) < 2: break

            # Button rotation (v5)
            sb_global = next((i for i in active if i > button), active[0])
            pivot = active.index(sb_global)
            active = active[pivot:] + active[:pivot]

            active_player_ids = [PLAYER_LABEL_MAP[i] for i in active if i in PLAYER_LABEL_MAP]
            cur_stacks = tuple(player_stacks[i] for i in active)
            hand_count += 1
            ss.hand_count = hand_count
            ss.button = button
            prog.progress(hand_count / max_hands, f"Hand {hand_count}/{max_hands}")

            state = NoLimitTexasHoldem.create_state(
                (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                 Automation.BLIND_OR_STRADDLE_POSTING, Automation.CARD_BURNING,
                 Automation.HOLE_DEALING, Automation.BOARD_DEALING,
                 Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
                 Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING),
                True, 0, (1000, 2000), 2000, cur_stacks, len(active),
            )

            hand_reasonings = []
            hand_equity_calls = []
            ss.mem_digest = med.read_digest() if hand_count > 1 else ""

            while state.status:
                if state.can_deal_hole(): state.deal_hole(); continue
                if state.can_deal_board(): state.deal_board(); continue
                if state.actor_index is None: break

                global_player_index = active[state.actor_index]

                ss.pot        = state.total_pot_amount
                ss.street_idx = state.street_index or 0
                ss.board_cards = [c for street in state.board_cards for c in street]
                ss.hole_cards  = (
                    [list(h) for h in state.hole_cards]
                    + [[] for _ in range(max(0, num_players - len(state.hole_cards)))]
                )
                render_table(num_players)

                obs = {
                    "your_index": state.actor_index,
                    "pot": state.total_pot_amount,
                    "position": position_name(state.actor_index),
                    "board": [repr(c) for street in state.board_cards for c in street],
                    "hole":  [repr(c) for c in state.hole_cards[state.actor_index]],
                    "stacks": list(state.stacks),
                    "bets":   list(state.bets),
                    "street": street_name(state.street_index),
                    "can_fold": state.can_fold(),
                    "can_check_or_call": state.can_check_or_call(),
                    "can_raise": bool(state.can_complete_bet_or_raise_to()),
                    "min_raise": state.min_completion_betting_or_raising_to_amount,
                    "max_raise": state.max_completion_betting_or_raising_to_amount,
                    "how_much_to_call": state.checking_or_calling_amount,
                    "action_history": [str(o) for o in get_visible_ops(state)[-6:]],
                }

                if global_player_index == 0:  # Hero
                    status.info(f"🤖 Claude deciding… Hand {hand_count}, {street_name(state.street_index)}")
                    result, usage, equity_calls = run_turn(
                        json.dumps(obs, default=str), api_key, ltm, active_player_ids)
                    if usage: ss.token_usage.append(("Claude", usage))
                    hand_equity_calls.extend(equity_calls)

                    act = result.get("action", "check")
                    amt = result.get("amount", 0)
                    rsn = result.get("reasoning", "")

                    if act == "fold" and state.can_fold(): state.fold()
                    elif act == "raise" and state.can_complete_bet_or_raise_to():
                        mn = state.min_completion_betting_or_raising_to_amount or 0
                        mx = state.max_completion_betting_or_raising_to_amount or mn
                        state.complete_bet_or_raise_to(int(max(mn, min(int(amt or mn), mx))))
                    elif state.can_check_or_call(): state.check_or_call()
                    elif state.can_fold(): state.fold()

                    ss.game_log.append({"hand":hand_count,"street":street_name(state.street_index),
                                        "player":"Claude","action":act,"amount":amt,"reasoning":rsn})
                    hand_reasonings.append([act, amt, rsn])
                else:
                    strat = opponent_strategies.get(global_player_index)
                    if strat:
                        _ = strat(state, state.actor_index)
                    else: state.check_or_call()
                    ss.game_log.append({"hand":hand_count,"street":street_name(state.street_index),
                                        "player":f"P{global_player_index+1}","action":"act","amount":0,"reasoning":""})

            # ── hand over ──────────────────────────────────────────
            payoffs_by_player = [0] * len(player_stacks)
            for seat, gi in enumerate(active):
                payoffs_by_player[gi] = int(state.payoffs[seat])

            ops = get_visible_ops(state)
            history = humanize_action_history(ops, active, players)
            med.ingest_hand(action_history=history, reasoning=hand_reasonings,
                            chip_changes=payoffs_by_player)

            status.info(f"⚖️ Judge analyzing hand {hand_count}…")
            judge_eval, jusage = run_judge(history, api_key)
            if jusage: ss.token_usage.append(("Judge", jusage))
            if judge_eval:
                med.log_trend(judge_eval)
                ss.judge_log.append({"hand":hand_count,"eval":judge_eval})

            run_post_hand_reflection(ltm=ltm, eval_text=judge_eval or "",
                active_player_ids=active_player_ids, reasonings=hand_reasonings,
                payoffs=tuple(payoffs_by_player))

            for g in range(len(player_stacks)):
                player_stacks[g] += payoffs_by_player[g]
            ss.player_stacks = list(player_stacks)
            render_table(num_players)

            ss.hand_summaries.append({
                "hand": hand_count,
                "payoffs": payoffs_by_player,
                "stacks": list(player_stacks),
                "claude_decisions": hand_reasonings,
                "judge": judge_eval,
                "equity_calls": hand_equity_calls,
            })
            ss.mem_digest = med.read_digest()
            ss.ltm_display = {
                "Hero": ltm.read_self_profile(),
                **{f"P{i+1}": ltm.lookup_player(f"P{i+1}") for i in range(1, num_players)},
            }

            button = (button + 1) % len(player_stacks)
            while player_stacks[button] <= 0:
                button = (button + 1) % len(player_stacks)
            ss.button = button
            time.sleep(0.2)

        # ── game over ──────────────────────────────────────────────
        med.close_game(final_stacks={p: player_stacks[i] for i,p in enumerate(players)})
        ltm.close_session()
        ss.phase = "done"
        ss.hand_count = hand_count
        prog.empty()
        status.success(f"✅ Game over — {hand_count} hands played!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  Results tabs
# ══════════════════════════════════════════════════════════════════
if ss.hand_count > 0:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    np_ = ss.num_players if ss.num_players else num_players
    player_names = ["Hero"] + [f"P{i+1}" for i in range(1, np_)]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Stacks", "🧠 Claude Decisions", "🔮 Equity Calls",
        "⚖️ Judge Reports", "📋 Action Log", "📈 Tokens",
    ])

    with tab1:
        import pandas as pd
        rows = []
        for h in ss.hand_summaries:
            row = {"Hand": h["hand"]}
            stacks = h.get("stacks", [])
            for j, pname in enumerate(player_names):
                row[pname] = stacks[j] if j < len(stacks) else 0
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows).set_index("Hand")
            st.line_chart(df, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    with tab2:
        for h in ss.hand_summaries:
            with st.expander(f"Hand {h['hand']} — payoffs {h.get('payoffs',[])}"):
                if h.get("claude_decisions"):
                    for act, amt, rsn in h["claude_decisions"]:
                        c1, c2 = st.columns([1,5])
                        with c1:
                            st.markdown(badge(act), unsafe_allow_html=True)
                            if amt: st.caption(f"${int(amt):,}")
                        with c2:
                            st.markdown(f'<div class="reasoning-box">{rsn}</div>', unsafe_allow_html=True)
                else:
                    st.caption("Claude folded or wasn't active.")

    with tab3:
        has_equity = any(h.get("equity_calls") for h in ss.hand_summaries)
        if has_equity:
            for h in ss.hand_summaries:
                calls = h.get("equity_calls",[])
                if calls:
                    with st.expander(f"Hand {h['hand']} — {len(calls)} equity call(s)"):
                        for msg in calls:
                            st.markdown(f'<div class="equity-box">🎯 {msg}</div>', unsafe_allow_html=True)
        else:
            st.caption("No equity tool calls yet.")

    with tab4:
        if ss.judge_log:
            for j in ss.judge_log:
                with st.expander(f"Hand {j['hand']} — Judge Evaluation"):
                    st.markdown(f'<div class="judge-box">{j["eval"]}</div>', unsafe_allow_html=True)
        else:
            st.caption("No judge evaluations yet.")

    with tab5:
        import pandas as pd
        if ss.game_log:
            df_log = pd.DataFrame(ss.game_log)
            st.dataframe(df_log[["hand","street","player","action","amount","reasoning"]],
                         use_container_width=True, height=400)
        else:
            st.caption("No actions logged.")

    with tab6:
        if ss.token_usage:
            total_in  = sum(u.input_tokens  for _,u in ss.token_usage)
            total_out = sum(u.output_tokens for _,u in ss.token_usage)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Calls",   len(ss.token_usage))
            c2.metric("Input Tokens",  f"{total_in:,}")
            c3.metric("Output Tokens", f"{total_out:,}")
            c4.metric("Est. Cost",     f"~${(total_in*0.00025+total_out*0.00125)/1000:.3f}")
            import pandas as pd
            rows2 = [{"#":i+1,"Agent":ag,"In":u.input_tokens,"Out":u.output_tokens}
                     for i,(ag,u) in enumerate(ss.token_usage)]
            st.dataframe(pd.DataFrame(rows2), use_container_width=True)
        else:
            st.caption("No API calls yet.")


if ss.phase == "idle":
    st.markdown("""
    <div style="text-align:center;padding:70px 20px 80px">
        <div style="font-size:3.8rem;margin-bottom:6px">
            <span class="suit-float" style="color:#1a5c30">♠</span>
            <span class="suit-float" style="color:#7a1a22">♥</span>
            <span class="suit-float" style="color:#1a5c30">♣</span>
            <span class="suit-float" style="color:#7a1a22">♦</span>
        </div>
        <div style="font-family:'Cinzel Decorative',serif;font-size:1.1rem;color:#6b5520;letter-spacing:.12em;margin-top:24px">
            The Table Awaits
        </div>
        <div style="font-family:Cinzel,serif;font-size:.68rem;color:#2a2010;letter-spacing:.22em;text-transform:uppercase;margin-top:12px">
            Configure in the sidebar &nbsp;·&nbsp; Press ▶ Run to begin
        </div>
        <div style="margin-top:18px;font-family:Cinzel,serif;font-size:.6rem;color:#1a1408;letter-spacing:.16em;text-transform:uppercase">
            v5 &nbsp;·&nbsp; LTM &nbsp;·&nbsp; Button Rotation &nbsp;·&nbsp; EV Bots &nbsp;·&nbsp; 2–7 Players
        </div>
    </div>
    """, unsafe_allow_html=True)
