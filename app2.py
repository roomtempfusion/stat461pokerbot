"""
app2.py — Claude Poker Agent: Full System
Integrates MediumTermMemory + Judge Agent + 3 opponent types
Run: streamlit run app2.py
"""

import os, random, json, time
from datetime import datetime
from pathlib import Path

import streamlit as st
from pokerkit import (
    NoLimitTexasHoldem, Automation,
    HoleDealing, CardBurning, ChipsPushing,
    HoleCardsShowingOrMucking,
)
from anthropic import Anthropic

# ══════════════════════════════════════════════════════════════════
#  MediumTermMemory  (from LLM_Integration.ipynb)
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
        streets = ["Preflop", "Flop", "Turn", "River"]
        for r, s in zip(reasoning, streets):
            self._append_to("raw_log.txt", f"{s}: {r}\n")
        stats = self._read_json("stats.json")
        stats["hands_played"] = self.hands_played
        player_names = list(stats.get("players", {}).keys())
        for name, change in zip(player_names, chip_changes):
            stats["players"][name]["net_chips"] = \
                stats["players"][name].get("net_chips", 0) + int(change)
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

    def read_raw(self):
        return self._read_text("raw_log.txt") if self._game_dir else ""

    def close_game(self, final_stacks, game_critique=""):
        stacks_str = " | ".join(f"{p}: {c}" for p, c in final_stacks.items())
        self._append_to("raw_log.txt",
            f"\n=== GAME {self.game_id} CLOSED | {datetime.now().isoformat()} ===\n"
            f"Final stacks: {stacks_str}\nHands: {self.hands_played}\n"
            f"Critique: {game_critique or 'pending'}\n{'='*60}\n")

    def purge_memory(self):
        if not self._game_dir:
            return ""
        raw = self._read_text("raw_log.txt")
        trends = self._read_text("trends.txt")
        for child in self._game_dir.iterdir():
            child.unlink()
        self._game_dir.rmdir()
        self._game_dir = None
        self.game_id = ""
        self.hands_played = 0
        return f"{raw}\n\n{'='*60}\n\n{trends}"

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
#  Page config & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Claude Poker Agent", page_icon="♠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
  --green:  #00c853;
  --gold:   #ffd600;
  --red:    #ff1744;
  --blue:   #2979ff;
  --purple: #aa00ff;
  --bg:     #0a0e1a;
  --surface:#111827;
  --border: rgba(255,255,255,.07);
}

html, body, [data-testid="stApp"] { background: var(--bg) !important; }
* { font-family: 'Rajdhani', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* felt texture on main area */
[data-testid="stMain"] {
  background: radial-gradient(ellipse at 50% 0%, #0d2b1a 0%, var(--bg) 70%) !important;
}

h1 { font-size: 2.4rem !important; font-weight: 700 !important;
     letter-spacing: .06em; color: var(--gold) !important;
     text-shadow: 0 0 40px rgba(255,214,0,.3); }
h2, h3 { color: #e2e8f0 !important; letter-spacing: .04em; }

/* cards — app.py style */
.card {
    display:inline-block; padding:6px 14px; margin:3px;
    border-radius:8px; font-weight:bold; font-size:1.2rem;
    background:#fff; color:#111; border:2px solid #ccc;
    box-shadow:2px 2px 6px rgba(0,0,0,.5);
}
.card-red  { color:#c0392b; }
.card-back { background:#1e3a8a; color:#fff; border-color:#3b82f6;
             font-size:1.5rem; }

/* player boxes */
.player-box {
  background:rgba(255,255,255,.06); border-radius:14px;
  padding:18px 14px; text-align:center; border:2px solid rgba(255,255,255,.13);
  transition: border-color .2s;
}
.player-box.agent { border-color:#10b981; background:rgba(16,185,129,.10); }
.player-label { font-size:1rem; font-weight:700; letter-spacing:.05em;
                color:#94a3b8; text-transform:uppercase; margin-bottom:10px; }
.player-label.agent-label { color:#10b981; }
.chip-count { font-size:1.4rem; font-weight:700; color:#f1f5f9;
              font-family:'JetBrains Mono'; margin-top:8px; }

/* pot */
.pot-box {
  background: linear-gradient(135deg, #0d2b1a, #0a1628);
  border: 1px solid var(--green); border-radius: 16px;
  padding: 20px; text-align: center;
}
.pot-label { font-size: .75rem; letter-spacing: .15em; color: var(--green);
             text-transform: uppercase; font-weight: 600; }
.pot-amount { font-size: 2.4rem; font-weight: 700; color: #fff;
              font-family: 'JetBrains Mono'; line-height: 1.1; }

/* street badge */
.street-pill {
  display: inline-block; padding: 4px 18px; border-radius: 20px;
  background: rgba(255,214,0,.1); border: 1px solid rgba(255,214,0,.3);
  color: var(--gold); font-size: .8rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
}

/* action badges */
.badge {
  display: inline-block; padding: 3px 12px; border-radius: 20px;
  font-size: .75rem; font-weight: 700; letter-spacing: .06em;
}
.badge-fold  { background: rgba(255,23,68,.15);  color: #ff6b9d; border: 1px solid rgba(255,23,68,.3); }
.badge-call  { background: rgba(41,121,255,.15); color: #82b1ff; border: 1px solid rgba(41,121,255,.3); }
.badge-check { background: rgba(0,200,83,.15);   color: #69f0ae; border: 1px solid rgba(0,200,83,.3); }
.badge-raise { background: rgba(255,214,0,.15);  color: #ffd600; border: 1px solid rgba(255,214,0,.3); }

/* memory / judge boxes */
.memory-box {
  background: var(--surface); border: 1px solid rgba(170,0,255,.25);
  border-radius: 12px; padding: 14px; font-size: .82rem;
  font-family: 'JetBrains Mono'; color: #c4b5fd; line-height: 1.6;
  white-space: pre-wrap; max-height: 260px; overflow-y: auto;
}
.judge-box {
  background: var(--surface); border-left: 3px solid var(--blue);
  border-radius: 0 12px 12px 0; padding: 12px 16px;
  font-style: italic; color: #93c5fd; font-size: .88rem; line-height: 1.6;
}
.reasoning-box {
  background: var(--surface); border-left: 3px solid var(--green);
  border-radius: 0 12px 12px 0; padding: 12px 16px;
  color: #86efac; font-size: .88rem; line-height: 1.6;
}

/* section divider */
.divider {
  border: none; border-top: 1px solid var(--border); margin: 24px 0;
}

/* tag */
.opponent-tag {
  display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: .7rem; font-weight: 700; margin-left: 6px; letter-spacing: .06em;
}
.tag-pocket { background: rgba(255,214,0,.15); color: var(--gold); }
.tag-punter { background: rgba(255,23,68,.15);  color: #ff6b9d; }
.tag-tight  { background: rgba(100,116,139,.2); color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════
SUITS_RED = {"h", "d"}
PLAYER_LABELS = ["Hero (Claude 🤖)", "P2 — Pocket Pair", "P3 — Punter", "P4 — Tight Passive"]
OPPONENT_TAGS = ["", "pocket", "punter", "tight"]

def to_short(c):
    """Convert any PokerKit card to short notation like Ah, Tc, 2d."""
    # Card objects have .rank and .suit attributes
    if hasattr(c, 'rank') and hasattr(c, 'suit'):
        return f"{c.rank}{c.suit}"
    # repr() gives short form for PokerKit cards: repr(card) -> "Ah"
    r = repr(c)
    if len(r) == 2:
        return r
    # Fallback: str may be "TEN OF SPADES (Ts)" - extract parenthesized part
    s = str(c)
    if "(" in s and ")" in s:
        return s[s.index("(")+1:s.index(")")]
    return s[:2] if len(s) >= 2 else s

def card_html(c):
    s = to_short(c)
    suit = s[-1].lower() if s else ""
    cls = "card-red" if suit in ("h","d") else ""
    return f'<span class="card {cls}">{s}</span>'

def cards_html(cards, hidden=False):
    if hidden or not cards:
        return '<span class="card card-back">🂠</span>'
    return "".join(card_html(c) for c in cards)

def street_name(idx):
    return ["Preflop","Flop","Turn","River"][idx] if idx is not None else "—"

def badge(action):
    a = action.lower()
    return f'<span class="badge badge-{a}">{a.upper()}</span>'

def get_visible_ops(state):
    try:
        winners = set()
        for op in state.operations:
            if isinstance(op, ChipsPushing):
                winners = {i for i, amt in enumerate(op.amounts) if amt > 0}
        showdown = sum(1 for p in state.statuses if p) > 1
        out = []
        for op in state.operations:
            if isinstance(op, (HoleDealing, CardBurning)):
                continue
            if isinstance(op, HoleCardsShowingOrMucking):
                if showdown or op.player_index in winners:
                    out.append(op)
            else:
                out.append(op)
        return out
    except:
        return [op for op in state.operations
                if not isinstance(op, (HoleDealing, CardBurning))]


# ══════════════════════════════════════════════════════════════════
#  LLM calls
# ══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a poker agent playing 3-handed (or 4-handed) Texas Hold'em No-Limit.
Blinds are 1000/2000. You are Player 0 (Hero). Other players are P2, P3, P4.
Think about pot odds, position, and hand strength. Use the memory digest when provided.
Use the take_action tool to submit your decision."""

JUDGE_PROMPT = """You are an evaluating agent for Texas Hold'em. Analyze the action history
of the previous hand and evaluate the non-Hero players only (index 1+). 
Be specific about mistakes, bluffs, and tendencies. Use the generate_observation tool."""

def run_turn(obs_json, api_key, memory_digest=""):
    client = Anthropic(api_key=api_key)
    content = f"It's your turn.\n\nMemory digest:\n{memory_digest}\n\nGame state:\n```json\n{obs_json}\n```\nDecide your action."
    tools = [{
        "name": "take_action",
        "description": "Submit your poker action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action":    {"type":"string","enum":["fold","check","call","raise"]},
                "amount":    {"type":"number","description":"Raise amount if action=raise"},
                "reasoning": {"type":"string","description":"Max 3 sentences. Rate opponent aggression 1-10 if past preflop."},
            },
            "required": ["action","reasoning"],
        }
    }]
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        tools=tools, system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":content}]
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "take_action":
            inp = block.input
            return {"action": inp.get("action","check"),
                    "amount": inp.get("amount",0),
                    "reasoning": inp.get("reasoning","")}, resp.usage
    return {"action":"check","amount":0,"reasoning":"Defaulted."}, resp.usage

def run_judge(action_history, api_key):
    client = Anthropic(api_key=api_key)
    tools = [{
        "name": "generate_observation",
        "description": "Evaluate non-Hero players from the previous hand.",
        "input_schema": {
            "type": "object",
            "properties": {
                "evaluation": {"type":"string","description":"Specific analysis of opponent play."}
            },
            "required": ["evaluation"],
        }
    }]
    content = f"Action history:\n```json\n{json.dumps([str(a) for a in action_history], default=str)}\n```\nGenerate observations."
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        tools=tools, system=JUDGE_PROMPT,
        messages=[{"role":"user","content":content}]
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "generate_observation":
            return block.input.get("evaluation",""), resp.usage
    return "", resp.usage


# ══════════════════════════════════════════════════════════════════
#  Opponent strategies
# ══════════════════════════════════════════════════════════════════
def pocket_pair_strategy(state, player_index):
    hole = state.hole_cards[player_index]
    is_pair = len(hole) >= 2 and hole[0].rank == hole[1].rank
    strong = {"A","K","Q","J","T","9","8","7"}
    if is_pair and hole[0].rank in strong and state.can_complete_bet_or_raise_to():
        mn = state.min_completion_betting_or_raising_to_amount
        mx = state.max_completion_betting_or_raising_to_amount
        state.complete_bet_or_raise_to(min(mn*2, mx))
    elif is_pair and state.can_check_or_call():
        state.check_or_call()
    elif state.can_fold():
        state.fold()
    else:
        state.check_or_call()

def make_punter_strategy(is_punting):
    def punter_strategy(state, player_index):
        if not is_punting:
            state.fold() if state.can_fold() else state.check_or_call()
            return
        if state.can_complete_bet_or_raise_to():
            mn = state.min_completion_betting_or_raising_to_amount
            mx = state.max_completion_betting_or_raising_to_amount
            state.complete_bet_or_raise_to(min(mn*2, mx))
        elif state.can_check_or_call():
            state.check_or_call()
        elif state.can_fold():
            state.fold()
    return punter_strategy

def tight_passive_strategy(state, player_index):
    if random.random() < 0.9 and state.can_fold():
        state.fold()
    elif state.can_check_or_call():
        state.check_or_call()
    elif state.can_fold():
        state.fold()


# ══════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════
def init():
    ss = st.session_state
    defaults = {
        "game_log": [], "reasonings": [], "judge_log": [],
        "player_stacks": [10000,10000,10000,10000],
        "hand_count": 0, "phase": "idle",
        "board_cards": [], "hole_cards": [[],[],[],[]],
        "pot": 0, "street_idx": 0,
        "token_usage": [], "mem_digest": "",
        "hand_summaries": [],
    }
    for k, v in defaults.items():
        if k not in ss: ss[k] = v

init()
ss = st.session_state


# ══════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ♠ Claude Poker")
    st.caption("Full System — Memory + Judge + Opponents")

    api_key = st.text_input("Anthropic API Key", type="password")

    st.markdown("---")
    st.markdown("**💰 Starting Stacks**")
    s0 = st.number_input("Hero (Claude)", 1000, 500000, 10000, 1000)
    s1 = st.number_input("P2 — Pocket Pair 🃏", 1000, 500000, 10000, 1000)
    s2 = st.number_input("P3 — Punter 🎲", 1000, 500000, 10000, 1000)
    s3 = st.number_input("P4 — Tight Passive 🧊", 1000, 500000, 10000, 1000)

    st.markdown("---")
    st.markdown("**⚙️ Options**")
    max_hands = st.slider("Max hands", 1, 30, 5)
    show_memory = st.checkbox("Show memory digest per hand", True)

    col1, col2 = st.columns(2)
    start = col1.button("▶ Run", use_container_width=True, type="primary")
    reset = col2.button("↺ Reset", use_container_width=True)

    if reset:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # live memory digest in sidebar
    if ss.mem_digest:
        st.markdown("---")
        st.markdown("**🧠 Memory Digest**")
        st.markdown(f'<div class="memory-box">{ss.mem_digest}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════
st.markdown('<h1>♠ Claude Poker Agent</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b;letter-spacing:.06em;font-size:.9rem">FULL SYSTEM — MEMORY · JUDGE · OPPONENT MODELS</p>', unsafe_allow_html=True)

# ── top row ──────────────────────────────────────────────────────
col_board, col_pot = st.columns([3,1])

with col_pot:
    st.markdown(f"""
    <div class="pot-box">
        <div class="pot-label">💰 Pot</div>
        <div class="pot-amount">{ss.pot:,}</div>
        <div style="margin-top:10px">
            <span class="street-pill">{street_name(ss.street_idx)}</span>
        </div>
        <div style="margin-top:8px;color:#475569;font-size:.82rem">Hand #{ss.hand_count}</div>
    </div>""", unsafe_allow_html=True)

with col_board:
    st.markdown("**Community Cards**")
    if ss.board_cards:
        st.markdown(cards_html(ss.board_cards), unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#334155;font-style:italic">No community cards yet</span>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── player boxes — app.py style ──────────────────────────────────
OPPONENT_TYPE_LABELS = ["", " 🃏 Pocket Pair", " 🎲 Punter", " 🧊 Tight Passive"]
pcols = st.columns(4)
for i, col in enumerate(pcols):
    with col:
        stack = ss.player_stacks[i] if i < len(ss.player_stacks) else 0
        hole  = ss.hole_cards[i] if i < len(ss.hole_cards) else []
        is_agent = (i == 0)
        cls = "agent" if is_agent else ""
        # Cards: show Claude's cards, hide opponents with card-back
        if hole:
            cards_disp = cards_html(hole)
        else:
            cards_disp = '<span class="card card-back">🂠</span><span class="card card-back">🂠</span>'
        label = PLAYER_LABELS[i] + OPPONENT_TYPE_LABELS[i]
        chip_icon = "💰"
        st.markdown(f"""
        <div class="player-box {cls}">
            <div style="font-weight:bold;font-size:1rem;margin-bottom:8px;color:{'#10b981' if is_agent else '#94a3b8'}">{label}</div>
            <div style="margin-bottom:10px">{cards_disp}</div>
            <div style="font-size:1.3rem">{chip_icon} {stack:,}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  Run simulation
# ══════════════════════════════════════════════════════════════════
if start:
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
    else:
        ss.player_stacks = [s0, s1, s2, s3]
        ss.hand_count = ss.pot = ss.street_idx = 0
        ss.board_cards = []
        ss.hole_cards = [[],[],[],[]]
        ss.game_log = []; ss.reasonings = []; ss.judge_log = []
        ss.token_usage = []; ss.hand_summaries = []
        ss.mem_digest = ""
        ss.phase = "running"

        med = MediumTermMemory()
        med.new_game("game_001", ["Hero","P2","P3","P4"])

        prog = st.progress(0, "Starting…")
        status = st.empty()

        while (len([s for s in ss.player_stacks if s > 0]) > 1
               and ss.hand_count < max_hands):

            active = [i for i, s in enumerate(ss.player_stacks) if s > 0]
            if len(active) < 2: break

            cur_stacks = tuple(ss.player_stacks[i] for i in active)
            ss.hand_count += 1
            prog.progress(ss.hand_count / max_hands, f"Hand {ss.hand_count}/{max_hands}")

            punting = random.random() < 0.1
            opponent_strats = {
                1: pocket_pair_strategy,
                2: make_punter_strategy(punting),
                3: tight_passive_strategy,
            }

            state = NoLimitTexasHoldem.create_state(
                (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
                 Automation.BLIND_OR_STRADDLE_POSTING, Automation.CARD_BURNING,
                 Automation.HOLE_DEALING, Automation.BOARD_DEALING,
                 Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                 Automation.HAND_KILLING, Automation.CHIPS_PUSHING,
                 Automation.CHIPS_PULLING),
                True, 0, (1000, 2000), 2000, cur_stacks, len(active)
            )

            hand_reasonings = []
            digest = med.read_digest() if ss.hand_count > 1 else ""
            ss.mem_digest = digest

            while state.status:
                if state.can_deal_hole():
                    state.deal_hole(); continue
                if state.can_deal_board():
                    state.deal_board(); continue
                if state.actor_index is None:
                    break

                ss.pot       = state.total_pot_amount
                ss.street_idx = state.street_index or 0
                ss.board_cards = list(state.board_cards)
                ss.hole_cards  = [list(h) for h in state.hole_cards] + \
                                  [[] for _ in range(4 - len(state.hole_cards))]

                obs = {
                    "your_index": 0,
                    "pot": state.total_pot_amount,
                    "board": [repr(c) for c in state.board_cards],
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

                if state.actor_index == 0:
                    status.info(f"🤖 Claude deciding… (Hand {ss.hand_count}, {street_name(state.street_index)})")
                    result, usage = run_turn(json.dumps(obs, default=str), api_key, digest)
                    ss.token_usage.append(("Claude", usage))
                    act = result["action"]; amt = result.get("amount",0)
                    rsn = result.get("reasoning","")

                    if act in ["check","call"]:
                        state.check_or_call()
                    elif act == "raise" and state.can_complete_bet_or_raise_to():
                        mn = state.min_completion_betting_or_raising_to_amount or 0
                        mx = state.max_completion_betting_or_raising_to_amount or mn
                        state.complete_bet_or_raise_to(int(max(mn, min(amt, mx))))
                    elif act == "fold":
                        state.fold()
                    else:
                        state.check_or_call()

                    ss.game_log.append({"hand": ss.hand_count,
                        "street": street_name(state.street_index),
                        "player":"Claude","action":act,"amount":amt,"reasoning":rsn})
                    hand_reasonings.append([act, amt, rsn])

                else:
                    gi = active[state.actor_index]
                    strat = opponent_strats.get(gi)
                    if strat:
                        strat(state, state.actor_index)
                    else:
                        state.check_or_call()
                    ss.game_log.append({"hand": ss.hand_count,
                        "street": street_name(state.street_index),
                        "player": f"P{gi+1}","action":"act","amount":0,"reasoning":""})

            # ── hand over ──────────────────────────────────────────
            full_changes = [0]*4
            for i, gi in enumerate(active):
                full_changes[gi] = int(state.payoffs[i])

            med.ingest_hand(
                action_history=[str(o) for o in get_visible_ops(state)],
                reasoning=hand_reasonings,
                chip_changes=full_changes,
            )

            # Judge agent
            status.info(f"⚖️ Judge agent analyzing hand {ss.hand_count}…")
            judge_eval, jusage = run_judge(get_visible_ops(state), api_key)
            ss.token_usage.append(("Judge", jusage))
            if judge_eval:
                med.log_trend(judge_eval)
                ss.judge_log.append({"hand": ss.hand_count, "eval": judge_eval})

            for i, gi in enumerate(active):
                ss.player_stacks[gi] += int(state.payoffs[i])

            ss.hand_summaries.append({
                "hand": ss.hand_count,
                "payoffs": list(state.payoffs),
                "stacks": list(ss.player_stacks),
                "claude_decisions": hand_reasonings,
                "judge": judge_eval,
            })
            ss.mem_digest = med.read_digest()

        med.close_game(
            final_stacks={"Hero": ss.player_stacks[0], "P2": ss.player_stacks[1],
                          "P3": ss.player_stacks[2], "P4": ss.player_stacks[3]})

        ss.phase = "done"
        prog.empty()
        status.success(f"✅ Game over — {ss.hand_count} hands played!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  Results
# ══════════════════════════════════════════════════════════════════
if ss.hand_count > 0:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Stacks", "🧠 Claude Decisions", "⚖️ Judge Reports", "📋 Action Log", "📈 Tokens"])

    with tab1:
        import pandas as pd
        def safe_stack(h, idx):
            stacks = h.get("stacks", [])
            return stacks[idx] if idx < len(stacks) else 0
        rows = [{"Hand": h["hand"],
                 "Claude":      safe_stack(h, 0),
                 "P2 — Pocket": safe_stack(h, 1),
                 "P3 — Punter": safe_stack(h, 2),
                 "P4 — Tight":  safe_stack(h, 3),
                 } for h in ss.hand_summaries if "hand" in h]
        if rows:
            df = pd.DataFrame(rows).set_index("Hand")
            st.line_chart(df, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    with tab2:
        for h in ss.hand_summaries:
            payoffs_str = str(h["payoffs"])
            with st.expander(f"Hand {h['hand']} — payoffs {payoffs_str}"):
                if h["claude_decisions"]:
                    for act, amt, rsn in h["claude_decisions"]:
                        c1, c2 = st.columns([1,5])
                        with c1:
                            st.markdown(badge(act), unsafe_allow_html=True)
                            if amt: st.caption(f"${int(amt):,}")
                        with c2:
                            st.markdown(f'<div class="reasoning-box">{rsn}</div>',
                                        unsafe_allow_html=True)
                else:
                    st.caption("Claude folded or wasn't active.")

    with tab3:
        if ss.judge_log:
            for j in ss.judge_log:
                with st.expander(f"Hand {j['hand']} — Judge Evaluation"):
                    st.markdown(f'<div class="judge-box">{j["eval"]}</div>',
                                unsafe_allow_html=True)
        else:
            st.caption("No judge evaluations yet.")

    with tab4:
        import pandas as pd
        if ss.game_log:
            df_log = pd.DataFrame(ss.game_log)
            st.dataframe(df_log[["hand","street","player","action","amount","reasoning"]],
                         use_container_width=True, height=400)
        else:
            st.caption("No actions logged.")

    with tab5:
        if ss.token_usage:
            total_in  = sum(u.input_tokens  for _, u in ss.token_usage)
            total_out = sum(u.output_tokens for _, u in ss.token_usage)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Calls", len(ss.token_usage))
            c2.metric("Input Tokens",  f"{total_in:,}")
            c3.metric("Output Tokens", f"{total_out:,}")
            c4.metric("Est. Cost", f"~${(total_in*0.00025 + total_out*0.00125)/1000:.3f}")

            import pandas as pd
            rows2 = [{"#":i+1,"Agent":ag,
                      "In":u.input_tokens,"Out":u.output_tokens}
                     for i,(ag,u) in enumerate(ss.token_usage)]
            st.dataframe(pd.DataFrame(rows2), use_container_width=True)
        else:
            st.caption("No API calls yet.")

if ss.phase == "idle":
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#334155">
        <div style="font-size:3rem">♠ ♣ ♥ ♦</div>
        <div style="margin-top:16px;font-size:1.1rem;letter-spacing:.06em">
            Configure in the sidebar and press <strong style="color:#ffd600">▶ Run</strong>
        </div>
        <div style="margin-top:8px;font-size:.85rem;color:#1e293b">
            Memory · Judge Agent · 3 Opponent Types
        </div>
    </div>
    """, unsafe_allow_html=True)
