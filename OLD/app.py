"""
Streamlit Poker Visualization App
Based on llm_integration_v2.ipynb - 3-handed No-Limit Texas Hold'em
with Claude as AI Agent (Player 0)
"""

import random
import json
import streamlit as st
from pokerkit import (
    NoLimitTexasHoldem, Automation,
    HoleDealing, CardBurning,
)
from anthropic import Anthropic

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claude Poker Agent",
    page_icon="🃏",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #1a1a2e; }
.main { background-color: #1a1a2e; color: #eee; }

.card {
    display:inline-block; padding:6px 12px; margin:3px;
    border-radius:8px; font-weight:bold; font-size:1.1rem;
    background:#fff; color:#111; border:2px solid #ccc;
    box-shadow:2px 2px 4px rgba(0,0,0,.4);
}
.card-red  { color:#c0392b; }
.card-back { background:#1e3a8a; color:#fff; border-color:#3b82f6; }

.player-box {
    background:rgba(255,255,255,.08); border-radius:14px;
    padding:16px; text-align:center; border:2px solid rgba(255,255,255,.15);
}
.player-box.active { border-color:#f59e0b; background:rgba(245,158,11,.15); }
.player-box.folded { opacity:.45; }
.player-box.agent  { border-color:#10b981; background:rgba(16,185,129,.12); }

.chip { font-size:1.4rem; }
.pot-display {
    background:rgba(16,185,129,.2); border-radius:12px;
    padding:14px; text-align:center; border:2px solid #10b981;
    font-size:1.4rem; font-weight:bold; color:#10b981;
}
.action-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:.8rem; font-weight:bold; margin:2px;
}
.action-fold   { background:#7f1d1d; color:#fca5a5; }
.action-call   { background:#1e3a8a; color:#93c5fd; }
.action-check  { background:#14532d; color:#86efac; }
.action-raise  { background:#78350f; color:#fcd34d; }

.reasoning-box {
    background:rgba(139,92,246,.1); border-left:3px solid #8b5cf6;
    padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0;
    font-style:italic; color:#c4b5fd;
}
.street-badge {
    background:#374151; color:#f9fafb; padding:4px 14px;
    border-radius:20px; font-weight:bold; text-transform:uppercase;
    letter-spacing:.05em;
}
.log-entry { font-size:.82rem; color:#9ca3af; border-bottom:1px solid #1f2937; padding:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
SUITS_RED = {"h", "d"}

def card_html(card_str: str) -> str:
    s = str(card_str)
    suit = s[-1] if s else ""
    cls = "card-red" if suit in SUITS_RED else ""
    return f'<span class="card {cls}">{s}</span>'

def cards_html(cards) -> str:
    if not cards:
        return '<span class="card card-back">🂠</span>'
    return "".join(card_html(c) for c in cards)

def street_name(idx) -> str:
    return ["Preflop", "Flop", "Turn", "River"][idx] if idx is not None else "—"

def get_visible_operations(state):
    return [op for op in state.operations
            if not isinstance(op, (HoleDealing, CardBurning))]

def action_badge(action: str) -> str:
    cls = f"action-{action.lower()}"
    return f'<span class="action-badge {cls}">{action.upper()}</span>'

# ── Claude LLM integration ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a poker agent playing 3-handed Texas Hold'em No-Limit.
Blinds are 1000/2000. You are Player 0 (Hero).
Think about pot odds, position, and hand strength before deciding.
Use the take_action tool to submit your decision."""

def run_turn(state_json: str, api_key: str):
    """Call Claude claude-haiku-4-5-20251001 to decide a poker action."""
    client = Anthropic(api_key=api_key)

    tools = [{
        "name": "take_action",
        "description": "Submit your poker action for this turn.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action":    {"type": "string",  "enum": ["fold","check","call","raise"],
                              "description": "Poker action to take"},
                "amount":    {"type": "number",  "description": "Raise amount (required if action=raise)"},
                "reasoning": {"type": "string",  "description": "Max 3 sentence explanation"},
            },
            "required": ["action", "reasoning"],
        },
    }]

    messages = [{"role": "user", "content":
        f"It's your turn. Game state:\n```json\n{state_json}\n```\nDecide your action."}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        tools=tools,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "take_action":
            inp = block.input
            return {
                "action":    inp.get("action", "check"),
                "amount":    inp.get("amount", 0),
                "reasoning": inp.get("reasoning", ""),
            }, response.usage

    return {"action": "check", "amount": 0, "reasoning": "Defaulted to check."}, response.usage

# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    if "game_log" not in ss:          ss.game_log = []
    if "reasonings" not in ss:        ss.reasonings = []
    if "player_stacks" not in ss:     ss.player_stacks = [10_000, 10_000, 10_000]
    if "hand_count" not in ss:        ss.hand_count = 0
    if "phase" not in ss:             ss.phase = "idle"   # idle | running | done
    if "current_obs" not in ss:       ss.current_obs = None
    if "board_cards" not in ss:       ss.board_cards = []
    if "hole_cards" not in ss:        ss.hole_cards = [[], [], []]
    if "pot" not in ss:               ss.pot = 0
    if "street_idx" not in ss:        ss.street_idx = 0
    if "last_action" not in ss:       ss.last_action = None
    if "token_usage" not in ss:       ss.token_usage = []

init_state()
ss = st.session_state

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🃏 Claude Poker Agent")
    st.caption("3-handed No-Limit Texas Hold'em")

    api_key = st.text_input("Anthropic API Key", type="password",
                             help="Required to let Claude play")

    st.markdown("---")
    st.subheader("💰 Starting Stacks")
    s0 = st.number_input("Player 0 (Claude 🤖)", 1000, 500_000, 10_000, 1000)
    s1 = st.number_input("Player 1 (Random)", 1000, 500_000, 10_000, 1000)
    s2 = st.number_input("Player 2 (Random)", 1000, 500_000, 10_000, 1000)

    st.markdown("---")
    st.subheader("⚙️ Game Options")
    bias_agent = st.checkbox("Bias agent cards (Ac As)", value=False)
    max_hands   = st.number_input("Max hands to simulate", 1, 50, 5)

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Run Game", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("↺ Reset",    use_container_width=True)

    if reset_btn:
        for k in ["game_log","reasonings","player_stacks","hand_count","phase",
                  "current_obs","board_cards","hole_cards","pot","street_idx",
                  "last_action","token_usage"]:
            if k in ss: del ss[k]
        st.rerun()

# ── Main content area ──────────────────────────────────────────────────────────
st.title("🃏 Claude Poker Agent — Live Visualization")

top_left, top_right = st.columns([3, 1])

with top_right:
    st.markdown(f"""
    <div class="pot-display">
        🏆 Pot<br>
        <span style="font-size:2rem">{ss.pot:,}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;margin-top:8px"><span class="street-badge">{street_name(ss.street_idx)}</span></div>', unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:#6b7280;margin-top:6px'>Hand #{ss.hand_count}</div>", unsafe_allow_html=True)

# ── Board cards ────────────────────────────────────────────────────────────────
with top_left:
    st.subheader("Community Cards")
    board_display = cards_html(ss.board_cards) if ss.board_cards else \
        '<span style="color:#6b7280;font-style:italic">No community cards yet</span>'
    st.markdown(board_display, unsafe_allow_html=True)

st.markdown("---")

# ── Player boxes ───────────────────────────────────────────────────────────────
player_cols = st.columns(3)
player_labels = ["🤖 Claude (Agent)", "👤 Player 1", "👤 Player 2"]

for i, col in enumerate(player_cols):
    with col:
        stack = ss.player_stacks[i] if i < len(ss.player_stacks) else 0
        hole  = ss.hole_cards[i] if i < len(ss.hole_cards) else []
        cls   = "agent" if i == 0 else ""

        cards_disp = cards_html(hole) if (i == 0 and hole) else \
                     (cards_html(hole) if hole else '<span class="card card-back">🂠</span><span class="card card-back">🂠</span>')

        st.markdown(f"""
        <div class="player-box {cls}">
            <div style="font-weight:bold;font-size:1.1rem;margin-bottom:6px">{player_labels[i]}</div>
            <div style="margin-bottom:8px">{cards_disp}</div>
            <div class="chip">💰 {stack:,}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Run simulation ─────────────────────────────────────────────────────────────
if start_btn:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    else:
        ss.player_stacks = [s0, s1, s2]
        ss.hand_count    = 0
        ss.game_log      = []
        ss.reasonings    = []
        ss.token_usage   = []
        ss.phase         = "running"

        progress_bar = st.progress(0, text="Starting game…")
        status_area  = st.empty()

        while (len([s for s in ss.player_stacks if s > 0]) > 1
               and ss.hand_count < max_hands):

            active_indices = [i for i, s in enumerate(ss.player_stacks) if s > 0]
            if len(active_indices) < 2:
                break

            current_stacks = tuple(ss.player_stacks[i] for i in active_indices)
            ss.hand_count += 1
            hand_reasonings = []
            progress_bar.progress(ss.hand_count / max_hands,
                                  text=f"Hand {ss.hand_count}/{max_hands}")

            automations = [
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ]

            if not bias_agent:
                automations.insert(4, Automation.HOLE_DEALING)
                automations.insert(5, Automation.BOARD_DEALING)

            state = NoLimitTexasHoldem.create_state(
                tuple(automations),
                True, 0, (1000, 2000), 2000,
                current_stacks, len(active_indices),
            )

            agent_card_idx = 0
            agent_cards    = ["Ac", "As"]

            while state.status:
                # Dealing
                if bias_agent and state.can_deal_hole():
                    if state.hole_dealee_index == 0:
                        state.deal_hole(agent_cards[agent_card_idx])
                        agent_card_idx += 1
                    else:
                        state.deal_hole()
                    continue
                if bias_agent and state.can_deal_board():
                    state.deal_board()
                    continue

                if state.actor_index is None:
                    break

                # Update live display
                ss.pot         = state.total_pot_amount
                ss.street_idx  = state.street_index or 0
                ss.board_cards = list(state.board_cards)
                ss.hole_cards  = [list(h) for h in state.hole_cards]

                res = get_visible_operations(state)
                obs = {
                    "your_index":        active_indices.index(0) if 0 in active_indices else 0,
                    "pot":               state.total_pot_amount,
                    "board":             [str(c) for c in state.board_cards],
                    "hole":              [str(c) for c in state.hole_cards[state.actor_index]],
                    "stacks":            list(state.stacks),
                    "bets":              list(state.bets),
                    "street":            street_name(state.street_index),
                    "can_fold":          state.can_fold(),
                    "can_check_or_call": state.can_check_or_call(),
                    "can_raise":         bool(state.can_complete_bet_or_raise_to()),
                    "min_raise":         state.min_completion_betting_or_raising_to_amount,
                    "max_raise":         state.max_completion_betting_or_raising_to_amount,
                    "how_much_to_call":  state.checking_or_calling_amount,
                    "action_history":    [str(o) for o in res[-6:]],
                }

                if state.actor_index == 0:  # Claude agent
                    status_area.info(f"🤖 Claude is thinking… (Hand {ss.hand_count}, {street_name(state.street_index)})")
                    result, usage = run_turn(json.dumps(obs, default=str), api_key)
                    ss.token_usage.append(usage)

                    act = result["action"]
                    amt = result.get("amount", 0)
                    rsn = result.get("reasoning", "")

                    if act in ["check", "call"]:
                        state.check_or_call()
                    elif act == "raise" and state.can_complete_bet_or_raise_to():
                        target = max(
                            state.min_completion_betting_or_raising_to_amount or 0,
                            min(amt, state.max_completion_betting_or_raising_to_amount or amt)
                        )
                        state.complete_bet_or_raise_to(int(target))
                    elif act == "fold":
                        state.fold()
                    else:
                        state.check_or_call()

                    log_entry = {
                        "hand": ss.hand_count,
                        "street": street_name(state.street_index),
                        "player": "Claude",
                        "action": act,
                        "amount": amt,
                        "reasoning": rsn,
                    }
                    ss.game_log.append(log_entry)
                    hand_reasonings.append([act, amt, rsn])
                    ss.last_action = log_entry

                else:  # Random opponents
                    rnd = random.random()
                    if rnd < 0.1 and state.can_complete_bet_or_raise_to():
                        min_r = state.min_completion_betting_or_raising_to_amount or 0
                        max_r = state.max_completion_betting_or_raising_to_amount or min_r
                        state.complete_bet_or_raise_to(min(min_r * 2, max_r))
                        player_act = "raise"
                    elif rnd < 0.2 and state.can_fold():
                        state.fold()
                        player_act = "fold"
                    else:
                        state.check_or_call()
                        player_act = "check/call"

                    ss.game_log.append({
                        "hand": ss.hand_count, "street": street_name(state.street_index),
                        "player": f"P{state.actor_index}", "action": player_act,
                        "amount": 0, "reasoning": "",
                    })

            # Hand complete — update stacks
            for i, gi in enumerate(active_indices):
                ss.player_stacks[gi] += int(state.payoffs[i])

            ss.reasonings.append({
                "hand": ss.hand_count,
                "payoffs": list(state.payoffs),
                "stacks_after": list(ss.player_stacks),
                "claude_decisions": hand_reasonings,
            })

        ss.phase = "done"
        progress_bar.empty()
        status_area.success(f"✅ Simulation complete — {ss.hand_count} hands played!")
        st.rerun()

# ── Results display ────────────────────────────────────────────────────────────
if ss.hand_count > 0:
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Stack History", "🧠 Claude Decisions", "📋 Full Action Log", "📈 Token Usage"])

    with tab1:
        st.subheader("Chip Stack History by Hand")
        import pandas as pd

        rows = []
        for hand_data in ss.reasonings:
            rows.append({
                "Hand":     hand_data["hand"],
                "Claude":   hand_data["stacks_after"][0],
                "Player 1": hand_data["stacks_after"][1] if len(hand_data["stacks_after"]) > 1 else 0,
                "Player 2": hand_data["stacks_after"][2] if len(hand_data["stacks_after"]) > 2 else 0,
            })

        if rows:
            df = pd.DataFrame(rows).set_index("Hand")
            st.line_chart(df, use_container_width=True)
            st.dataframe(df.style.highlight_max(axis=1, color="#10b981"), use_container_width=True)

    with tab2:
        st.subheader("Claude's Reasoning by Hand")
        for hand_data in ss.reasonings:
            with st.expander(f"Hand {hand_data['hand']} — Payoffs: {hand_data['payoffs']}"):
                if hand_data["claude_decisions"]:
                    for act, amt, rsn in hand_data["claude_decisions"]:
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            st.markdown(action_badge(act), unsafe_allow_html=True)
                            if amt: st.caption(f"${amt:,}")
                        with col_b:
                            st.markdown(f'<div class="reasoning-box">{rsn}</div>',
                                        unsafe_allow_html=True)
                else:
                    st.caption("Claude didn't act this hand (folded by others or not in hand).")

    with tab3:
        st.subheader("Complete Action Log")
        if ss.game_log:
            import pandas as pd
            df_log = pd.DataFrame(ss.game_log)
            df_log["action"] = df_log["action"].str.upper()
            st.dataframe(df_log, use_container_width=True, height=400)
        else:
            st.caption("No actions logged yet.")

    with tab4:
        st.subheader("Token Usage (Claude calls)")
        if ss.token_usage:
            total_in  = sum(u.input_tokens  for u in ss.token_usage)
            total_out = sum(u.output_tokens for u in ss.token_usage)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total API Calls", len(ss.token_usage))
            c2.metric("Total Input Tokens",  f"{total_in:,}")
            c3.metric("Total Output Tokens", f"{total_out:,}")

            rows = [{"Call #": i+1,
                     "Input Tokens":  u.input_tokens,
                     "Output Tokens": u.output_tokens}
                    for i, u in enumerate(ss.token_usage)]
            import pandas as pd
            st.bar_chart(pd.DataFrame(rows).set_index("Call #"))
        else:
            st.caption("No API calls made yet.")

# ── Last action banner ─────────────────────────────────────────────────────────
if ss.last_action and ss.phase == "done":
    la = ss.last_action
    st.sidebar.markdown("---")
    st.sidebar.subheader("Last Claude Action")
    st.sidebar.markdown(action_badge(la["action"]), unsafe_allow_html=True)
    if la.get("amount"):
        st.sidebar.caption(f"Amount: ${la['amount']:,}")
    if la.get("reasoning"):
        st.sidebar.markdown(f'<div class="reasoning-box">{la["reasoning"]}</div>',
                            unsafe_allow_html=True)

if ss.phase == "idle":
    st.info("👈 Configure settings in the sidebar and press **▶ Run Game** to start the simulation.")
