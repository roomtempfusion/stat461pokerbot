from dotenv import load_dotenv
import json
from anthropic import Anthropic, beta_tool

load_dotenv()

client = Anthropic()
# suits: s, d, h, c
# hole_cards = 'AsKc'
p2_cards = '6sKs'
p3_cards = 'Qh4d'

flop_cards = 'QcKh6d'
turn_cards = '2h'
river_cards = '7s'

game_state = {
    "hole_cards": 'AsKc',
    "your_chips": 500,
    "p2_chips": 495,
    "p3_chips": 495,
    "pot": 10,
    "current_bet_to_call": 5,
    "flop_cards": None,
    "turn_card": None,
    "river_card": None,
    "preflop_actions": "P2 raises 5, P3 calls 5",
    "flop_actions": None,
    "turn_actions": None,
    "river_actions": None
}
gs_json = json.dumps(game_state, indent=2)
system_prompt = """You are a poker agent playing 3-handed Texas Hold'em No-Limit. 
Blinds are 1/2. You are Player 1 (Hero).
Think about pot odds, position, and hand strength before deciding.
Use the take_action tool to submit your decision."""

@beta_tool
def take_action(action: str, amount: float = 0) -> str:
    """Submit your poker action for this turn.

    Args:
        action: One of fold, check, call, raise
        amount: Raise amount in chips. Required if action is raise.
    """
    return json.dumps({"status": "accepted", "action": action, "amount": amount})

def run_turn(state_json=gs_json):
    messages = [{
        "role": "user",
        "content": f"""It's your turn to act. Here is the current game state:
    
    ```json
    {state_json}
    ```
    
    Decide your action."""}]

    runner = client.beta.messages.tool_runner(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=[take_action],
        system=system_prompt,
        messages=messages
    )

    agent_action = None

    for message in runner:
        tokens = f"Input: {message.usage.input_tokens}, Output: {message.usage.output_tokens}"

        for block in message.content:
            if block.type == "tool_use" and block.name == "take_action":
                agent_action = block.input

    return agent_action, tokens

# Pre-flop scenario
# print(run_turn())

# Flop scenario

game_state = {
    "hole_cards": 'AsKc',
    "your_chips": 485,
    "p2_chips": 485,
    "p3_chips": 485,
    "pot": 45,
    "current_bet_to_call": 30,
    "flop_cards": None,
    "turn_card": None,
    "river_card": None,
    "preflop_actions": "P2 raises 5, P3 calls 5, You raise 15, P2 calls 10, P3 calls 10",
    "flop_actions": "P2 raises 30, P3 calls 30",
    "turn_actions": None,
    "river_actions": None
}
gs_json = json.dumps(game_state, indent=2)

# print(run_turn(gs_json))

# Turn scenario

game_state = {
    "hole_cards": 'AsKc',
    "your_chips": 455,
    "p2_chips": 450,
    "p3_chips": 455,
    "pot": 140,
    "current_bet_to_call": 5,
    "flop_cards": None,
    "turn_card": None,
    "river_card": None,
    "preflop_actions": "P2 raises 5, P3 calls 5, You raise 15, P2 calls 10, P3 calls 10",
    "flop_actions": "P2 raises 30, P3 calls 30, You call 30",
    "turn_actions": "P2 raises 5, P3 folds",
    "river_actions": None
}
gs_json = json.dumps(game_state, indent=2)

print(run_turn(gs_json))



