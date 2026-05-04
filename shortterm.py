import os
from datetime import datetime
from pathlib import Path

class ShortTermMemory:
    """
    Append-only text buffer scoped to a single hand.
    Use order:
        new_hand() called by the game engine at the start of each hand
        append() called by the agent after every turn to log reasoning + action
        read() called by the agent at the start of each turn (full context)
        close_hand() called at end of hand to write outcome/critique before flush
        purge_memory() wipes the buffer; returns final content for medium term ingestion
    """

    STREET_HEADERS = ["PREFLOP", "FLOP", "TURN", "RIVER"]

    def __init__(self, buffer_dir: str = "./memory/short_term"):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_path: Path | None = None
        self.hand_number: int = 0

    def new_hand(self, hand_number: int, seat_positions: dict):
        """
        seat_positions:dictionary w/ format {"Hero": "BTN", "P2": "SB", "P3": "BB"}
        creates the header of the log file with the time, hand number and seat positions.
        """
        self.hand_number = hand_number
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"hand_{hand_number:04d}.txt"
        self._buffer_path = self.buffer_dir / filename
        seats = ", ".join(f"{p}={pos}" for p, pos in seat_positions.items())
        header = (f"=== HAND #{hand_number} | {timestamp} ===\n"f"SEAT POSITIONS: {seats}\n\n")
        self._write(header, mode="w")

    def log_new_deal_information(self, street: str, game_state: dict):
        """
        Logs new information revealed by dealer. (at flop, turn or river).
        Must be called by the game engine before the agent reasons.
        street: either "PREFLOP", "FLOP", "TURN", or "RIVER"
        """
        assert street in self.STREET_HEADERS, f"Unknown street: {street}"

        #board will contain the concat of all revealed cards so far.
        board = game_state.get("flop_cards", "") or ""
        if game_state.get("turn_card"):
            board += f" {game_state['turn_card']}"
        if game_state.get("river_card"):
            board += f" {game_state['river_card']}"

        state_line = ("[{street}]\n"
                    f"Cards: {game_state['hole_cards']} | "f"Board: {board.strip() or 'n/a'} | " f"Pot: {game_state['pot']} | " "Stack: {game_state['your_chips']}\n")
        action_key = f"{street.lower()}_actions"

        #example "turn_actions":"P2 checks, Hero bets 60, P2 raises all-in to 432, Hero calls",
        if game_state.get(action_key):
            state_line += f"{game_state[action_key]}\n"
        self._write(state_line)

    def append_reasoning(self, reasoning: str, action: str, amount: float=0):
        """
        Log CoT. Called after agent decision.
        """
        action_str = f"{action} {amount}" if amount else action
        entry = (f">>> REASONING: {reasoning.strip()}\n"
                 f">>> ACTION: {action_str}\n\n")
        self._write(entry)

    def close_hand(self, outcome: str, showdown_cards: dict | None = None, self_critique: str = ""):
        """
        Records showdown outcome.
        outcome: string of objective outcome fact. Ex. "Won 180 chips" or "Lost 30 chips (folded turn)"
        showdown_cards: e.g. {"P2": "6sKs", "P3": "Qh4d"} OR None if no showdown happens
        self_critique: agent's reflection on the hand
        """
        showdown_str = "N/A"
        if showdown_cards:
            showdown_str = ", ".join(f"{p}: {c}" for p, c in showdown_cards.items())

        footer = (
            f"[END OF HAND]\n"
            f"Outcome: {outcome}\n"
            f"Showdown: {showdown_str}\n"
            f"Self-critique: {self_critique.strip() or 'pending'}\n"
            f"{'=' * 50}\n\n"
        )
        self._write(footer)

    def read(self):
        "Return full buffer contents for injection into the agent's context."
        if not self._buffer_path or not self._buffer_path.exists():
            return "[Short term memory is empty — new hand]"
        return self._buffer_path.read_text(encoding="utf-8")

    def purge_memory(self):
        """
        Wipe the buffer. Returns the full hand record for medium term ingestion.
        Called after close_hand().
        """
        content = self.read()
        if self._buffer_path and self._buffer_path.exists():
            os.remove(self._buffer_path)
        self._buffer_path = None
        return content

    def _write(self, text: str, mode: str = "a"): #internal write method.
        assert self._buffer_path is not None, "Call new_hand() before writing."
        with open(self._buffer_path, mode, encoding="utf-8") as f:
            f.write(text)