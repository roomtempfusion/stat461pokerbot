import os
import json
from datetime import datetime
from pathlib import Path


class MediumTermMemory:
    """
    Append-only table-scoped buffer that accumulates hand records and trend
    observations across an entire game (one session at a single table).
    Order of calls:
        new_game() called by the game engine when a new table session begins
        ingest_hand() called after each hand with the raw record from ShortTermMemory.purge_memory(); appends raw log and bumps stats
        log_trend() called by the agent after reflecting on a hand to record a human-readable trend observation (e.g. "P2 has 3-bet preflop
                    in 4 of 6 hands — likely a wide 3-bet range")
        read_digest() called by the agent at the start of each turn; returns a structured summary (stats + recent trends) rather than the
                    full raw log, keeping context window usage bounded
        read_raw() returns the full unabridged game log (for end-of-game reflection)
        close_game() seals the buffer with final chip counts and a game-level critique
        purge_memory() wipes the buffer; returns the full game record for long-term ingestion

    File layout under buffer_dir/:
        game_<id>/
            raw_log.txt every hand record, appended verbatim
            trends.txt agent trend observations, one per hand
            stats.json running numeric stats (hands played, vpip counts, etc.)
            metadata.json game-level metadata (start time, players, game_id)
    """

    def __init__(self, buffer_dir: str = "./memory/medium_term"):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self._game_dir: Path | None = None
        self.game_id: str = ""
        self.hands_played: int = 0


    def new_game(self, game_id: str, players: list[str]):
        """
        Begin a new game session.
        Args:
        game_id:  unique identifier for this table session (e.g. "game_001")
        players:  list of player labels at the table, e.g. ["Hero", "P2", "P3"]
        """
        self.game_id = game_id
        self.hands_played = 0
        self._game_dir = self.buffer_dir / game_id
        self._game_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "game_id": game_id,
            "started_at": datetime.now().isoformat(),
            "players": players,
        }
        self._write_json("metadata.json", metadata)

        #for each player:
        stats = {
            "hands_played": 0,
            "players": {p: self._empty_player_stats() for p in players},
        }
        self._write_json("stats.json", stats)

        # log games
        self._append_to("raw_log.txt",
            f"=== GAME {game_id} STARTED | {metadata['started_at']} ===\n"
            f"Players: {', '.join(players)}\n\n"
        )
        self._append_to("trends.txt",
            f"=== TREND LOG | GAME {game_id} ===\n\n"
        )


    # def ingest_hand(self, hand_record: str, hand_stats: dict | None = None):
    #     """
    #     Append a completed hand record (from ShortTermMemory.purge_memory()) to the
    #     raw log and update running statistics.

    #     Args:
    #     hand_stats (optional): structured numeric observations extracted from the hand,
    #     keyed by player. Expected shape (all fields optional):
    #         {
    #             "P2": {"vpip": True, "preflop_raise": True, "went_to_showdown": False},
    #             "P3": {"vpip": False, ...},
    #         }
    #     Stats that are missing from hand_stats are left unchanged.
    #     """
    #     self.hands_played += 1
    #     self._append_to("raw_log.txt", hand_record)

    #     if hand_stats:
    #         stats = self._read_json("stats.json")
    #         stats["hands_played"] = self.hands_played
    #         for player, observations in hand_stats.items():
    #             if player not in stats["players"]:
    #                 stats["players"][player] = self._empty_player_stats()
    #             bucket = stats["players"][player]
    #             for key in ("vpip", "preflop_raise", "preflop_3bet",
    #                         "went_to_showdown", "won_hand", "continuation_bet"):
    #                 if observations.get(key):
    #                     bucket[key] = bucket.get(key, 0) + 1
    #             #tracks total chip delta.
    #             bucket["net_chips"] = (bucket.get("net_chips", 0) + observations.get("net_chips", 0))
    #         self._write_json("stats.json", stats)
    #     else:
    #         stats = self._read_json("stats.json")
    #         stats["hands_played"] = self.hands_played
    #         self._write_json("stats.json", stats)

    def ingest_hand_from_json_and_reasoning(self, action_history: list[str], reasoning: list[list], chip_changes: list[int])->None:
        """
        Alternative method to ingest hand to deprecate shortterm memory. Instead,
        read from the json and the reasoning list.        
        """
        self.hands_played += 1
        #action history is a list of string
        for item in action_history:
            self._append_to("rawlog.txt", f'{item}\n')

        self._append_to("rawlog.txt", '\nReasonings:\n')
        streets = ["Preflop", "Flop", "Turn", "River"]
        for reasoning_on_street, streetname in zip(reasoning, streets):
            self._append_to("rawlog.txt", f'{streetname}: {reasoning_on_street}')
        stats = self._read_json("stats.json")
        stats["hands_played"] = self.hands_played
        stats["chip_cahnges"] = chip_changes



    def log_trend(self, observation: str, hand_number: int | None = None):
        """
        Record a free-prose trend observation after hand-level reflection.

        Args:
        observation: the agent's natural-language finding, e.g.:
            "P2 has continuation-bet the flop in 3/4 opportunities — treat flop
             c-bets from P2 as weak until proven otherwise."
        hand_number: if provided, prefixes the entry for traceability.
        """
        hand_ref = f"[After hand #{hand_number}]" if hand_number is not None else f"[After hand #{self.hands_played}]"
        entry = f"{hand_ref}\n{observation.strip()}\n\n"
        self._append_to("trends.txt", entry)


    def read_digest(self, recent_trends: int = 5):
        """
        Return a bounded summary suitable for injection into the agent's context
        at the start of each turn.

        Includes:
          - Running numeric stats (frequencies derived from stats.json)
          - The N most recent trend observations
        unlike read_raw(), this is a summary

        Args:
        recent_trends: how many of the latest trend entries to include (default 5). 
        """
        if not self._game_dir:
            return "[Medium term memory is empty — no active game]"

        stats = self._read_json("stats.json")
        n = max(stats.get("hands_played", 0), 1) #div/0 error sentinel

        lines: list[str] = [f"=== LIVE TABLE DIGEST | Game {self.game_id} | {n} hands played ===\n"]

        for player, bucket in stats.get("players", {}).items():
            vpip_pct = 100 * bucket.get("vpip", 0)/n
            pfr_pct = 100 * bucket.get("preflop_raise", 0)/n
            pf3b_pct = 100 * bucket.get("preflop_3bet", 0)/n
            cbet_pct = 100 * bucket.get("continuation_bet", 0)/n
            wtsd_pct = 100 * bucket.get("went_to_showdown", 0)/n
            net = bucket.get("net_chips", 0)
            lines.append(
                f"  {player}: VPIP={vpip_pct:.0f}% | PFR={pfr_pct:.0f}% | "
                f"3B={pf3b_pct:.0f}% | CBet={cbet_pct:.0f}% | "
                f"WTSD={wtsd_pct:.0f}% | Net={net:+d} chips"
            )
        lines.append("")

        trends_text = self._read_text("trends.txt")
        #sptit on \n\n means we split on blank single lines.
        entries = [e.strip() for e in trends_text.split("\n\n") if e.strip() and not e.strip().startswith("=== TREND LOG")]
        if entries:
            lines.append(f"--- Recent Observations (last {recent_trends}) ---")
            for entry in entries[-recent_trends:]:
                lines.append(entry)
        else: lines.append("--- No trend observations recorded yet ---")

        return "\n".join(lines)

    def read_raw(self):
        """Return the full unabridged game log. Used for end-of-game reflection."""
        if not self._game_dir:
            return "[Medium term memory is empty — no active game]"
        return self._read_text("raw_log.txt")


    def close_game(self, final_stacks: dict, game_critique: str = ""):
        """
        Seals the buffer with final results and an optional game-level critique.
        Args:
        final_stacks: e.g. {"Hero": 620, "P2": 430, "P3": 450}
        game_critique: agent's high-level reflection on the session
        """
        stacks_str = " | ".join(f"{p}: {c}" for p, c in final_stacks.items())
        footer = (
            f"\n=== GAME {self.game_id} CLOSED | {datetime.now().isoformat()} ===\n"
            f"Final stacks: {stacks_str}\n"
            f"Hands played: {self.hands_played}\n"
            f"Game critique: {game_critique.strip() or 'pending'}\n"
            f"{'=' * 60}\n"
        )
        self._append_to("raw_log.txt", footer)
        self._append_to("trends.txt", footer)

    def purge_memory(self):
        """
        Return the full game record (raw log + trends concatenated) and wipe
        the game directory from disk.
        The returned string is what gets handed to the long-term reflection
        pipeline for player-profile updates.
        """
        if not self._game_dir:
            return "[Nothing to purge — no active game]"

        raw = self._read_text("raw_log.txt")
        trends = self._read_text("trends.txt")
        combined = f"{raw}\n\n{'=' * 60}\n\n{trends}"

        # Wipe game directory.
        for child in self._game_dir.iterdir():
            child.unlink()
        self._game_dir.rmdir()

        self._game_dir = None
        self.game_id = ""
        self.hands_played = 0

        return combined

  

  #internal class methods below

    @staticmethod
    def _empty_player_stats():
        return {
            "vpip": 0,
            "preflop_raise": 0,
            "preflop_3bet": 0,
            "continuation_bet": 0,
            "went_to_showdown": 0,
            "won_hand": 0,
            "net_chips": 0,
        }

    def _path(self, filename: str) -> Path:
        assert self._game_dir is not None, "Call new_game() before accessing memory."
        return self._game_dir / filename

    def _append_to(self, filename: str, text: str) -> None:
        with open(self._path(filename), "a", encoding="utf-8") as f:
            f.write(text)

    def _read_text(self, filename: str) -> str:
        p = self._path(filename)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def _read_json(self, filename: str) -> dict:
        p = self._path(filename)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_json(self, filename: str, data: dict) -> None:
        with open(self._path(filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)