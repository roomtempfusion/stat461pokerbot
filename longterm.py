import json
import os
import re
from datetime import datetime
from pathlib import Path


class LongTermMemory:
    """
    Long term memory system inspired by mongodb.

    Each player gets one text file including us (self supervised strategy), + one index.json file
    that indexes store for quick lookup.

    Profile text files are weakly structured, but agent has free will in managing how 
    it is written.

    Use order:
        new_session() called before a game begins; loads existing profiles from disk into memory
        lookup_player() called by the agent at turn-start to retrieve an opponent's full profile text
        read_self_profile() called by the agent to read its own strategic notes
        list_known_players() called by the agent to enumerate known opponents optionally filtered by behavioural tags
        log_player_update() called by the reflection pipeline after each game to append new observations to an opponent's profile
        log_self_update() called by the reflection pipeline after each game to append new self-notes
        close_session() closes the session; rebuilds the index from disk
        rebuild_index() rebuiilds index.json from .txt files

    File layout under root/:
        players/
            <player_id>.txt full prose profile, one file per known opponent
        self.txt gent's own strategic self-notes
        index.json  queryable metadata sidecar (tags, last_seen, etc.)
    """

    INDEX_FILENAME = "index.json"
    PLAYERS_DIRNAME = "players"
    SELF_FILENAME = "self.txt"

    _NO_PROFILE = "No prior profile for this player."
    _SELF_SCAFFOLD = (
        "# Self Profile\n"
        "## Last updated: never\n"
        "## Games played: 0\n\n"
        "## Current strategy\n"
        "(no strategy notes yet)\n\n"
        "## Known leaks\n"
        "(none identified yet)\n\n"
        "## Lessons learned\n"
        "(none yet)\n\n"
        "## Open questions\n"
        "(none yet)\n"
    )


#touch these regexes at own risk
    _HEADER_RE = re.compile(r"^##\s*([^:]+?)\s*:\s*(.*?)\s*$")
    _SECTION_RE = re.compile(r"^##\s+([^:]+?)\s*$")

    def __init__(self, root: str = "./memory/long_term"):
        self.root = Path(root)
        self.players_dir = self.root / self.PLAYERS_DIRNAME
        self.index_path = self.root / self.INDEX_FILENAME
        self.self_path = self.root / self.SELF_FILENAME

        self.root.mkdir(parents=True, exist_ok=True)
        self.players_dir.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, dict] = self._load_index()
        self._session_active: bool = False


    def new_session(self, game_id: str):
        """
        Loads index from disk so the agent has access to all previously observed player profiles.
        Args:
        game_id: identifier for the current game, e.g. "game_003"
        """
        self._index = self._load_index()
        self._session_active = True
        self._session_game_id = game_id

    def close_session(self):
        """
        Rebuilds the index from .txt files on disk
        to catch any changes by log_player_update() writes.
        """
        assert self._session_active, "Call new_session() before close_session()."
        self.rebuild_index()
        self._session_active = False


    def lookup_player(self, player_id: str):
        """
        Returns the full profile text for player_id.
        Returns a _NO_PROFILE constant if not found.
        """
        path = self._player_path(player_id)
        if not path.exists():
            return self._NO_PROFILE
        return path.read_text(encoding="utf-8")

    def read_self_profile(self):
        """
        Returns the agent's own strategic self-profile. Creates the file with header
        first if does not exist and returns that.
        """
        if not self.self_path.exists():
            self.self_path.write_text(self._SELF_SCAFFOLD, encoding="utf-8")
        return self.self_path.read_text(encoding="utf-8")

    def list_known_players(self, filter_tags: list[str] | None = None):
        """
        Return index entries for all known players, optionally filtered by tag.
        A player is included if at least one of their tags appears in filter_tags. 
        Tag matching is case-insensitive. Passing an empty list filters to nothing;
        passing None returns all.

        Returns a list of dictionaries:
        Each returned dict has the shape:
            {
                "id":str,
                "tags": list[str],
                "last_seen" str,
                "games_observed":int,
                "summary":str,
            }

        Results are ordered most-recently-seen first, then alphabetically by id.
        """
        if filter_tags is None:
            wanted: set[str] | None = None
        else:
            wanted = {t.strip().lower() for t in filter_tags if t.strip()}

        results: list[dict] = []
        for player_id, entry in self._index.items():
            if wanted is not None:
                player_tags = {t.lower() for t in entry.get("tags", [])}
                if not (wanted & player_tags):
                    continue
            results.append({"id": player_id, **entry})

        results.sort(
            key=lambda r: (r.get("last_seen", ""), r["id"]),
            reverse=True,
        )
        return results


    def log_player_update(self, player_id: str, observation: str, tags: list[str] | None = None):
        """
        Appends a new observation to an opponent's profile after end-of-game
        reflection. Creates if it does not exist yet.

        Args:
        player_id: opponent label, e.g. "P2"
        observation: free-prose finding, e.g. "Consistently 3-bets light from the BTN; likely a polarised 3-bet range."
        tags:        behavioural tags to attach or refresh, e.g. ["aggressive", "bluff-heavy"]. Merged with any existing tags in the index.
        """
        path = self._player_path(player_id)
        timestamp = datetime.now().isoformat()
        game_ref = getattr(self, "_session_game_id", "unknown")


        #if not found create.
        if not path.exists():
            header = (
                f"# Player Profile: {player_id}\n"
                f"## Tags: {', '.join(tags or [])}\n"
                f"## Last seen: {timestamp}\n"
                f"## Games observed: 1\n\n"
                f"## Summary\n"
                f"(no summary yet — see observations below)\n\n"
                f"## Observations\n"
            )
            self._write(path, header, mode="w")
            games_observed = 1
        else:
            games_observed = self._index.get(player_id, {}).get("games_observed", 0) + 1

        #write
        entry = (f"[{timestamp} | {game_ref}]\n" f"{observation.strip()}\n\n")
        self._write(path, entry, mode="a")
        self._patch_headers(path, {
            "last seen": timestamp,
            "games observed": str(games_observed),
            "tags": ", ".join(self._merge_tags(self._index.get(player_id, {}).get("tags", []), tags or []))})
        self._index[player_id] = self._parse_profile(path.read_text(encoding="utf-8"))
        self._save_index()

    def log_self_update(self, observation: str, section: str = "Lessons learned"):
        """
        Append a new self-note after end-of-game reflection. Scaffolds
        self.txt if it does not exist yet.

        Args:
        observation: free-prose strategic note, e.g. "Leaking chips by calling too wide on the river vs tight opponents."
        section:     which section header to append under. Defaults to "Lessons learned". Must already exist in the file.
        """
        _ = self.read_self_profile()
        timestamp = datetime.now().isoformat()
        game_ref = getattr(self, "_session_game_id", "unknown")
        entry = f"[{timestamp} | {game_ref}] {observation.strip()}\n"

        text = self.self_path.read_text(encoding="utf-8")
        target = f"## {section}"
        if target not in text:
            text += f"\n## {section}\n{entry}"
        else:
            lines = text.splitlines(keepends=True)
            out: list[str] = []
            for line in lines:
                out.append(line)
                if line.strip() == target:
                    out.append(entry)
            text = "".join(out)

        self.self_path.write_text(text, encoding="utf-8")
        self._patch_headers(self.self_path, {"last updated": timestamp})


    def rebuild_index(self):
        """
        Helper to rebuild index.json from the .txt files on disk.
        """
        new_index: dict[str, dict] = {}
        for txt_path in self.players_dir.glob("*.txt"):
            player_id = txt_path.stem
            try:
                text = txt_path.read_text(encoding="utf-8")
            except OSError:
                continue
            new_index[player_id] = self._parse_profile(text)
        self._index = new_index
        self._save_index()



#Helper functions
    def _player_path(self, player_id: str):
        if "/" in player_id or "\\" in player_id or ".." in player_id:
            raise ValueError(f"Invalid player_id: {player_id!r}")
        return self.players_dir / f"{player_id}.txt"

    def _write(self, path: Path, text: str, mode: str = "a"):
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)

    def _load_index(self):
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save_index(self):
        self.index_path.write_text(
            json.dumps(self._index, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _parse_profile(self, text: str):
        entry = {"tags": [], "last_seen": "", "games_observed": 0, "summary": ""}
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            m = self._HEADER_RE.match(line)
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).strip()
                if key == "tags":
                    entry["tags"] = [t.strip() for t in val.split(",") if t.strip()]
                elif key == "last seen":
                    entry["last_seen"] = val
                elif key == "games observed":
                    try:
                        entry["games_observed"] = int(val)
                    except ValueError:
                        pass
                i += 1
                continue

            m = self._SECTION_RE.match(line)
            if m and m.group(1).strip().lower() == "summary":
                j = i + 1
                buf: list[str] = []
                while j < len(lines) and not lines[j].lstrip().startswith("##"):
                    buf.append(lines[j])
                    j += 1
                entry["summary"] = self._first_paragraph(buf)
                i = j
                continue

            i += 1

        return entry

    @staticmethod
    def _first_paragraph(lines: list[str]):
        paragraph: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if paragraph:
                    break
                continue
            paragraph.append(stripped)
        return " ".join(paragraph)

    @staticmethod
    def _merge_tags(existing: list[str], new: list[str]):
        seen: set[str] = set()
        merged: list[str] = []
        for tag in existing + new:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                merged.append(tag)
        return merged

    def _patch_headers(self, path: Path, updates: dict[str, str]):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        for line in lines:
            m = self._HEADER_RE.match(line.rstrip("\n"))
            if m and m.group(1).strip().lower() in updates:
                key_display = m.group(1)
                new_val = updates[m.group(1).strip().lower()]
                line = f"## {key_display}: {new_val}\n"
            out.append(line)
        path.write_text("".join(out), encoding="utf-8")