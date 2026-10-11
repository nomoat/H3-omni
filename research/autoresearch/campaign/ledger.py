#!/usr/bin/env python3
"""Append-only experiment journal and isolated Git candidate trees.

Only the campaign's new repository is changed. ``seed_dir`` is an extracted,
accepted source snapshot, not a branch to reset in the developer's repository.
Examples (all subcommands print JSON):

  ledger.py init --root RUN --seed-dir EXTRACTED_SOURCE --rounds 100
  ledger.py start --root RUN --config experiment.json --hypothesis 'Fuse norm'
  # Edit the returned worktree, then preserve its exact source before execution.
  ledger.py seal --root RUN --candidate 1
  ledger.py attempt --root RUN --candidate 1 --stage build --outcome success
  ledger.py finish --root RUN --candidate 1 --status keep --correctness true \
      --fullgate --latency 12.3 --raw-result measured-result.json

An infra_error attempt is retryable and does not complete a round. A candidate
build failure can complete a crash round, but duplicate/empty patches cannot.
Keep requires the full correctness gate. The only ranking value is E2E seconds;
all other measurements remain uninterpreted in the immutable raw result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


class LedgerError(RuntimeError):
    pass


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def git(directory: Path, *args: str) -> str:
    run = subprocess.run(["git", "-C", str(directory), *args],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if run.returncode:
        raise LedgerError(f"git {' '.join(args)} failed: {run.stderr.strip()}")
    return run.stdout.strip()


class Campaign:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.repo = self.root / "repo"
        self.journal = self.root / "events.jsonl"

    @contextmanager
    def locked(self):
        if not (self.root / "campaign.json").is_file():
            raise LedgerError("Campaign is not initialized")
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def events(self) -> list[dict]:
        records = []
        previous = "0" * 64
        raw = self.journal.read_bytes() if self.journal.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise LedgerError("Incomplete journal tail; use recover-tail explicitly")
        for line_number, line in enumerate(raw.splitlines(), 1):
            try:
                record = json.loads(line)
                supplied = record.pop("event_hash")
                if (record["seq"] != line_number or record["prev_hash"] != previous
                        or digest(encoded(record)) != supplied):
                    raise ValueError("event hash/sequence mismatch")
                record["event_hash"] = supplied
            except (ValueError, KeyError, TypeError) as exc:
                raise LedgerError(f"Invalid journal line {line_number}: {exc}") from exc
            previous = supplied
            records.append(record)
        return records

    def append(self, kind: str, **fields: Any) -> dict:
        # Caller holds the campaign lock.
        records = self.events()
        event = {"seq": len(records) + 1, "time": now(), "kind": kind,
                 "prev_hash": records[-1]["event_hash"] if records else "0" * 64,
                 **fields}
        event["event_hash"] = digest(encoded(event))
        with self.journal.open("ab", buffering=0) as out:
            data = encoded(event) + b"\n"
            written = 0
            while written < len(data):
                written += out.write(data[written:])
            os.fsync(out.fileno())
        return event

    def _state(self) -> dict:
        meta = json.loads((self.root / "campaign.json").read_text())
        events = self.events()
        if not events or events[0]["kind"] != "init" or any(
                events[0].get(key) != value for key, value in meta.items()):
            raise LedgerError("Campaign metadata differs from its initialization record")
        candidates = {}
        incumbent = meta["baseline_commit"]
        for event in events:
            number = event.get("candidate")
            if event["kind"] == "start":
                candidates[number] = {**event, "status": "editing", "attempts": []}
            elif event["kind"] == "seal":
                candidates[number].update(event, status="sealed")
            elif event["kind"] == "attempt":
                candidates[number]["attempts"].append(event)
            elif event["kind"] == "finish":
                candidates[number].update(event)
                if event["status"] == "keep":
                    incumbent = candidates[number]["commit"]
        complete = sum(c["status"] in ("keep", "discard", "crash")
                       for c in candidates.values())
        return {"root": str(self.root), "max_rounds": meta["max_rounds"],
                "baseline_commit": meta["baseline_commit"],
                "incumbent_commit": incumbent, "completed_rounds": complete,
                "remaining_rounds": meta["max_rounds"] - complete,
                "candidates": candidates}

    def status(self) -> dict:
        with self.locked():
            return self._state()

    @classmethod
    def init(cls, root: str | Path, seed_dir: str | Path, rounds: int = 100) -> dict:
        campaign = cls(root)
        seed = Path(seed_dir).resolve()
        if rounds < 1 or not seed.is_dir():
            raise LedgerError("rounds must be positive and seed_dir must exist")
        if campaign.root == seed or seed in campaign.root.parents:
            raise LedgerError("Campaign must be outside the extracted seed directory")
        if campaign.root.exists():
            raise LedgerError("Campaign root already exists; use status to resume")
        campaign.root.mkdir(parents=True)
        excluded = {".git", "__pycache__", ".pytest_cache", "build-cuda", "runs"}
        shutil.copytree(seed, campaign.repo,
                        ignore=lambda _directory, names: [n for n in names if n in excluded],
                        symlinks=True)
        git(campaign.repo, "init", "-b", "seed")
        git(campaign.repo, "config", "user.name", "H3 Autoresearch")
        git(campaign.repo, "config", "user.email", "h3-autoresearch@localhost")
        git(campaign.repo, "config", "commit.gpgsign", "false")
        git(campaign.repo, "config", "core.hooksPath", "/dev/null")
        # The accepted tar is already the boundary; include ignored source files
        # that an upstream .gitignore might otherwise omit from its exact seed.
        git(campaign.repo, "add", "--force", "--all")
        git(campaign.repo, "commit", "--allow-empty", "-m", "Accepted seed snapshot")
        baseline = git(campaign.repo, "rev-parse", "HEAD")
        meta = {"schema_version": 1, "created": now(), "seed_dir": str(seed),
                "max_rounds": rounds, "baseline_commit": baseline,
                "baseline_tree": git(campaign.repo, "rev-parse", "HEAD^{tree}")}
        atomic_write(campaign.root / "campaign.json", encoded(meta) + b"\n")
        with campaign.locked():
            campaign.append("init", **meta)
        return campaign.status()

    def start(self, config: dict, hypothesis: str, parent_commit: str | None = None) -> dict:
        if not isinstance(config, dict) or not hypothesis.strip():
            raise LedgerError("config must be an object and hypothesis must be nonempty")
        with self.locked():
            state = self._state()
            if state["completed_rounds"] >= state["max_rounds"]:
                raise LedgerError("Campaign round budget is complete")
            parent = parent_commit or state["incumbent_commit"]
            if parent.startswith("-"):
                raise LedgerError("Invalid parent commit")
            parent = git(self.repo, "rev-parse", "--verify", parent + "^{commit}")
            known = {state["baseline_commit"]} | {
                c["commit"] for c in state["candidates"].values() if "commit" in c}
            if parent not in known:
                raise LedgerError("Parent must be the baseline or a sealed candidate commit")
            number = max(state["candidates"], default=0) + 1
            folder = self.root / "artifacts" / f"{number:04d}"
            tree = self.root / "worktrees" / f"{number:04d}"
            if tree.exists() or folder.exists():
                raise LedgerError("Unjournaled candidate directory exists; preserve and inspect it")
            folder.mkdir(parents=True)
            config_bytes = encoded(config) + b"\n"
            atomic_write(folder / "config.json", config_bytes)
            # Record before creating the worktree. If Git is interrupted, resume
            # can recreate this still-empty worktree from the recorded parent.
            event = self.append("start", candidate=number, parent_commit=parent,
                                worktree=str(tree), artifact_dir=str(folder),
                                hypothesis=hypothesis, config=config,
                                config_sha256=digest(config_bytes))
            git(self.repo, "worktree", "add", "--detach", str(tree), parent)
            return event

    def resume(self, number: int) -> dict:
        """Return candidate state, repairing only a missing unsealed worktree."""
        with self.locked():
            candidate = self._candidate(number, ("editing", "sealed"))
            tree = Path(candidate["worktree"])
            if not tree.exists():
                if candidate["status"] != "editing":
                    raise LedgerError("Sealed worktree is missing; preserve and inspect its artifacts")
                git(self.repo, "worktree", "prune")
                git(self.repo, "worktree", "add", "--detach", str(tree), candidate["parent_commit"])
            return candidate

    def _candidate(self, number: int, allowed: tuple[str, ...]) -> dict:
        candidate = self._state()["candidates"].get(number)
        if candidate is None or candidate["status"] not in allowed:
            raise LedgerError(f"Candidate {number} must be in state {allowed}")
        return candidate

    def seal(self, number: int) -> dict:
        with self.locked():
            candidate = self._candidate(number, ("editing",))
            tree = Path(candidate["worktree"])
            git(tree, "add", "--all")
            patch = subprocess.check_output([
                "git", "-C", str(tree), "diff", "--cached", "--binary",
                candidate["parent_commit"], "--", ".", ":(exclude).autoresearch"])
            if not patch:
                raise LedgerError("Empty source patch is not a distinct candidate")
            patch_hash = digest(patch)
            # Metadata is stored outside the tree so tree identity means source.
            source_tree = git(tree, "write-tree")
            state = self._state()
            baseline_tree = git(self.repo, "rev-parse", state["baseline_commit"] + "^{tree}")
            for other in state["candidates"].values():
                if (other.get("patch_sha256") == patch_hash
                        or other.get("source_tree") == source_tree):
                    raise LedgerError(f"Duplicate source candidate: {other['candidate']}")
            if source_tree == baseline_tree:
                raise LedgerError("Candidate reproduces the unchanged baseline")
            folder = Path(candidate["artifact_dir"])
            atomic_write(folder / "candidate.patch", patch)
            message = (f"Candidate {number:04d}: {candidate['hypothesis']}\n\n"
                       f"Parent: {candidate['parent_commit']}\n"
                       f"Config-SHA256: {candidate['config_sha256']}\n"
                       f"Patch-SHA256: {patch_hash}\n")
            head = git(tree, "rev-parse", "HEAD")
            if head == candidate["parent_commit"]:
                git(tree, "commit", "-m", message)
                commit = git(tree, "rev-parse", "HEAD")
            elif (git(tree, "show", "-s", "--format=%P", "HEAD") == candidate["parent_commit"]
                  and git(tree, "show", "-s", "--format=%B", "HEAD") == message.strip()
                  and git(tree, "rev-parse", "HEAD^{tree}") == source_tree):
                # A previous seal committed successfully but was interrupted
                # before its journal append. Reuse that exact commit.
                commit = head
            else:
                raise LedgerError("Unrecorded HEAD change; refusing to seal")
            # Detached commits are also retained by an explicit ref even if a
            # user later prunes the candidate worktree.
            git(self.repo, "update-ref", f"refs/candidates/{number:04d}", commit)
            return self.append("seal", candidate=number, commit=commit,
                               source_tree=source_tree, patch_sha256=patch_hash,
                               patch_path=str(folder / "candidate.patch"))

    def _raw(self, candidate: dict, raw_result: str | Path | None,
             label: str) -> dict:
        if raw_result is None:
            return {"raw_result_path": None, "raw_result_sha256": None}
        source = Path(raw_result).resolve()
        if not source.is_file():
            raise LedgerError("raw_result must name an existing local file")
        data = source.read_bytes()
        target = Path(candidate["artifact_dir"]) / f"{label}-{digest(data)[:16]}.raw"
        atomic_write(target, data)
        return {"raw_result_path": str(target), "original_result_path": str(source),
                "raw_result_sha256": digest(data)}

    def attempt(self, number: int, stage: str, outcome: str,
                raw_result: str | Path | None = None, note: str = "") -> dict:
        if stage not in ("build", "run") or outcome not in (
                "success", "candidate_error", "infra_error"):
            raise LedgerError("Invalid attempt stage/outcome")
        with self.locked():
            candidate = self._candidate(number, ("sealed",))
            return self.append("attempt", candidate=number, stage=stage, outcome=outcome,
                               note=note, **self._raw(candidate, raw_result, "attempt"))

    def finish(self, number: int, status: str, correctness: bool,
               latency: float | None, raw_result: str | Path | None,
               fullgate: bool = False, note: str = "") -> dict:
        if status not in ("keep", "discard", "crash") or type(correctness) is not bool:
            raise LedgerError("Invalid terminal status or correctness boolean")
        if latency is not None and (not math.isfinite(latency) or latency <= 0):
            raise LedgerError("E2E latency must be finite and strictly positive")
        if status == "keep" and (not correctness or not fullgate or latency is None):
            raise LedgerError("keep requires correctness, fullgate, and E2E latency")
        if status != "crash" and raw_result is None:
            raise LedgerError("A non-crash finish requires a raw result file")
        with self.locked():
            candidate = self._candidate(number, ("sealed",))
            valid_attempts = [a for a in candidate["attempts"]
                              if a["outcome"] != "infra_error"]
            if not valid_attempts:
                raise LedgerError("No actual build/run attempt; infrastructure errors do not count")
            if candidate["attempts"][-1]["outcome"] == "infra_error":
                raise LedgerError("Last attempt was an infrastructure failure; retry this candidate")
            if status != "crash" and not any(a["stage"] == "run" for a in valid_attempts):
                raise LedgerError("A non-crash finish requires an actual run attempt")
            if status == "keep" and not any(a["stage"] == "run" and a["outcome"] == "success"
                                            for a in valid_attempts):
                raise LedgerError("keep requires a successful run attempt")
            state = self._state()
            if state["completed_rounds"] >= state["max_rounds"]:
                raise LedgerError("Campaign round budget is complete")
            worktree = Path(candidate["worktree"])
            if (git(worktree, "rev-parse", "HEAD") != candidate["commit"]
                    or git(worktree, "status", "--porcelain")):
                raise LedgerError("Candidate source changed after seal; cannot finish")
            return self.append("finish", candidate=number, status=status,
                               correctness=correctness, fullgate=fullgate,
                               e2e_latency_seconds=latency,
                               completed_round=state["completed_rounds"] + 1, note=note,
                               **self._raw(candidate, raw_result, "result"))

    def recover_tail(self) -> dict:
        """Recover only an interrupted final write, never a bad complete line."""
        with self.locked():
            raw = self.journal.read_bytes()
            if not raw or raw.endswith(b"\n"):
                self.events()  # Complete but tampered journals still fail closed.
                return {"recovered": False}
            end = raw.rfind(b"\n") + 1
            retained, tail = raw[:end], raw[end:]
            # Validate retained events without replacing the live journal first.
            previous = "0" * 64
            for seq, line in enumerate(retained.splitlines(), 1):
                event = json.loads(line)
                supplied = event.pop("event_hash")
                if (event["seq"] != seq or event["prev_hash"] != previous
                        or digest(encoded(event)) != supplied):
                    raise LedgerError("Complete records are corrupt; refusing tail recovery")
                previous = supplied
            backup = self.root / f"incomplete-tail-{digest(raw)[:16]}.bin"
            atomic_write(backup, raw)
            atomic_write(self.journal, retained)
            return self.append("recover_tail", recovered=True, backup=str(backup),
                               removed_bytes=len(tail), removed_sha256=digest(tail))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "start", "resume", "seal", "attempt", "finish", "status", "recover-tail"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        if name == "init":
            command.add_argument("--seed-dir", required=True)
            command.add_argument("--rounds", type=int, default=100)
        elif name == "start":
            command.add_argument("--config", required=True)
            command.add_argument("--hypothesis", required=True)
            command.add_argument("--parent-commit")
        elif name in ("resume", "seal", "attempt", "finish"):
            command.add_argument("--candidate", type=int, required=True)
        if name == "attempt":
            command.add_argument("--stage", choices=("build", "run"), required=True)
            command.add_argument("--outcome", choices=("success", "candidate_error", "infra_error"), required=True)
        if name in ("attempt", "finish"):
            command.add_argument("--raw-result")
            command.add_argument("--note", default="")
        if name == "finish":
            command.add_argument("--status", choices=("keep", "discard", "crash"), required=True)
            command.add_argument("--correctness", choices=("true", "false"), required=True)
            command.add_argument("--latency", type=float)
            command.add_argument("--fullgate", action="store_true")
    args = parser.parse_args(argv)
    campaign = Campaign(args.root)
    try:
        if args.command == "init":
            result = Campaign.init(args.root, args.seed_dir, args.rounds)
        elif args.command == "start":
            result = campaign.start(json.loads(Path(args.config).read_text()), args.hypothesis,
                                    args.parent_commit)
        elif args.command == "seal":
            result = campaign.seal(args.candidate)
        elif args.command == "resume":
            result = campaign.resume(args.candidate)
        elif args.command == "attempt":
            result = campaign.attempt(args.candidate, args.stage, args.outcome, args.raw_result, args.note)
        elif args.command == "finish":
            result = campaign.finish(args.candidate, args.status, args.correctness == "true",
                                     args.latency, args.raw_result, args.fullgate, args.note)
        elif args.command == "recover-tail":
            result = campaign.recover_tail()
        else:
            result = campaign.status()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (LedgerError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
