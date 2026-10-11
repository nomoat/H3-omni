#!/usr/bin/env python3
"""Observe the frozen worker through small files; never run HTTP or copy tensors.

Install this file outside seed/research/autoresearch/campaign: that directory is
part of the frozen worker identity. Example independent process:

  python3 /var/tmp/h3-search-100/telemetry/telemetry_observer.py watch \
    --stage /var/tmp/h3-search-100 \
    --shared-results /home/cj/h3-cuda/autoresearch-100/results \
    --outbox /home/cj/h3-cuda/autoresearch-100/telemetry \
    --worker-pid PID --session-id SLURM_JOB_ID

Run ``drain --producer-stopped`` before resuming a stopped worker, and after its
process group has stopped. This captures interrupted metadata and requests
collection; even drain never reads, hashes, copies or creates a large archive.
The external collector owns those operations. A SIGKILL needs a later drain.

Outbox v1: progress.json (worker snapshot), observer.json (observer status),
events/HASH.json (immutable observations), blobs/HASH.json (raw small JSON),
attempts/ID/index.json (latest metadata map), archives/ID.json (collection
descriptor). The observer-state.json checkpoint is private to this observer.
Only atomic final names are readable outputs. Timing-sensitive polling reads
fixed JSON metadata names, never arbitrary logs, tensor files or archive bytes.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import tempfile
import time


MAX_JSON_BYTES = 2 * 1024 * 1024
METADATA = (
    "inflight.json", "result.json", "infrastructure-error.json", "build/build.json",
    "screen/verification.json", "screen/units.json", "fullgate/verification.json",
    "fullgate/units.json", "benchmark/samples.json", "benchmark/benchmark.json",
    "confirmation/pairs.json",
) + tuple(f"confirmation/{index:02d}-{role}/{name}.json"
          for index in range(5) for role in ("parent", "candidate")
          for name in ("samples", "benchmark"))
TERMINAL = {"keep", "discard", "crash"}
CLOSED = {"terminal", "interrupted", "infrastructure_error", "terminal_pending_archive"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def attempt_id(candidate, record):
    started = record.get("started_unix")
    if type(started) not in (float, int) or not math.isfinite(started) or started <= 0:
        raise ValueError("Attempt has no finite positive recorded start time")
    return f"{candidate:04d}-" + sha(canonical(started))[:16]


def atomic_write(path, raw, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Immutable observer output changed: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def file_stat(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev)


def same_directory(path, inode):
    try:
        return path.stat().st_ino == inode
    except FileNotFoundError:
        return False


def process_identity(pid):
    """PID plus Linux start tick avoids mistaking a reused PID for our worker."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


class Observer:
    def __init__(self, stage, shared_results, outbox, session_id="manual",
                 discover_seconds=30, max_json_bytes=MAX_JSON_BYTES):
        self.stage = Path(stage).resolve()
        self.results = self.stage / "results"
        self.shared = Path(shared_results).resolve()
        self.outbox = Path(outbox).resolve()
        if self.outbox == self.results or self.results in self.outbox.parents:
            raise ValueError("Observer outbox must be outside frozen result directories")
        self.session_id = str(session_id)
        self.discover_seconds = discover_seconds
        self.max_json_bytes = max_json_bytes
        self.node = socket.gethostname()
        self.last_discover = float("-inf")
        self.cache = {}
        self.progress = {}
        self.dirty = False
        plan = json.loads((self.stage / "plan.json").read_text())
        self.plan = {entry["candidate"]: entry for entry in plan}
        if len(self.plan) != len(plan):
            raise ValueError("Plan has duplicate candidate IDs")
        self.plan_sha256 = sha((self.stage / "plan.json").read_bytes())
        state_path = self.outbox / "observer-state.json"
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {
            "schema_version": 1, "plan_sha256": self.plan_sha256,
            "campaign_identity_sha256": None, "attempts": {}, "progress_sha256": None}
        if self.state["plan_sha256"] != self.plan_sha256:
            raise ValueError("Observer outbox belongs to a different plan")

    def event(self, kind, attempt=None, **fields):
        identity = dict(schema_version=1, kind=kind,
                        candidate=attempt["candidate"] if attempt else None,
                        attempt_id=attempt["attempt_id"] if attempt else None, **fields)
        event_id = sha(canonical(identity))
        target = self.outbox / "events" / (event_id + ".json")
        # First observation wins; restart must not replace its time/session.
        if not target.exists():
            record = dict(identity, event_id=event_id, observed_unix=time.time(),
                          session_id=self.session_id, node=self.node)
            atomic_write(target, canonical(record) + b"\n", immutable=True)
        self.dirty = True

    def small_json(self, path):
        """Read stable, bounded JSON only; an unfinished write is retried."""
        path = Path(path)
        try:
            if path.is_symlink() or not path.is_file():
                return None
            before = file_stat(path)
            if before["size"] > self.max_json_bytes:
                self.event("metadata_skipped", source_path=str(path), reason="JSON exceeds observer byte limit", bytes=before["size"])
                return None
            key = str(path)
            if key in self.cache and self.cache[key][0] == before:
                return self.cache[key][1]
            with path.open("rb") as stream:
                raw = stream.read(self.max_json_bytes + 1)
            after = file_stat(path)
            if before != after or len(raw) != before["size"] or len(raw) > self.max_json_bytes:
                return None
            value = json.loads(raw)
            if not isinstance(value, (dict, list)):
                return None
            item = (value, raw, before)
            self.cache[key] = (before, item)
            return item
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
            return None

    def bind_progress(self):
        found = self.small_json(self.results / "progress.json")
        if not found:
            return
        progress, raw, stat = found
        identity = progress.get("campaign_identity_sha256")
        if not identity:
            raise ValueError("Worker progress has no campaign identity")
        old = self.state["campaign_identity_sha256"]
        if old and old != identity:
            raise ValueError("Worker campaign identity changed")
        self.state["campaign_identity_sha256"] = identity
        self.progress = progress
        digest = sha(raw)
        if self.state["progress_sha256"] != digest:
            atomic_write(self.outbox / "progress.json", raw)
            self.state["progress_sha256"] = digest
            self.event("worker_progress", source_sha256=digest,
                       state=progress.get("status"), phase=progress.get("phase"),
                       active=progress.get("active"), completed=progress.get("completed", []),
                       source_time_unix=stat["mtime_ns"] / 1e9)
            self.dirty = True

    def register(self, folder):
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.results):
            raise ValueError("Attempt directory escapes stage/results")
        found = self.small_json(folder / "inflight.json") or self.small_json(folder / "result.json")
        if not found:
            return None
        record = found[0]
        number = record.get("candidate")
        if (number not in self.plan or any(record.get(k) != v for k,v in self.plan[number].items())
                or record.get("campaign_identity_sha256") != self.state["campaign_identity_sha256"]):
            raise ValueError(f"Attempt identity differs from frozen plan/progress: {folder}")
        aid = attempt_id(number, record)
        attempts = self.state["attempts"]
        if aid not in attempts:
            identity = {key:record.get(key) for key in ("started_unix", "commit", "parent_commit",
                        "source_sha256", "patch_sha256", "campaign_identity_sha256")}
            attempts[aid] = dict(schema_version=1, candidate=number, attempt_id=aid,
                                 identity=identity, state="running", phase=None,
                                 counts_as_completed=False, source_directory=str(folder),
                                 metadata={}, archive=None, node=self.node)
            self.event("attempt_begin", attempts[aid], state="running", phase="build",
                       counts_as_completed=False, source_time_unix=record["started_unix"],
                       source_path=str(folder / "inflight.json"))
        attempt = attempts[aid]
        if attempt["source_directory"] != str(folder):
            attempt["source_directory"] = str(folder)
            self.dirty = True
        if folder.parent.name == "retries" and attempt["state"] == "running":
            self.interrupt(attempt, "Worker preserved this directory for a retry")
        return attempt

    def snapshot(self, attempt, relative, directory_inode):
        folder = Path(attempt["source_directory"])
        path = folder / relative
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Metadata symlink escapes its attempt directory")
        found = self.small_json(path)
        if not found:
            return None
        # Resume renames results/NNNN and creates a fresh directory. Never
        # publish a replacement execution's metadata under the old attempt ID.
        if not same_directory(folder, directory_inode):
            return None
        value, raw, stat = found
        digest = sha(raw)
        previous = attempt["metadata"].get(relative)
        if not previous or previous["sha256"] != digest:
            blob = f"blobs/{digest}.json"
            atomic_write(self.outbox / blob, raw, immutable=True)
            attempt["metadata"][relative] = dict(blob=blob, sha256=digest, bytes=len(raw))
            self.event("metadata_available", attempt, relative_path=relative, blob=blob,
                       sha256=digest, bytes=len(raw), source_time_unix=stat["mtime_ns"] / 1e9)
        return value

    def interrupt(self, attempt, reason, state="interrupted", safe_to_collect=False):
        if attempt["state"] == "terminal":
            return
        if attempt["state"] != state:
            attempt["state"] = state
            attempt["counts_as_completed"] = False
            self.event("attempt_interrupted", attempt, state=state, phase=attempt["phase"],
                       counts_as_completed=False, reason=reason, source_time_unix=None)
        safe_to_collect = safe_to_collect or bool((attempt.get("archive") or {}).get("safe_to_collect"))
        descriptor = dict(schema_version=1, candidate=attempt["candidate"], attempt_id=attempt["attempt_id"],
                          ready=False, kind="interrupted_directory", source_path=attempt["source_directory"],
                          source_stat=None, result_sha256=None, hash_pending=True,
                          safe_to_collect=safe_to_collect, node=self.node,
                          expected_started_unix=attempt["identity"]["started_unix"])
        self.archive_descriptor(attempt, descriptor)

    def archive_descriptor(self, attempt, descriptor):
        if attempt["archive"] != descriptor:
            attempt["archive"] = descriptor
            atomic_write(self.outbox / "archives" / (attempt["attempt_id"] + ".json"), canonical(descriptor) + b"\n")
            self.event("archive_available" if descriptor["ready"] else "archive_collection_requested",
                       attempt, archive_kind=descriptor["kind"],
                       **{key:value for key,value in descriptor.items()
                          if key not in ("schema_version", "candidate", "attempt_id", "node", "kind")})

    def observe_attempt(self, attempt, stopped=False):
        folder = Path(attempt["source_directory"])
        try:
            directory_inode = folder.stat().st_ino
        except FileNotFoundError:
            return
        live = self.small_json(folder / "inflight.json") or self.small_json(folder / "result.json")
        if (not live or not same_directory(folder, directory_inode)
                or attempt_id(attempt["candidate"], live[0]) != attempt["attempt_id"]):
            # A new attempt may already own results/NNNN. Never attach its files
            # to an old attempt. Discovery will find the preserved retry folder.
            return
        if attempt["state"] == "running" and self.progress.get("active") == attempt["candidate"]:
            phase = self.progress.get("phase")
            if phase != attempt["phase"]:
                attempt["phase"] = phase
                item = self.small_json(self.results / "progress.json")
                self.event("attempt_phase", attempt, state="running", phase=phase,
                           counts_as_completed=False, source_time_unix=item[2]["mtime_ns"] / 1e9 if item else None,
                           source_path=str(self.results / "progress.json"))
        for relative in METADATA:
            self.snapshot(attempt, relative, directory_inode)
        if not same_directory(folder, directory_inode):
            return
        result = self.small_json(folder / "result.json")
        result = result[0] if result else None
        if not same_directory(folder, directory_inode):
            return
        if result and result.get("terminal") is True:
            if attempt_id(attempt["candidate"], result) != attempt["attempt_id"]:
                raise ValueError("Terminal start time differs from inflight identity")
            if result.get("status") not in TERMINAL:
                raise ValueError("Worker terminal has an unknown decision")
            number = attempt["candidate"]
            if (any(result.get(k) != v for k,v in self.plan[number].items())
                    or result.get("campaign_identity_sha256") != self.state["campaign_identity_sha256"]):
                raise ValueError("Worker terminal differs from the frozen candidate identity")
            shared_record = self.small_json(self.shared / f"{number:04d}" / "result.json")
            archive = self.shared / f"{number:04d}" / "evidence.tar"
            committed = (number in self.progress.get("completed", []) and shared_record
                         and shared_record[0] == result and archive.is_file()
                         and not archive.with_suffix(".tar.pending").exists())
            if committed:
                descriptor = dict(schema_version=1, candidate=number, attempt_id=attempt["attempt_id"],
                                  ready=True, kind="worker_terminal_tar", source_path=str(archive),
                                  source_stat=file_stat(archive), result_sha256=sha(shared_record[1]),
                                  hash_pending=True, safe_to_collect=True, node=self.node,
                                  expected_started_unix=attempt["identity"]["started_unix"])
                self.archive_descriptor(attempt, descriptor)
                if attempt["state"] != "terminal":
                    attempt.update(state="terminal", counts_as_completed=True)
                    self.event("attempt_terminal", attempt, state=result["status"], phase=attempt["phase"],
                               counts_as_completed=True, decision=result["status"],
                               source_time_unix=result.get("finished_unix"), result_sha256=descriptor["result_sha256"])
                return
            if attempt["state"] != "terminal_pending_archive":
                attempt.update(state="terminal_pending_archive", counts_as_completed=False)
                self.dirty = True
        infrastructure = self.small_json(folder / "infrastructure-error.json")
        if infrastructure:
            self.interrupt(attempt, "Worker recorded an infrastructure error", "infrastructure_error", stopped)
        elif stopped:
            self.interrupt(attempt, "Caller confirmed producer process group stopped", safe_to_collect=True)

    def discover(self):
        folders = sorted(path for path in self.results.glob("[0-9][0-9][0-9][0-9]") if path.is_dir())
        folders += sorted(path for path in (self.results / "retries").glob("[0-9][0-9][0-9][0-9]-*") if path.is_dir())
        for folder in folders:
            self.register(folder)
        # Starting a new execution of one candidate closes its previous open
        # execution, even if the observer missed the rename between two polls.
        newest = {}
        for attempt in self.state["attempts"].values():
            n = attempt["candidate"]
            if n not in newest or attempt["identity"]["started_unix"] > newest[n]["identity"]["started_unix"]:
                newest[n] = attempt
        for attempt in self.state["attempts"].values():
            if attempt is not newest[attempt["candidate"]] and attempt["state"] == "running":
                self.interrupt(attempt, "A newer execution of this candidate was observed")
        self.last_discover = time.monotonic()

    def save(self):
        if not self.dirty:
            return
        for attempt in self.state["attempts"].values():
            target = self.outbox / "attempts" / attempt["attempt_id"] / "index.json"
            data = canonical(attempt) + b"\n"
            if not target.exists() or target.read_bytes() != data:
                atomic_write(target, data)
        atomic_write(self.outbox / "observer-state.json", canonical(self.state) + b"\n")
        self.dirty = False

    def poll(self, force_discover=False, stopped=False):
        self.bind_progress()
        if not self.state["campaign_identity_sha256"]:
            return
        discover = force_discover or time.monotonic() - self.last_discover >= self.discover_seconds
        if discover:
            self.discover()
        active = self.progress.get("active")
        if type(active) is int and active in self.plan:
            self.register(self.results / f"{active:04d}")
        for attempt in self.state["attempts"].values():
            if discover or stopped or attempt["state"] not in CLOSED or attempt["state"] == "terminal_pending_archive":
                self.observe_attempt(attempt, stopped=stopped)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("watch", "once", "drain"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--shared-results", type=Path, required=True)
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--session-id", default="manual")
    parser.add_argument("--worker-pid", type=int)
    parser.add_argument("--producer-stopped", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args()
    if args.command == "watch" and not args.worker_pid:
        parser.error("watch requires --worker-pid")
    if args.command == "drain" and not args.producer_stopped:
        parser.error("drain requires explicit --producer-stopped after all producer children stop")
    if args.poll_seconds < .2:
        parser.error("poll interval must be at least 0.2 seconds")
    if args.command == "drain" and args.worker_pid and process_identity(args.worker_pid) is not None:
        parser.error("worker PID is still alive; do not drain a running producer")
    args.outbox.mkdir(parents=True, exist_ok=True)
    with (args.outbox / ".observer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observer = Observer(args.stage, args.shared_results, args.outbox, args.session_id)
        worker_identity = process_identity(args.worker_pid) if args.worker_pid else None
        stopping = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, frame: stopping.append(number))
        try:
            os.nice(10)
        except OSError:
            pass
        while True:
            observer.poll(force_discover=args.command != "watch", stopped=args.command == "drain")
            alive = bool(worker_identity and process_identity(args.worker_pid) == worker_identity)
            status = dict(schema_version=1, session_id=args.session_id, node=observer.node,
                          worker_pid=args.worker_pid, worker_alive=alive, command=args.command,
                          last_poll_unix=time.time(), stopped_by_signal=stopping[-1] if stopping else None)
            atomic_write(args.outbox / "observer.json", canonical(status) + b"\n")
            if args.command != "watch" or stopping:
                break
            if not alive:
                observer.poll(force_discover=True)
                for attempt in observer.state["attempts"].values():
                    if attempt["state"] == "running":
                        observer.interrupt(attempt, "Worker PID exited; child process quiescence is not yet confirmed")
                observer.save()
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
