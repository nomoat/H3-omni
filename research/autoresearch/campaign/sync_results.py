"""Fetch immutable terminal evidence and idempotently settle the local ledger."""
import argparse
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

from ledger import Campaign, atomic_write

TERMINAL = ("keep", "discard", "crash")
REMOTE_ATTEMPT_NOTE = "Remote fixed-oracle campaign execution"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def validate_record(record, candidate, entry, campaign_hash, number):
    require(type(record.get("candidate")) is int and record["candidate"] == number,
            "Terminal candidate number differs from evidence path")
    require(record.get("terminal") is True and record.get("status") in TERMINAL,
            "Evidence is not a complete terminal result")
    require(all(record.get(key) == value for key, value in entry.items()),
            "Terminal candidate identity differs from the local plan")
    require(all(record.get(key) == candidate.get(key)
                for key in ("commit", "parent_commit", "patch_sha256")),
            "Terminal candidate identity differs from the Git ledger")
    require(record.get("campaign_identity_sha256") == campaign_hash,
            "Terminal campaign identity differs")
    require(type(record.get("correctness")) is bool and type(record.get("fullgate")) is bool,
            "Terminal correctness/fullgate must be booleans")
    score = record.get("score_seconds")
    require(score is None or (type(score) in (int, float) and math.isfinite(score) and score > 0),
            "Terminal score must be finite positive E2E seconds")
    attempts = record.get("attempts")
    require(isinstance(attempts, list) and attempts, "Terminal result has no attempts")
    require(all(isinstance(a, dict) and a.get("stage") in ("build", "run") and
                a.get("outcome") in ("success", "candidate_error") for a in attempts),
            "Terminal attempts are invalid or include an infrastructure failure")
    require(isinstance(record.get("reason"), str), "Terminal result has no reason")
    if record["status"] != "crash":
        require(any(a["stage"] == "run" for a in attempts), "Terminal result has no run attempt")
    after = record.get("state_after")
    require(isinstance(after, dict) and set(after) == {"incumbent", "incumbent_score"},
            "Terminal state transition is missing")
    require(after["incumbent"] == (number if record["status"] == "keep" else record.get("benchmark_parent"))
            and type(after["incumbent_score"]) in (int, float)
            and math.isfinite(after["incumbent_score"]) and after["incumbent_score"] > 0,
            "Terminal state transition is inconsistent")
    if record["status"] == "keep":
        require(record["correctness"] and record["fullgate"] and score is not None
                and record.get("confirmation", {}).get("passed") is True
                and any(a["stage"] == "run" and a["outcome"] == "success" for a in attempts),
                "Kept result lacks a full correctness gate and confirmed timing")


def progress_identity(root, payload):
    progress = json.loads(payload)
    identity = progress.get("campaign_identity")
    require(isinstance(identity, dict), "Remote progress has no campaign identity")
    require(progress.get("campaign_identity_sha256") == identity_hash(identity),
            "Remote campaign identity hash is invalid")
    require(identity.get("plan_sha256") == digest((root / "plan.json").read_bytes()),
            "Remote progress belongs to another candidate plan")
    protocol = json.loads((root / "protocol.json").read_text())
    require(identity.get("prompt_sha256") == protocol["prompt_sha256"]
            and identity.get("limit") == protocol["rounds"],
            "Remote progress differs from the fixed campaign protocol")
    return progress


def install_archive(root, data):
    """Validate the whole download before replacing any local evidence file."""
    local = root / "evidence/search"
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            require(member.isfile() and (member.name == "progress.json" or
                    re.fullmatch(r"[0-9]{4}/result\.json", member.name)),
                    "Unexpected evidence archive member")
            require(member.name not in payloads, "Duplicate evidence archive member")
            target = local / member.name
            require(target.resolve().is_relative_to(local.resolve()), "Evidence path escapes local directory")
            payload = archive.extractfile(member).read()
            parsed = json.loads(payload)
            if member.name != "progress.json":
                require(parsed.get("terminal") is True, "Nonterminal result was published")
                if target.exists():
                    require(target.read_bytes() == payload, "Previously downloaded terminal evidence changed")
            payloads[member.name] = payload
    progress_path = local / "progress.json"
    old_progress = progress_identity(root, progress_path.read_bytes()) if progress_path.exists() else None
    progress_data = payloads.get("progress.json")
    progress = progress_identity(root, progress_data) if progress_data is not None else old_progress
    require(progress is not None or not payloads, "Terminal evidence has no campaign identity")
    if old_progress is not None and progress is not None:
        require(progress["campaign_identity_sha256"] == old_progress["campaign_identity_sha256"],
                "Previously recorded campaign identity changed")
    state = Campaign(root).status()
    plan = {entry["candidate"]: entry for entry in json.loads((root / "plan.json").read_text())}
    for name, payload in payloads.items():
        if name == "progress.json":
            continue
        number = int(name.split("/", 1)[0])
        require(number in state["candidates"] and number in plan, "Unknown terminal candidate")
        validate_record(json.loads(payload), state["candidates"][number], plan[number],
                        progress["campaign_identity_sha256"], number)
    for name, payload in payloads.items():
        atomic_write(local / name, payload)


def check_saved_raw(event, expected_hash):
    require(event.get("raw_result_sha256") == expected_hash,
            "Ledger evidence hash differs from the terminal result")
    path = event.get("raw_result_path")
    require(path and Path(path).is_file() and digest(Path(path).read_bytes()) == expected_hash,
            "Ledger's saved raw terminal evidence changed or disappeared")


def imported_attempts(candidate, result_path):
    """Return only attempts belonging to this remote terminal import.

    Compile/preflight attempts remain valid ledger history but do not advance
    this import's cursor. Identify ownership before checking the evidence hash:
    filtering on matching hashes would silently hide corrupted import records.
    The path also recognizes imports written by the original sync implementation.
    """
    result_path = result_path.resolve()
    owned = []
    for event in candidate["attempts"]:
        original = event.get("original_result_path")
        same_path = bool(original) and Path(original).resolve() == result_path
        marked = str(event.get("note", "")).startswith(REMOTE_ATTEMPT_NOTE)
        if same_path or marked:
            owned.append(event)
    return owned


def settle_results(campaign):
    """An imported attempt prefix is a durable cursor; never append it twice."""
    root = campaign.root
    local = root / "evidence/search"
    paths = sorted(local.glob("[0-9][0-9][0-9][0-9]/result.json"))
    if not paths:
        return campaign.status()
    progress = progress_identity(root, (local / "progress.json").read_bytes())
    plan = {entry["candidate"]: entry for entry in json.loads((root / "plan.json").read_text())}
    for path in paths:
        payload = path.read_bytes()
        record = json.loads(payload)
        number = int(path.parent.name)
        state = campaign.status()
        require(number in state["candidates"] and number in plan, "Unknown terminal candidate")
        candidate = state["candidates"][number]
        validate_record(record, candidate, plan[number], progress["campaign_identity_sha256"], number)
        expected_hash = digest(payload)
        existing = imported_attempts(candidate, path)
        require(len(existing) <= len(record["attempts"]), "Ledger has excess/duplicate attempt records")
        for index, event in enumerate(existing):
            expected = record["attempts"][index]
            require(all(event.get(key) == expected[key] for key in ("stage", "outcome")),
                    "Ledger attempts are not a prefix of the terminal evidence")
            check_saved_raw(event, expected_hash)
        if candidate["status"] in TERMINAL:
            require(len(existing) == len(record["attempts"]), "Finished candidate has incomplete attempts")
            check_saved_raw(candidate, expected_hash)
            require(candidate["status"] == record["status"] and
                    candidate["correctness"] == record["correctness"] and
                    candidate["fullgate"] == record["fullgate"] and
                    candidate["e2e_latency_seconds"] == record["score_seconds"] and
                    candidate["note"] == record["reason"], "Finished ledger values differ from terminal evidence")
            continue
        require(candidate["status"] == "sealed", "Terminal candidate has not been sealed locally")
        for index in range(len(existing), len(record["attempts"])):
            attempt = record["attempts"][index]
            campaign.attempt(number, attempt["stage"], attempt["outcome"], path,
                             note=f"{REMOTE_ATTEMPT_NOTE}; attempt {index + 1}")
        campaign.finish(number, record["status"], record["correctness"], record["score_seconds"], path,
                        fullgate=record["fullgate"], note=record["reason"])
    return campaign.status()


REMOTE_SCRIPT = """import io,json,tarfile,sys
from pathlib import Path
root=Path('/home/cj/h3-cuda/autoresearch-100/results')
paths=sorted(root.glob('[0-9][0-9][0-9][0-9]/result.json'))
if (root/'progress.json').is_file():paths.append(root/'progress.json')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for path in paths:
  data=path.read_bytes()
  try:record=json.loads(data)
  except (ValueError,UnicodeError):continue
  if path.name=='result.json' and record.get('terminal') is not True:continue
  info=tarfile.TarInfo(str(path.relative_to(root)));info.size=len(data)
  archive.addfile(info,io.BytesIO(data))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    require((root / "campaign.json").is_file(), "Campaign is not initialized")
    # Serialize complete syncs; Campaign still locks individual journal writes.
    with (root / ".sync.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = ["ssh", "-i", str(Path.home() / ".ssh/msb-build"), "-p", "2222",
                   "-o", "BatchMode=yes", "cj@129.212.240.50", "python3 -"]
        result = subprocess.run(command, input=REMOTE_SCRIPT.encode(), stdout=subprocess.PIPE,
                                check=True, timeout=60)
        install_archive(root, result.stdout)
        state = settle_results(Campaign(root))
        summary = {key: value for key, value in state.items() if key != "candidates"}
        progress_path = root / "evidence/search/progress.json"
        if progress_path.exists():
            progress = progress_identity(root, progress_path.read_bytes())
            summary["remote"] = {key: progress.get(key) for key in (
                "status", "active", "phase", "incumbent", "incumbent_score", "last_result", "error")}
        atomic_write(root / "status.json", (json.dumps(summary, indent=2) + "\n").encode())
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
