#!/usr/bin/env python3
"""Render local, read-only campaign evidence into report.json and report.html.

Usage: python3 research/autoresearch/campaign/report.py [--root RUN]
       python3 research/autoresearch/campaign/report.py --self-test

Only these two report outputs are replaced. No Git, GPU, network, journal or
judge operation is performed. Sealed sources and preflight attempts never count
as completed trials. Compact terminal evidence is authoritative; progress and
ledger settlement are shown separately because synchronization can lag.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import tempfile
from urllib.parse import quote


TERMINAL = {"keep", "discard", "crash"}
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs/20261010-100"


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(root, relative, inputs, warnings, required=False):
    path = root / relative
    if not path.is_file():
        if required:
            raise ValueError(f"Missing required input: {path}")
        return None
    raw = path.read_bytes()
    inputs[relative] = {"sha256": sha256(raw), "bytes": len(raw)}
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as error:
        if required:
            raise ValueError(f"Invalid required input: {path}") from error
        warnings.append(f"{relative} is incomplete or not valid JSON; not used this time.")
        return None


def read_events(root, inputs, warnings):
    path = root / "events.jsonl"
    if not path.is_file():
        warnings.append("events.jsonl is missing; cannot check the local ledger.")
        return []
    raw = path.read_bytes()
    inputs["events.jsonl"] = {"sha256": sha256(raw), "bytes": len(raw)}
    events, previous = [], "0" * 64
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        if not line.endswith(b"\n"):
            warnings.append("The last line of events.jsonl is incomplete; only the complete records before it are read.")
            break
        try:
            event = json.loads(line)
            payload = {k: v for k, v in event.items() if k != "event_hash"}
            if (event["seq"] != index or event["prev_hash"] != previous
                    or event["event_hash"] != sha256(encoded(payload))):
                raise ValueError("hash chain differs")
        except (ValueError, KeyError, TypeError) as error:
            warnings.append(f"events.jsonl line {index} failed validation; only the valid records before it are read.")
            break
        events.append(event)
        previous = event["event_hash"]
    return events


def terminal_problem(record, entry, progress, root):
    """Fail closed on incomplete/foreign evidence; do not re-run the judge."""
    if not isinstance(record, dict):
        return "result is not an object"
    if record.get("terminal") is not True or record.get("status") not in TERMINAL:
        return "no complete final state"
    if any(record.get(key) != value for key, value in entry.items()):
        return "result identity differs from the plan"
    if (progress.get("campaign_identity_sha256") and
            record.get("campaign_identity_sha256") != progress["campaign_identity_sha256"]):
        return "campaign identity differs from progress"
    attempts = record.get("attempts", [])
    if not isinstance(attempts, list) or any(not isinstance(a, dict) for a in attempts):
        return "invalid attempts format"
    real_attempts = [a for a in attempts if isinstance(a, dict)
                     and a.get("stage") in ("build", "run")
                     and a.get("outcome") in ("success", "candidate_error")]
    if not real_attempts or attempts[-1].get("outcome") == "infra_error":
        return "no real candidate build/run result, or the last attempt was an infrastructure error"
    if record["status"] != "crash" and not any(a["stage"] == "run" for a in real_attempts):
        return "non-crash final state has no run record"
    if type(record.get("correctness")) is not bool or not positive(record.get("finished_unix")):
        return "no explicit correctness result or finish time"
    parent = record.get("benchmark_parent")
    if type(parent) is not int or parent < 0:
        return "no measured benchmark_parent"
    source = root / f"sources/{entry['candidate']:04d}.cu"
    if not source.is_file() or sha256(source.read_bytes()) != entry["source_sha256"]:
        return "local candidate source missing or hash mismatch"
    score = record.get("score_seconds")
    if score is not None and not positive(score):
        return "invalid cold-start seconds"
    if score is not None:
        samples = record.get("samples", [])
        if not isinstance(samples, list) or len(samples) != 3 or not all(map(positive, samples)):
            return "fewer than three valid raw initial samples"
        confirmation = record.get("confirmation")
        if confirmation is not None:
            if not isinstance(confirmation, dict):
                return "invalid confirmation format"
            pairs = confirmation.get("pairs", [])
            if (not isinstance(pairs, list) or len(pairs) != 5 or not all(isinstance(pair, dict) and
                    positive(pair.get("parent")) and positive(pair.get("candidate")) for pair in pairs)):
                return "paired confirmation samples incomplete"
            expected = statistics.median(pair["candidate"] for pair in pairs)
        else:
            expected = statistics.median(samples)
        if not math.isclose(score, expected, rel_tol=1e-10, abs_tol=1e-10):
            return "score differs from the median of the raw samples"
    if record["status"] == "keep" and not (
            record["correctness"] is True and record.get("fullgate") is True
            and isinstance(record.get("confirmation"), dict)
            and record["confirmation"].get("passed") is True and positive(score)):
        return "keep without fullgate, confirmation or a valid score"
    return None


def collect(root):
    root = Path(root).resolve()
    warnings, inputs = [], {}
    plan = read_json(root, "plan.json", inputs, warnings, required=True)
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a nonempty list")
    numbers = [entry["candidate"] for entry in plan]
    if any(type(n) is not int or n <= 0 for n in numbers) or numbers != sorted(set(numbers)):
        raise ValueError("Plan IDs must be unique, positive and ordered")
    status = read_json(root, "status.json", inputs, warnings) or {}
    progress = read_json(root, "evidence/search/progress.json", inputs, warnings) or {}
    protocol = read_json(root, "protocol.json", inputs, warnings) or {}
    preflight = read_json(root, "evidence/preflight/baseline/benchmark.json", inputs, warnings) or {}
    calibration = read_json(root, "evidence/calibration/seed-verification.json", inputs, warnings) or {}
    events = read_events(root, inputs, warnings)
    starts = {e["candidate"]: e for e in events if e["kind"] == "start"}
    seals = {e["candidate"]: e for e in events if e["kind"] == "seal"}
    finishes = {e["candidate"]: e for e in events if e["kind"] == "finish"}
    baseline_commit = next((e["baseline_commit"] for e in events if e["kind"] == "init"),
                           status.get("baseline_commit"))
    commit_numbers = {entry["commit"]: entry["candidate"] for entry in plan}
    commit_numbers[baseline_commit] = 0
    budget = status.get("max_rounds", protocol.get("rounds", len(plan)))
    rows, source_hashes = [], set()
    for entry in plan:
        number = entry["candidate"]
        result_path = f"evidence/search/{number:04d}/result.json"
        record = read_json(root, result_path, inputs, warnings)
        problem = terminal_problem(record, entry, progress, root) if record is not None else None
        if record is not None and not problem and entry["source_sha256"] in source_hashes:
            problem = "same source as an already completed candidate"
        complete = record is not None and problem is None
        if complete and record["benchmark_parent"] not in {0, *numbers}:
            problem, complete = "benchmark_parent is not in this plan", False
        if complete:
            source_hashes.add(entry["source_sha256"])
        if problem:
            warnings.append(f"Candidate {number:04d} not counted as complete: {problem}.")
        if number in finishes and not complete:
            warnings.append(f"Candidate {number:04d} has a ledger finish but no compact final-state evidence to check.")
        row = dict(entry, source_parent_candidate=commit_numbers.get(entry["parent_commit"]),
                   completed=complete, status=record["status"] if complete else "pending",
                   correctness=record["correctness"] if complete else None,
                   fullgate=record.get("fullgate", False) if complete else None,
                   score_seconds=record.get("score_seconds") if complete else None,
                   score_sample_set=("paired_confirmation_candidate_median" if complete and record.get("confirmation")
                                     else "preliminary_median" if complete and record.get("score_seconds") is not None else None),
                   benchmark_parent=record.get("benchmark_parent") if complete else None,
                   samples=record.get("samples", []) if complete else [],
                   confirmation=record.get("confirmation") if complete else None,
                   reason=record.get("reason", "") if complete else (problem or "no final result yet"),
                   attempts=record.get("attempts", []) if complete else [],
                   started_unix=record.get("started_unix") if complete else None,
                   finished_unix=record.get("finished_unix") if complete else None,
                   wall_seconds=(record["finished_unix"]-record["started_unix"]
                                 if complete and positive(record.get("started_unix"))
                                 and record["finished_unix"] >= record["started_unix"] else None),
                   campaign_identity_sha256=record.get("campaign_identity_sha256") if complete else None,
                   ledger_sealed=number in seals, ledger_finished=number in finishes,
                   worker_completed=number in progress.get("completed", []),
                   worktree=starts.get(number, {}).get("worktree", str(root / f"worktrees/{number:04d}")),
                   links=dict(worktree=f"worktrees/{number:04d}/", source=f"sources/{number:04d}.cu",
                              patch=f"artifacts/{number:04d}/candidate.patch", result=result_path if record is not None else None))
        if complete and number in finishes:
            finish = finishes[number]
            if (finish.get("status") != row["status"] or finish.get("correctness") != row["correctness"]
                    or finish.get("e2e_latency_seconds") != row["score_seconds"]):
                warnings.append(f"Candidate {number:04d}: ledger differs from the compact final state; this report keeps the compact value.")
        rows.append(row)
    completed = [row for row in rows if row["completed"]]
    completed_ids = [row["candidate"] for row in completed]
    eligible = [row for row in completed if row["status"] == "keep" and row["correctness"] is True
                and row["fullgate"] is True and (row["confirmation"] or {}).get("passed") is True]
    best = min(eligible, key=lambda row: (row["score_seconds"], row["candidate"])) if eligible else None
    if progress and progress.get("completed", []) != completed_ids:
        warnings.append("Compact final states differ from the worker progress list; archiving or sync may still be running. This report counts only valid final states.")
    if status.get("completed_rounds", 0) != len(completed):
        warnings.append("The completed count in status.json differs from the compact final states; status may not be synced yet.")
    missing = set(progress.get("completed", [])) - set(numbers)
    if missing:
        warnings.append(f"worker progress has candidates outside the plan: {sorted(missing)}.")
    counts = Counter(row["status"] for row in completed)
    durations = [row["wall_seconds"] for row in completed if positive(row["wall_seconds"])]
    baseline_samples = progress.get("baseline_samples", [])
    baseline_score = progress.get("baseline_score")
    baseline_source = "worker progress" if positive(baseline_score) else None
    if (not positive(baseline_score) and preflight.get("passed") is True
            and len(preflight.get("samples", [])) == 3 and all(map(positive, preflight["samples"]))
            and positive(preflight.get("score_seconds"))
            and math.isclose(preflight["score_seconds"], statistics.median(preflight["samples"]), rel_tol=1e-10)):
        baseline_score, baseline_samples = preflight["score_seconds"], preflight["samples"]
        baseline_source = "preflight baseline; not a completed search round"
    return dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                score="median cold-process fixed-prompt seed-7 E2E seconds",
                timing_boundary="Fresh native process start through complete MP4 and process exit; correctness/media checks excluded.",
                file_cache_policy="OS/filesystem pages are not forcibly cleared; no process/model/CUDA state survives samples.",
                counting_rule="Only identity-matched compact terminal results with actual build/run attempts; no seal, preflight, infrastructure error or status counter inferred as completion.",
                summary=dict(completed=len(completed), budget=budget, planned=len(plan), remaining=max(0,budget-len(completed)),
                             keep=counts["keep"], discard=counts["discard"], crash=counts["crash"],
                             correctness_pass=sum(row["correctness"] is True for row in completed),
                             correctness_fail=sum(row["correctness"] is False for row in completed),
                             sealed=len(seals), ledger_completed=len(finishes),
                             best_confirmed=({key:best[key] for key in ("candidate","commit","score_seconds","fullgate","confirmation","links")}
                                             if best else None)),
                baseline=dict(commit=baseline_commit, score_seconds=baseline_score,
                              samples=baseline_samples, timing_source=baseline_source,
                              fullgate_passed=calibration.get("passed"),
                              fullgate_wall_seconds=calibration.get("wall_seconds_including_checks_and_io")),
                campaign_timing=dict(observed_rounds=len(durations), sum_round_seconds=sum(durations),
                                     median_round_seconds=statistics.median(durations) if durations else None,
                                     rough_remaining_seconds=statistics.median(durations)*(budget-len(completed)) if durations else None,
                                     estimate_note="Observed round median × unfinished rounds; excludes queue/infra delay and unknown future gate mix; not a promised completion time."),
                worker={key:(progress or status.get("remote", {})).get(key) for key in ("status","active","phase","incumbent","incumbent_score","completed","error")},
                reported_status=status, protocol=protocol, rounds=rows, warnings=warnings,
                provenance=dict(inputs=inputs, journal_events=len(events),
                                journal_last_hash=events[-1]["event_hash"] if events else None))


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    def link(path, label, title=None):
        if not path:
            return esc(label)
        return f'<a href="{esc(quote(path, safe="/#"))}"' + (f' title="{esc(title)}"' if title else '') + f'>{esc(label)}</a>'
    def seconds(value):
        return f"{value:.3f}" if positive(value) else "—"
    def parent(number):
        return "baseline" if number == 0 else (link(f"#round-{number:04d}", f"{number:04d}") if number is not None else "—")
    summary = report["summary"]
    best = summary["best_confirmed"]
    best_text = (f'Candidate {best["candidate"]:04d} · {seconds(best["score_seconds"])} s'
                 if best else "No candidate has passed the full gate and paired confirmation yet")
    warning_html = ("<aside><strong>Data checks</strong><ul>" + "".join(f"<li>{esc(w)}</li>" for w in report["warnings"]) + "</ul></aside>"
                    if report["warnings"] else "")
    cards = [("Completed", f'{summary["completed"]} / {summary["budget"]}'),
             ("Correctness PASS / FAIL", f'{summary["correctness_pass"]} / {summary["correctness_fail"]}'),
             ("keep / discard / crash", f'{summary["keep"]} / {summary["discard"]} / {summary["crash"]}')]
    card_html = "".join(f"<div class=card><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)
    rows = []
    for row in report["rounds"]:
        number = row["candidate"]
        state = row["status"]
        active = state == "pending" and report["worker"]["active"] == number
        correct = "PASS" if row["correctness"] is True else "FAIL" if row["correctness"] is False else "—"
        if row["correctness"] is True:
            correct += " · fullgate" if row["fullgate"] else " · screen"
        samples = " / ".join(f"{sample:.6f}" for sample in row["samples"]) or "—"
        confirmation = row["confirmation"] or {}
        pairs = confirmation.get("pairs", [])
        pair_text = "; ".join(f'{p["parent"]:.6f} → {p["candidate"]:.6f}' for p in pairs)
        sample_label = "confirmed median" if pairs else "initial median" if row["score_seconds"] is not None else "not timed"
        details = (f'<details><summary>Raw samples and configuration</summary><p>3 initial runs (s): {esc(samples)}</p>'
                   + (f'<p>5 parent → candidate pairs (s): {esc(pair_text)}</p>' if pairs else '')
                   + f'<pre>{esc(json.dumps(row["config"], ensure_ascii=False, indent=2))}</pre>'
                   + f'<p>Source SHA-256: <code>{esc(row["source_sha256"])}</code></p>'
                   + f'<p>source parent: {esc(row["parent"])}; parent commit: <code>{esc(row["parent_commit"])}</code></p>'
                   + f'<p>started_unix: {esc(row["started_unix"])}; finished_unix: {esc(row["finished_unix"])}</p>'
                   + f'<p>Final state synced to ledger: {"yes" if row["ledger_finished"] else "no"}; worker progress: {"yes" if row["worker_completed"] else "no"}</p></details>')
        local_links = " · ".join(link(row["links"][key], label) for key, label in
                                (("worktree", "worktree"), ("source", "source"), ("patch", "patch"), ("result", "result"))
                                if row["links"][key])
        rows.append(f'<tr id="round-{number:04d}" data-status="{state}"><td><strong>{number:04d}</strong><br><small>{esc(", ".join(row["families"]))}</small></td>'
                    f'<td>{parent(row["source_parent_candidate"])} → {number:04d}<br>{link(row["links"]["worktree"], row["commit"][:12], row["commit"])}</td>'
                    f'<td>{parent(row["benchmark_parent"])}</td><td><span class="badge {state}">{state if state != "pending" else "running" if active else "pending"}</span>'
                    + (f'<br><small>{esc(report["worker"]["phase"])}</small>' if active else '') + '</td>'
                    f'<td>{correct}</td><td class=num>{seconds(row["score_seconds"])}<br><small>{sample_label}</small></td>'
                    f'<td class=num>{seconds(row["wall_seconds"])}</td>'
                    f'<td>{esc(row["reason"])}<br>{local_links}{details}</td></tr>')
    worker = report["worker"]
    worker_text = (f'{worker["status"]} · active={worker["active"]} · phase={worker["phase"]}'
                   if worker["status"] else "no worker progress yet")
    baseline = report["baseline"]
    baseline_text = (f'Baseline: {seconds(baseline["score_seconds"])} s; raw samples: ' +
                     " / ".join(f"{sample:.6f}" for sample in baseline["samples"]) + ". Neither the baseline nor the preflight counts as a search round.") if positive(baseline["score_seconds"]) else "The baseline score record is not synced yet."
    if positive(baseline["fullgate_wall_seconds"]):
        baseline_text += f' Full baseline gate: {"PASS" if baseline["fullgate_passed"] else "FAIL"}, validation took {seconds(baseline["fullgate_wall_seconds"])} s; this is validation cost.'
    timing = report["campaign_timing"]
    timing_text = (f'Observed {timing["observed_rounds"]} rounds, median {seconds(timing["median_round_seconds"])} s, total {seconds(timing["sum_round_seconds"])} s.'
                   f' Rough remaining time at this median: {timing["rough_remaining_seconds"]/3600:.2f} h. The share of full gates, queueing and infrastructure failures change this; it is not a promised finish time.'
                   if timing["observed_rounds"] else 'No completed round timed yet; no estimate of the remaining time.')
    empty_text = '<p class=empty>0 rounds completed. Committed sources and preflight checks do not mean the search is done and do not prove a speedup.</p>' if not summary["completed"] else ''
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 H200 · 100-round search</title><style>
:root{{color-scheme:light;--ink:#193044;--muted:#526575;--line:#dbe3e9}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fa;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:30px 24px}}h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:20px;margin:26px 0 8px}}p{{margin:8px 0}}a{{color:#0759a8}}small,.muted{{color:var(--muted)}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}}.card{{background:white;border:1px solid var(--line);border-radius:9px;padding:16px 22px;min-width:240px}}.card span{{display:block;color:var(--muted)}}.card strong{{display:block;font-size:28px;margin-top:4px}}.best{{background:#eaf2fa;border-left:4px solid #3478b6;padding:14px 18px}}.empty,aside{{padding:14px 18px;background:#fff3d7;border-radius:6px}}.tablewrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;min-width:1180px}}th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#edf2f7;font-weight:600}}td:last-child{{max-width:470px}}.num{{white-space:nowrap;font-variant-numeric:tabular-nums}}code,pre{{font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap}}.badge{{display:inline-block;padding:2px 7px;border-radius:4px;background:#edf0f3;white-space:nowrap}}.keep{{background:#daf1e1;color:#1d6538}}.discard{{background:#fff0cd;color:#775614}}.crash{{background:#fde3e1;color:#9b2720}}details{{margin-top:8px}}summary{{cursor:pointer;color:#0759a8}}select{{padding:6px;margin:12px 0}}tr:target{{background:#eff8ff}}footer{{color:var(--muted);margin-top:24px;font-size:13px}}[hidden]{{display:none!important}}
</style></head><body><main>
<h1>H3 · automatic search on one H200</h1>
<p>864 × 480 × 56 frames · 20 steps · 50 layers · fixed prompt / seed 7</p>
<p class=muted>Each run starts a new process and measures from start until the complete MP4 is written and the process exits. The only performance score is cold-start end-to-end seconds, lower is better; validation is not counted. The shared system's page cache is not dropped.</p>
<div class=cards>{card_html}</div>{empty_text}
<div class=best><strong>Best confirmed candidate</strong><p>{esc(best_text)}</p><small>Requires correctness PASS, fullgate PASS, confirmation PASS and status keep.</small></div>
<p>{esc(baseline_text)}</p><p class=muted>worker: {esc(worker_text)}; sealed sources: {summary["sealed"]}; rounds settled in the ledger: {summary["ledger_completed"]}.</p>
<p class=muted>{esc(timing_text)}</p>
{warning_html}
<h2>Results per round and source lineage</h2>
<p>"Source parent → candidate" shows the Git source lineage; "measured benchmark_parent" shows the incumbent when the round was timed. The two are independent. Candidates that have not run have no measured parent.</p>
<p class=muted>"Round time" is finished_unix − started_unix: build, validation and all timing runs, without archiving after the final state. It estimates search progress; it is not a performance score.</p>
<label>Show <select id=filter><option value=all>all candidates</option><option value=complete>completed</option><option value=keep>keep</option><option value=discard>discard</option><option value=crash>crash</option><option value=pending>pending</option></select></label>
<div class=tablewrap><table><thead><tr><th>Round / change family</th><th>Source parent → candidate<br>source commit</th><th>Measured<br>benchmark_parent</th><th>Verdict</th><th>Correctness</th><th>Cold-start E2E s</th><th>Round time<br>s</th><th>Reason / local evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<footer>Generated: {esc(report["generated_at"])}. Summarizes local evidence only; no GPU runs and nothing published.{link("report.json", "full JSON")} · {link("plan.json", "plan")} · {link("status.json", "status")} · {link("events.jsonl", "events")} · {link("evidence/search/", "search results")}</footer>
</main><script>document.getElementById('filter').addEventListener('change',function(){{document.querySelectorAll('tbody tr').forEach(row=>{{row.hidden=this.value!=='all'&&(this.value==='complete'?row.dataset.status==='pending':row.dataset.status!==this.value);}});}});</script></body></html>'''


def write_report(root):
    report = collect(root)
    root = Path(root).resolve()
    for name, text in (("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                       ("report.html", render(report))):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=f".{name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        try:
            temporary.replace(root / name)
        finally:
            temporary.unlink(missing_ok=True)
    return report


def self_test():
    """Exercise completion accounting without touching a real campaign."""
    with tempfile.TemporaryDirectory(prefix="h3-report-test-") as folder:
        root = Path(folder)
        def save(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + "\n")
        plan = []
        for number in range(1, 4):
            source = root / f"sources/{number:04d}.cu"
            source.parent.mkdir(exist_ok=True)
            source.write_text(f"int variant = {number};\n")
            plan.append(dict(candidate=number, id=f"candidate-{number:03d}", parent="baseline",
                             commit=f"commit{number}", parent_commit="seed", source_sha256=sha256(source.read_bytes()),
                             config={"variant":number}, families=["test"]))
        save("plan.json", plan)
        save("status.json", dict(max_rounds=100, completed_rounds=3, baseline_commit="seed"))
        events = []
        for fields in (dict(kind="init", baseline_commit="seed"), dict(kind="seal", candidate=1),
                       dict(kind="finish", candidate=3),
                       dict(kind="attempt", candidate=1, stage="run", outcome="success", note="preflight only")):
            event = dict(seq=len(events)+1, prev_hash=events[-1]["event_hash"] if events else "0"*64, **fields)
            event["event_hash"] = sha256(encoded(event))
            events.append(event)
        (root / "events.jsonl").write_bytes(b"".join(encoded(event)+b"\n" for event in events))
        report = collect(root)
        assert report["summary"]["completed"] == 0 and report["summary"]["best_confirmed"] is None
        assert "0 rounds completed" in render(report)
        good = dict(plan[0], terminal=True, status="keep", correctness=True, fullgate=True,
                    benchmark_parent=0, score_seconds=9, samples=[9,9,9], started_unix=10, finished_unix=100,
                    attempts=[dict(stage="build", outcome="success"),dict(stage="run", outcome="success")],
                    confirmation=dict(passed=True,pairs=[dict(parent=10,candidate=9)]*5), reason="<script>unsafe</script>")
        save("evidence/search/0001/result.json", good)
        # Nonterminal/infrastructure records never complete a round.
        save("evidence/search/0002/result.json", dict(plan[1], terminal=False, status="infrastructure_error"))
        # A foreign source identity cannot borrow successful results.
        save("evidence/search/0003/result.json", dict(good, **{k:v for k,v in plan[2].items() if k!='source_sha256'}))
        report = write_report(root)
        assert report["summary"]["completed"] == 1 and report["summary"]["best_confirmed"]["candidate"] == 1
        assert report["rounds"][0]["wall_seconds"] == 90 and report["campaign_timing"]["rough_remaining_seconds"] == 90*99
        assert report["rounds"][1]["benchmark_parent"] is None
        assert "&lt;script&gt;unsafe&lt;/script&gt;" in (root / "report.html").read_text()
        # Keep requires both fullgate and confirmation, with actual raw timing.
        save("evidence/search/0001/result.json", dict(good, fullgate=False))
        assert collect(root)["summary"]["completed"] == 0
        save("evidence/search/0001/result.json", dict(good, score_seconds=1))
        assert collect(root)["summary"]["completed"] == 0
        save("evidence/search/0001/result.json", dict(good, confirmation=dict(passed=True,pairs=None)))
        assert collect(root)["summary"]["completed"] == 0
        # Numerical failure and a real compile crash are valid completed trials.
        save("evidence/search/0001/result.json", dict(good, status="discard",correctness=False,fullgate=False,confirmation=None,score_seconds=None,samples=[]))
        crash = dict(plan[1],terminal=True,status="crash",correctness=False,fullgate=False,benchmark_parent=0,
                     finished_unix=100,score_seconds=None,attempts=[dict(stage="build",outcome="candidate_error")])
        save("evidence/search/0002/result.json", crash)
        report = collect(root)
        assert report["summary"]["completed"] == 2 and report["summary"]["crash"] == 1
        assert report["summary"]["correctness_fail"] == 2 and report["summary"]["best_confirmed"] is None
        # Source ancestry is independent of the measured incumbent.
        plan[1].update(parent="candidate-001",parent_commit="commit1")
        crash.update(parent="candidate-001",parent_commit="commit1")
        save("plan.json",plan);save("evidence/search/0002/result.json",crash)
        report=collect(root)
        assert report["rounds"][1]["source_parent_candidate"] == 1 and report["rounds"][1]["benchmark_parent"] == 0
    print("report self-test: completion, source identity, raw timing, fullgate, crash and HTML escaping passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        report = write_report(args.root)
        print(json.dumps(dict(root=report["root"], **report["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
