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
        warnings.append(f"{relative} 尚未完整或不是有效 JSON；本次不使用。")
        return None


def read_events(root, inputs, warnings):
    path = root / "events.jsonl"
    if not path.is_file():
        warnings.append("缺少 events.jsonl，无法核对本地 ledger。")
        return []
    raw = path.read_bytes()
    inputs["events.jsonl"] = {"sha256": sha256(raw), "bytes": len(raw)}
    events, previous = [], "0" * 64
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        if not line.endswith(b"\n"):
            warnings.append("events.jsonl 末行未写完；只读取此前完整记录。")
            break
        try:
            event = json.loads(line)
            payload = {k: v for k, v in event.items() if k != "event_hash"}
            if (event["seq"] != index or event["prev_hash"] != previous
                    or event["event_hash"] != sha256(encoded(payload))):
                raise ValueError("hash chain differs")
        except (ValueError, KeyError, TypeError) as error:
            warnings.append(f"events.jsonl 第 {index} 行校验失败；只读取此前有效记录。")
            break
        events.append(event)
        previous = event["event_hash"]
    return events


def terminal_problem(record, entry, progress, root):
    """Fail closed on incomplete/foreign evidence; do not re-run the judge."""
    if not isinstance(record, dict):
        return "result 不是对象"
    if record.get("terminal") is not True or record.get("status") not in TERMINAL:
        return "没有完整终态"
    if any(record.get(key) != value for key, value in entry.items()):
        return "result 身份与 plan 不同"
    if (progress.get("campaign_identity_sha256") and
            record.get("campaign_identity_sha256") != progress["campaign_identity_sha256"]):
        return "campaign identity 与 progress 不同"
    attempts = record.get("attempts", [])
    if not isinstance(attempts, list) or any(not isinstance(a, dict) for a in attempts):
        return "attempts 格式无效"
    real_attempts = [a for a in attempts if isinstance(a, dict)
                     and a.get("stage") in ("build", "run")
                     and a.get("outcome") in ("success", "candidate_error")]
    if not real_attempts or attempts[-1].get("outcome") == "infra_error":
        return "没有真实候选 build/run 结果，或末次是基础设施错误"
    if record["status"] != "crash" and not any(a["stage"] == "run" for a in real_attempts):
        return "非 crash 终态缺少 run 记录"
    if type(record.get("correctness")) is not bool or not positive(record.get("finished_unix")):
        return "缺少明确正确性结果或完成时间"
    parent = record.get("benchmark_parent")
    if type(parent) is not int or parent < 0:
        return "缺少实测 benchmark_parent"
    source = root / f"sources/{entry['candidate']:04d}.cu"
    if not source.is_file() or sha256(source.read_bytes()) != entry["source_sha256"]:
        return "本地候选源码缺失或 hash 不符"
    score = record.get("score_seconds")
    if score is not None and not positive(score):
        return "冷启动秒数无效"
    if score is not None:
        samples = record.get("samples", [])
        if not isinstance(samples, list) or len(samples) != 3 or not all(map(positive, samples)):
            return "缺少三个有效的原始初测样本"
        confirmation = record.get("confirmation")
        if confirmation is not None:
            if not isinstance(confirmation, dict):
                return "confirmation 格式无效"
            pairs = confirmation.get("pairs", [])
            if (not isinstance(pairs, list) or len(pairs) != 5 or not all(isinstance(pair, dict) and
                    positive(pair.get("parent")) and positive(pair.get("candidate")) for pair in pairs)):
                return "配对确认样本不完整"
            expected = statistics.median(pair["candidate"] for pair in pairs)
        else:
            expected = statistics.median(samples)
        if not math.isclose(score, expected, rel_tol=1e-10, abs_tol=1e-10):
            return "分数与原始样本中位数不同"
    if record["status"] == "keep" and not (
            record["correctness"] is True and record.get("fullgate") is True
            and isinstance(record.get("confirmation"), dict)
            and record["confirmation"].get("passed") is True and positive(score)):
        return "keep 缺少 fullgate、confirmation 或有效分数"
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
            problem = "与已有完成候选的源码重复"
        complete = record is not None and problem is None
        if complete and record["benchmark_parent"] not in {0, *numbers}:
            problem, complete = "benchmark_parent 不属于此 plan", False
        if complete:
            source_hashes.add(entry["source_sha256"])
        if problem:
            warnings.append(f"候选 {number:04d} 不计入完成：{problem}。")
        if number in finishes and not complete:
            warnings.append(f"候选 {number:04d} 有 ledger finish，但缺少可核对的 compact 终态证据。")
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
                   reason=record.get("reason", "") if complete else (problem or "尚无正式终态结果"),
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
                warnings.append(f"候选 {number:04d} 的 ledger 与 compact 终态不同；本报告保留 compact 原值。")
        rows.append(row)
    completed = [row for row in rows if row["completed"]]
    completed_ids = [row["candidate"] for row in completed]
    eligible = [row for row in completed if row["status"] == "keep" and row["correctness"] is True
                and row["fullgate"] is True and (row["confirmation"] or {}).get("passed") is True]
    best = min(eligible, key=lambda row: (row["score_seconds"], row["candidate"])) if eligible else None
    if progress and progress.get("completed", []) != completed_ids:
        warnings.append("compact 终态与 worker progress 完成列表不同；可能仍在归档或同步。本报告只计有效终态。")
    if status.get("completed_rounds", 0) != len(completed):
        warnings.append("status.json 的完成计数与 compact 终态不同；status 可能尚未同步。")
    missing = set(progress.get("completed", [])) - set(numbers)
    if missing:
        warnings.append(f"worker progress 含 plan 外的候选：{sorted(missing)}。")
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
    best_text = (f'候选 {best["candidate"]:04d} · {seconds(best["score_seconds"])} 秒'
                 if best else "暂无通过完整 gate 与配对确认的候选")
    warning_html = ("<aside><strong>数据核对</strong><ul>" + "".join(f"<li>{esc(w)}</li>" for w in report["warnings"]) + "</ul></aside>"
                    if report["warnings"] else "")
    cards = [("正式完成", f'{summary["completed"]} / {summary["budget"]}'),
             ("正确性 PASS / FAIL", f'{summary["correctness_pass"]} / {summary["correctness_fail"]}'),
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
        sample_label = "确认中位数" if pairs else "初测中位数" if row["score_seconds"] is not None else "未计时"
        details = (f'<details><summary>原始样本与配置</summary><p>初测 3 次（秒）：{esc(samples)}</p>'
                   + (f'<p>5 对 parent → candidate（秒）：{esc(pair_text)}</p>' if pairs else '')
                   + f'<pre>{esc(json.dumps(row["config"], ensure_ascii=False, indent=2))}</pre>'
                   + f'<p>源码 SHA-256：<code>{esc(row["source_sha256"])}</code></p>'
                   + f'<p>source parent：{esc(row["parent"])}；parent commit：<code>{esc(row["parent_commit"])}</code></p>'
                   + f'<p>started_unix：{esc(row["started_unix"])}；finished_unix：{esc(row["finished_unix"])}</p>'
                   + f'<p>终态已同步到 ledger：{"是" if row["ledger_finished"] else "否"}；worker progress：{"是" if row["worker_completed"] else "否"}</p></details>')
        local_links = " · ".join(link(row["links"][key], label) for key, label in
                                (("worktree", "worktree"), ("source", "源码"), ("patch", "patch"), ("result", "结果"))
                                if row["links"][key])
        rows.append(f'<tr id="round-{number:04d}" data-status="{state}"><td><strong>{number:04d}</strong><br><small>{esc(", ".join(row["families"]))}</small></td>'
                    f'<td>{parent(row["source_parent_candidate"])} → {number:04d}<br>{link(row["links"]["worktree"], row["commit"][:12], row["commit"])}</td>'
                    f'<td>{parent(row["benchmark_parent"])}</td><td><span class="badge {state}">{state if state != "pending" else "运行中" if active else "待完成"}</span>'
                    + (f'<br><small>{esc(report["worker"]["phase"])}</small>' if active else '') + '</td>'
                    f'<td>{correct}</td><td class=num>{seconds(row["score_seconds"])}<br><small>{sample_label}</small></td>'
                    f'<td class=num>{seconds(row["wall_seconds"])}</td>'
                    f'<td>{esc(row["reason"])}<br>{local_links}{details}</td></tr>')
    worker = report["worker"]
    worker_text = (f'{worker["status"]} · active={worker["active"]} · phase={worker["phase"]}'
                   if worker["status"] else "尚无 worker progress")
    baseline = report["baseline"]
    baseline_text = (f'基准：{seconds(baseline["score_seconds"])} 秒；原始样本：' +
                     " / ".join(f"{sample:.6f}" for sample in baseline["samples"]) + "。基准和预检均不计入正式搜索轮次。") if positive(baseline["score_seconds"]) else "基准计分记录尚未同步。"
    if positive(baseline["fullgate_wall_seconds"]):
        baseline_text += f' 完整基准 gate：{"PASS" if baseline["fullgate_passed"] else "FAIL"}，验证耗时 {seconds(baseline["fullgate_wall_seconds"])} 秒；这是验证成本。'
    timing = report["campaign_timing"]
    timing_text = (f'已观测 {timing["observed_rounds"]} 轮执行耗时，中位数 {seconds(timing["median_round_seconds"])} 秒，累计 {seconds(timing["sum_round_seconds"])} 秒。'
                   f' 按此中位数粗估剩余 {timing["rough_remaining_seconds"]/3600:.2f} 小时。后续完整 gate 比例、排队和基础设施中断会改变耗时；此值不是完成时间承诺。'
                   if timing["observed_rounds"] else '尚无完成轮次的执行耗时，暂不估算剩余时间。')
    empty_text = '<p class=empty>目前正式完成 0 轮。已有源码提交及预检不代表搜索完成，也不能证明性能提升。</p>' if not summary["completed"] else ''
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 H200 · 100 轮搜索</title><style>
:root{{color-scheme:light;--ink:#193044;--muted:#526575;--line:#dbe3e9}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fa;color:var(--ink);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:30px 24px}}h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:20px;margin:26px 0 8px}}p{{margin:8px 0}}a{{color:#0759a8}}small,.muted{{color:var(--muted)}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}}.card{{background:white;border:1px solid var(--line);border-radius:9px;padding:16px 22px;min-width:240px}}.card span{{display:block;color:var(--muted)}}.card strong{{display:block;font-size:28px;margin-top:4px}}.best{{background:#eaf2fa;border-left:4px solid #3478b6;padding:14px 18px}}.empty,aside{{padding:14px 18px;background:#fff3d7;border-radius:6px}}.tablewrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;min-width:1180px}}th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#edf2f7;font-weight:600}}td:last-child{{max-width:470px}}.num{{white-space:nowrap;font-variant-numeric:tabular-nums}}code,pre{{font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap}}.badge{{display:inline-block;padding:2px 7px;border-radius:4px;background:#edf0f3;white-space:nowrap}}.keep{{background:#daf1e1;color:#1d6538}}.discard{{background:#fff0cd;color:#775614}}.crash{{background:#fde3e1;color:#9b2720}}details{{margin-top:8px}}summary{{cursor:pointer;color:#0759a8}}select{{padding:6px;margin:12px 0}}tr:target{{background:#eff8ff}}footer{{color:var(--muted);margin-top:24px;font-size:13px}}[hidden]{{display:none!important}}
</style></head><body><main>
<h1>H3 · 单张 H200 自动搜索</h1>
<p>864 × 480 × 56 帧 · 20 步 · 50 层 · 固定 prompt / seed 7</p>
<p class=muted>每次启动新进程，从启动到完整 MP4 输出并退出。唯一性能分数是冷启动端到端秒数，越低越好；验证不计入秒数。共享系统的文件页缓存不强制清空。</p>
<div class=cards>{card_html}</div>{empty_text}
<div class=best><strong>最好已确认候选</strong><p>{esc(best_text)}</p><small>需 correctness PASS、fullgate PASS、confirmation PASS 且状态为 keep。</small></div>
<p>{esc(baseline_text)}</p><p class=muted>worker：{esc(worker_text)}；已封存源码 {summary["sealed"]} 个，ledger 已结算 {summary["ledger_completed"]} 轮。</p>
<p class=muted>{esc(timing_text)}</p>
{warning_html}
<h2>每轮结果与源码关系</h2>
<p>“源码父节点 → 候选”显示 Git 源码继承关系；“实测 benchmark_parent”显示该轮测速时的 incumbent。两者独立。尚未运行的候选不填实测父节点。</p>
<p class=muted>“本轮耗时”是 finished_unix − started_unix，含构建、验证和全部测速，不含终态后归档。它用于估算搜索进度，不是性能分数。</p>
<label>显示 <select id=filter><option value=all>全部候选</option><option value=complete>正式完成</option><option value=keep>keep</option><option value=discard>discard</option><option value=crash>crash</option><option value=pending>待完成</option></select></label>
<div class=tablewrap><table><thead><tr><th>轮次 / 变更族</th><th>源码父节点 → 候选<br>source commit</th><th>实测<br>benchmark_parent</th><th>判定</th><th>正确性</th><th>冷启动 E2E 秒</th><th>本轮耗时<br>秒</th><th>原因 / 本地证据</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<footer>生成时间：{esc(report["generated_at"])}。仅汇总本地已有证据，无 GPU 运行或外部发布。{link("report.json", "完整 JSON")} · {link("plan.json", "plan")} · {link("status.json", "status")} · {link("events.jsonl", "events")} · {link("evidence/search/", "搜索结果目录")}</footer>
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
        assert "目前正式完成 0 轮" in render(report)
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
