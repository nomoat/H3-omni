#!/usr/bin/env python3
"""Generate the saved AutoGrad dashboard spec; never upload or deploy it.

Generate: python3 lake_dashboard.py --output /path/to/dashboard.json
Validate: python3 lake_dashboard.py --self-test --autograd-root /path/to/AutoGrad

The self-test validates the lake's outer JSON schema and compiles/runs every
chart with AutoGrad's own pinned Vega-Lite/Vega browser bundles, including the
renderer hover transformation. No credentials or network calls are used.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess


EXPERIMENT_ID = "h3-h200-cold-search-20261010-100"
DASHBOARD_NAME = "h3-h200-20261010-100"
DEFAULT_AUTOGRAD_ROOT = Path("/home/homo/model/srp/AutoGrad")


def field(name, kind="nominal", **options):
    return {"field": name, "type": kind, **options}


def source(stream, limit=5000):
    return {"stream": stream, "limit": limit}


def chart(title, streams, spec, span=1):
    return {"title": title, "span": span, "data": streams,
            "vega_lite": {"height": 230, **spec}}


def table(title, stream, columns, rows=100, limit=5000, span=2):
    return {"title": title, "span": span, "data": [source(stream, limit)],
            "table": {"columns": columns, "rows": rows}}


def positive(name):
    return f"isValid(datum.{name}) && isNumber(datum.{name}) && isFinite(datum.{name}) && datum.{name} > 0"


def attempt_status():
    """Join completion onto samples/checks without a second dataset or URL.

    Lake panels concatenate their source rows. The result row supplies the
    completion bit; joinaggregate broadcasts it to rows of the same attempt.
    A measured provisional sample is retained, but can never become a score.
    """
    return [
        {"calculate": "datum._stream === 'attempts/results' && datum.counts_as_completed === true ? 1 : 0", "as": "completed_flag"},
        {"joinaggregate": [{"op": "max", "field": "completed_flag", "as": "attempt_completed"}],
         "groupby": ["candidate", "attempt_id"]},
        {"calculate": "datum.attempt_completed === 1 ? 'completed' : 'provisional / interrupted'", "as": "sample_status"},
    ]


def build_spec(experiment_id=EXPERIMENT_ID):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", experiment_id):
        raise ValueError("Invalid experiment ID")
    panels = []
    labels = [
        ("progress_label", "datum.status + ' | ' + format(datum.completed, 'd') + ' / ' + format(datum.total, 'd') + ' completed'", 6, 21),
        ("decision_label", "format(datum.kept, 'd') + ' keep  |  ' + format(datum.discarded, 'd') + ' discard  |  ' + format(datum.crashed, 'd') + ' crash'", 49, 15),
        ("best_label", "isValid(datum.best_candidate) && datum.best_candidate > 0 && isValid(datum.best_seconds) && datum.best_seconds > 0 ? 'Best confirmed: #' + datum.best_candidate + ' | ' + format(datum.best_seconds, '.3f') + ' s' : 'Best confirmed: none'", 88, 17),
        ("baseline_label", f"{positive('baseline_seconds')} ? 'Original seed baseline: ' + format(datum.baseline_seconds, '.3f') + ' s' : 'Original seed baseline: unavailable'", 125, 14),
        ("active_label", "isValid(datum.active) && datum.active > 0 ? 'Active candidate: #' + datum.active : 'Active candidate: none'", 158, 14),
        ("updated_label", "'Updated: ' + datum.time_utc", 191, 12),
    ]
    panels.append(chart("Overview / Progress", [source("search/overview", 1)], {
        "transform": [{"calculate": expression, "as": name} for name, expression, _, _ in labels],
        "layer": [{"mark": {"type": "text", "align": "left", "baseline": "top", "fontSize": size},
                   "encoding": {"x": {"value": 0}, "y": {"value": y}, "text": field(name)}}
                  for name, _, y, size in labels],
        "config": {"view": {"stroke": None}},
    }))
    panels.append(chart("Overview / Cold E2E score — completed attempts only", [
        source("attempts/results"), source("search/overview", 1)], {
        "layer": [
            {"transform": [
                {"filter": "datum._stream === 'attempts/results' && datum.counts_as_completed === true && " + positive("score_seconds")},
                {"window": [{"op": "row_number", "as": "latest"}], "groupby": ["candidate", "attempt_id"], "sort": [{"field": "_ts", "order": "descending"}]},
                {"filter": "datum.latest === 1"}],
             "mark": {"type": "point", "filled": True, "size": 100},
             "encoding": {
                 "x": field("candidate", "quantitative", title="Candidate", axis={"tickMinStep": 1}),
                 "y": field("score_seconds", "quantitative", title="Cold E2E seconds · lower is better", scale={"zero": False}),
                 "color": field("decision", scale={"domain": ["keep", "discard", "crash"], "range": ["#16834a", "#c88215", "#c84136"]}),
                 "tooltip": [field("candidate"), field("attempt_id"), field("decision"), field("score_seconds", "quantitative", format=".6f"), field("correctness_screen_passed"), field("correctness_full_passed"), field("benchmark_parent"), field("reason")]}},
            {"transform": [{"filter": "datum._stream === 'search/overview' && " + positive("baseline_seconds")}],
             "mark": {"type": "rule", "color": "#738395", "strokeDash": [5, 4]},
             "encoding": {"y": field("baseline_seconds", "quantitative"),
                          "tooltip": [field("baseline_seconds", "quantitative", title="Original seed baseline", format=".6f")]}},
        ],
    }))
    panels.append(chart("Samples / Raw cold measurements — includes provisional interrupted attempts", [
        source("metrics/cold_samples", 20000), source("attempts/results")], {
        "height": 285,
        "params": [{"name": "sample_status_filter", "value": "all", "bind": {
            "input": "select", "name": "Attempt status ",
            "options": ["all", "completed", "provisional / interrupted"]}}],
        "transform": attempt_status() + [
            {"filter": "datum._stream === 'metrics/cold_samples' && " + positive("seconds")},
            {"filter": "sample_status_filter === 'all' || datum.sample_status === sample_status_filter"}],
        "mark": {"type": "point", "filled": True, "size": 65, "opacity": .75},
        "encoding": {
            "x": field("candidate", "quantitative", title="Candidate", axis={"tickMinStep": 1}),
            "y": field("seconds", "quantitative", title="One complete cold request · seconds", scale={"zero": False}),
            "color": field("sample_status", title="Attempt status", scale={"domain": ["completed", "provisional / interrupted"], "range": ["#2677b8", "#dc8c1f"]}),
            "shape": field("role", title="Measured role"),
            "tooltip": [field("candidate"), field("attempt_id"), field("sample_status"), field("phase"), field("role"), field("index"), field("pair"), field("seconds", "quantitative", format=".6f"), field("mp4_sha256")],
        },
    }, span=2))
    panels.append(chart("Correctness / Failed numerical checks — latest 20; package retains all", [
        source("metrics/numerical_checks", 100000), source("attempts/results")], {
        "height": 300,
        "params": [{"name": "failed_case_filter", "value": "", "bind": {
            "input": "text", "name": "Case contains "}}],
        "transform": attempt_status() + [
            {"calculate": "datum._stream === 'metrics/numerical_checks' && datum.passed === false && (failed_case_filter === '' || indexof(lower(toString(datum.case)), lower(failed_case_filter)) >= 0) ? 1 : 0", "as": "failed_check"},
            {"joinaggregate": [{"op": "sum", "field": "failed_check", "as": "failure_count"}]},
        ],
        "layer": [
            {"transform": [{"filter": "datum.failure_count === 0"},
                           {"aggregate": [{"op": "max", "field": "failure_count", "as": "n"}]}],
             "mark": {"type": "text", "align": "left", "baseline": "top", "fontSize": 15, "color": "#16834a"},
             "encoding": {"x": {"value": 0}, "y": {"value": 10}, "text": {"value": "No failed checks match the loaded evidence and case filter; see verification coverage below."}}},
            {"transform": [
                {"filter": "datum.failed_check === 1"},
                {"window": [{"op": "row_number", "as": "failure_rank"}], "sort": [{"field": "_ts", "order": "descending"}]},
                {"filter": "datum.failure_rank <= 20"},
                {"calculate": "'#' + datum.candidate + ' | ' + datum.gate + ' | ' + datum.case + ' | ' + datum.label + ' | ' + datum.sample_status", "as": "failure_label"}],
             "mark": {"type": "text", "align": "left", "fontSize": 11, "color": "#b7352b"},
             "encoding": {"x": {"value": 0}, "y": field("failure_rank", "ordinal", axis=None),
                          "text": field("failure_label"),
                          "tooltip": [field("candidate"), field("attempt_id"), field("gate"), field("case"), field("label"), field("sample_status"), field("relative_l2", "quantitative"), field("normalized_max", "quantitative"), field("details")]}}
        ],
    }, span=2))
    panels.append(table("Attempts / Decisions and measured incumbent — unfinished rows do not count", "attempts/results", [
        "candidate", "attempt_id", "state", "decision", "counts_as_completed", "correctness_screen_passed",
        "correctness_full_passed", "score_seconds", "round_seconds", "benchmark_parent", "commit", "reason"], rows=200))
    panels.append(table("Correctness / Verification gates and coverage", "metrics/verification", [
        "candidate", "attempt_id", "gate", "passed", "scope", "comparison_count", "seconds"], rows=200))
    panels.append(table("Samples / Cold request records — provisional samples are not scores", "metrics/cold_samples", [
        "candidate", "attempt_id", "phase", "role", "index", "pair", "seconds", "mp4_sha256", "command"], rows=200, limit=20000))
    panels.append(table("Samples / Promotion pairs — parent and candidate", "metrics/promotion_pairs", [
        "candidate", "attempt_id", "index", "parent_seconds", "candidate_seconds", "difference_seconds"], rows=100))
    panels.append(table("Git / Source ancestry — planned commits are not completed trials", "search/candidates", [
        "candidate", "parent", "parent_commit", "commit", "source_sha256", "patch_sha256", "hypothesis", "config"], rows=100, limit=1000))
    panels.append(table("Attempts / Lifecycle and interrupted phases", "attempts/lifecycle", [
        "candidate", "attempt_id", "state", "phase", "source_time_utc"], rows=200, limit=20000))
    panels.append(chart("Packages / Open evidence archive — Google sign-in; latest 20 or select a candidate", [source("packages/catalog", 1000)], {
        "height": 320,
        "params": [{"name": "package_candidate", "value": -1,
                    "bind": {"input": "select", "name": "Candidate ",
                             "options": list(range(-1, 101)),
                             "labels": ["Latest 20", "Baseline"] + [str(n) for n in range(1, 101)]}}],
        "transform": [
            {"filter": "isValid(datum.download_url) && indexof(datum.download_url, 'https://storage.cloud.google.com/') === 0"},
            {"filter": "package_candidate < 0 || datum.candidate === package_candidate"},
            {"window": [{"op": "row_number", "as": "package_rank"}], "sort": [{"field": "_ts", "order": "descending"}]},
            {"filter": "datum.package_rank <= 20"},
            {"calculate": "'#' + datum.candidate + ' | ' + datum.attempt_id + ' | ' + datum.state + ' | ' + (datum.complete === true ? 'complete package' : 'partial package') + ' | Open archive'", "as": "archive_label"},
        ],
        "mark": {"type": "text", "align": "left", "fontSize": 12, "color": "#2677b8", "cursor": "pointer"},
        "encoding": {"x": {"value": 0}, "y": field("package_id", "ordinal", axis=None),
                     "text": field("archive_label"), "href": field("download_url"),
                     "tooltip": [field("candidate"), field("attempt_id"), field("state"), field("complete"), field("artifact_count"), field("total_bytes"), field("archive_sha256"), field("download_url")]},
    }, span=2))
    panels.append(table("Packages / Archive manifests and integrity", "packages/catalog", [
        "candidate", "attempt_id", "package_id", "state", "complete", "artifact_count", "total_bytes", "manifest_sha256", "archive_sha256", "storage_uri", "download_url"], rows=200, limit=1000))
    panels.append(table("Packages / File catalog — metadata only; raw bodies stay in the archive", "artifacts/catalog", [
        "candidate", "attempt_id", "path", "sha256", "bytes", "media_type", "package_id", "access_status", "storage_uri", "download_url"], rows=200, limit=20000))
    panels.append(table("Collection / Available and missing evidence", "collection/coverage", ["field", "status", "reason"], rows=100, limit=1000))
    return {
        "version": 1, "title": "H3 on H200 — 100 cold-start search trials",
        "description": "1 H200 · 864×480×56 · 20 steps · 50 layers · fixed prompt and seed 7. Only completed attempts enter the cold E2E score plot. Orange raw points are valid measurements from provisional/interrupted attempts, not final scores. Git source parents and measured benchmark parents are independent. Evidence collection runs outside the frozen benchmark process.",
        "experiment": experiment_id, "columns": 2, "refresh_seconds": 30,
        "panels": panels,
    }


NODE_CHECK = r"""
const fs = require('fs'), vm = require('vm');
async function main() {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const context = { console, setTimeout, clearTimeout, structuredClone };
  vm.createContext(context);
  for (const file of ['vega.min.js', 'vega-lite.min.js', 'dashboard.js']) {
    vm.runInContext(fs.readFileSync(input.bundle_dir + '/' + file, 'utf8'), context);
  }
  const rows = input.rows;
  let charts = 0;
  function leaves(view, type) {
    const result = [];
    function walk(node) {
      if (node.mark && node.mark.marktype === type && node.datum) result.push(node.datum);
      for (const child of node.items || []) walk(child);
    }
    walk(view.scenegraph().root);
    return result;
  }
  for (const panel of input.spec.panels) {
    if (!panel.vega_lite) continue;
    const data = panel.data.flatMap(source => (rows[source.stream] || []).slice(-source.limit).map(row => ({
      _stream: source.stream, _experiment: input.spec.experiment, _ts: '2026-10-10T00:00:00Z', ...row
    })));
    const chart = context.dashboardHoverDefaults(structuredClone(panel.vega_lite));
    const warnings = [];
    const logger = { warn: (...args) => warnings.push(args.join(' ')), info() {}, debug() {}, error: (...args) => { throw Error(args.join(' ')); } };
    const compiled = context.vegaLite.compile({
      width: 800, height: 230, ...chart, data: {name:'lake'}, datasets:{lake:data}
    }, {logger}).spec;
    if (warnings.length) throw Error(panel.title + ': ' + warnings.join('; '));
    const view = new context.vega.View(context.vega.parse(compiled, {}, {ast:true}), {renderer:'none'});
    await view.runAsync();
    if (panel.title.includes('Cold E2E score')) {
      const plotted = leaves(view, 'symbol').filter(d=>d._stream==='attempts/results').map(d => d.candidate).sort();
      if (JSON.stringify(plotted) !== '[1,2]') throw Error('Final score includes interrupted/zero/unknown data: '+JSON.stringify(plotted));
    }
    if (panel.title.includes('Raw cold measurements')) {
      const plotted = leaves(view, 'symbol').filter(d=>d._stream==='metrics/cold_samples').sort((a,b)=>a.candidate-b.candidate);
      if (JSON.stringify(plotted.map(d=>d.candidate)) !== '[1,2,3]') throw Error('Valid provisional samples missing or invalid sample retained');
      if (plotted[2].sample_status !== 'provisional / interrupted' || plotted[0].sample_status !== 'completed') throw Error('Raw sample status join is incorrect');
      await view.signal('sample_status_filter','completed').runAsync();
      if (leaves(view,'symbol').filter(d=>d._stream==='metrics/cold_samples').length !== 2) throw Error('Interactive raw-status filter failed');
      await view.signal('sample_status_filter','all').runAsync();
    }
    if (panel.title.includes('Failed numerical checks')) {
      const failed = leaves(view, 'text').filter(d=>d.failed_check===1);
      if (failed.length !== 1 || failed[0].label !== 'actual failure') throw Error('Failure filter includes PASS checks or drops a real failure');
      await view.signal('failed_case_filter','missing-case').runAsync();
      if (leaves(view,'text').some(d=>d.failed_check===1)) throw Error('Interactive failed-case filter failed');
      await view.signal('failed_case_filter','').runAsync();
    }
    if (panel.title.includes('Open evidence archive')) {
      const links = leaves(view, 'text').filter(d=>d.download_url);
      if (links.length !== 1 || !links[0].archive_label.includes('partial package')) throw Error('Archive link or partial-package label is missing');
    }
    // The charts must also accept an empty stream and a live refresh.
    await view.change('lake', context.vega.changeset().remove(context.vega.truthy)).runAsync();
    await view.change('lake', context.vega.changeset().insert(data)).runAsync();
    view.finalize(); charts++;
  }
  console.log(JSON.stringify({vega:context.vega.version,vega_lite:context.vegaLite.version,compiled_and_executed_charts:charts,semantic_checks:'passed'}));
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});
"""


def self_test(autograd_root=DEFAULT_AUTOGRAD_ROOT):
    import jsonschema
    root = Path(autograd_root)
    spec = build_spec()
    schema = json.loads((root / "services/data-lake/schema/dashboard-v1.json").read_text())
    # The outer schema points to an online Vega-Lite schema. Keep validation
    # offline: validate its container here, then compile and execute each chart
    # with the exact Vega-Lite version shipped by the live renderer below.
    schema = copy.deepcopy(schema)
    schema["$defs"]["panel"]["properties"]["vega_lite"] = {"type": "object"}
    jsonschema.Draft202012Validator(schema).validate(spec)
    assert spec["experiment"] == EXPERIMENT_ID and spec["columns"] == 2
    assert len(json.dumps(spec).encode()) < 1_000_000
    assert not any("raw" in item["stream"] for panel in spec["panels"] for item in panel["data"])
    def check_references(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name == "data":
                    assert child == {"name": "lake"}, "Embedded/external chart data is forbidden"
                check_references(child)
        elif isinstance(value, list):
            for child in value:
                check_references(child)
    for panel in spec["panels"]:
        if "vega_lite" in panel:
            check_references(panel["vega_lite"])
    rows = {
        "search/overview": [{"status":"paused", "completed":2, "total":100, "remaining":98, "kept":1, "discarded":1, "crashed":0, "active":None, "best_candidate":2, "best_seconds":79, "baseline_seconds":86.4, "time_utc":"2026-10-10T00:00:00Z"}],
        "attempts/results": [
            {"candidate":1,"attempt_id":"a1","state":"finished","decision":"discard","counts_as_completed":True,"score_seconds":80},
            {"candidate":2,"attempt_id":"a2","state":"finished","decision":"keep","counts_as_completed":True,"score_seconds":79},
            {"candidate":3,"attempt_id":"a3","state":"interrupted","decision":None,"counts_as_completed":False,"score_seconds":60},
            {"candidate":4,"attempt_id":"a4","state":"finished","decision":"crash","counts_as_completed":True,"score_seconds":0},
            {"candidate":5,"attempt_id":"a5","state":"planned","decision":None,"counts_as_completed":False,"score_seconds":None},
        ],
        "metrics/cold_samples": [
            {"candidate":n,"attempt_id":f"a{n}","phase":"preliminary","role":"candidate","index":0,"seconds":seconds}
            for n,seconds in ((1,80),(2,79),(3,60),(4,0),(5,None))],
        "metrics/numerical_checks": [
            {"candidate":1,"attempt_id":"a1","gate":"screen","case":"small","label":"passing check","passed":True},
            {"candidate":3,"attempt_id":"a3","gate":"screen","case":"small","label":"actual failure","passed":False,"relative_l2":.01},
        ],
        "packages/catalog": [{"candidate":3,"attempt_id":"a3","package_id":"pkg3","state":"interrupted","complete":False,"download_url":"https://storage.cloud.google.com/cj-dev-498907-autograd-training-lake/artifacts/h3-test/a3.tar"}],
    }
    result = subprocess.run(["node", "-e", NODE_CHECK], input=json.dumps({"spec":spec, "rows":rows, "bundle_dir":str(root / "apps/web/public/monitor")}),
                            text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Dashboard Vega validation failed: " + result.stderr.strip())
    print("Outer dashboard schema: PASS")
    print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--autograd-root", type=Path, default=DEFAULT_AUTOGRAD_ROOT)
    args = parser.parse_args()
    if args.self_test:
        self_test(args.autograd_root)
        return
    text = json.dumps(build_spec(args.experiment_id), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(str(args.output.resolve()))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
