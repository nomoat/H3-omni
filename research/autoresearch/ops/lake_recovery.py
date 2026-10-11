#!/usr/bin/env python3
"""Read-only by default. Controlled scale-zero recovery after explicit approval.

No database requests or data writes. Credentials stay in subprocess/memory.
The only cloud mutations are service-level scaling to zero and restoration.
Missing metrics, HTTP failures and elapsed time alone never prove termination.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROJECT = "cj-dev-498907"
REGION = "us-central1"
SERVICE = "autograd-data-lake"
KEY = "/home/homo/model/srp/AutoGrad/secrets/autograd-deployer.json"
MONITOR_ACCOUNT = "sandbox-control-plane-deployer@cj-dev-498907.iam.gserviceaccount.com"
APPROVAL = "scale-zero-then-restore"
SHUTDOWN = "Shutting down user disabled instance"
SCALING_KEYS = {
    "run.googleapis.com/scalingMode", "run.googleapis.com/manualInstanceCount",
    "run.googleapis.com/minScale", "run.googleapis.com/maxScale",
    "autoscaling.knative.dev/minScale", "autoscaling.knative.dev/maxScale",
}


class RecoveryError(RuntimeError):
    pass


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sanitized_config(raw):
    metadata, spec, status = raw["metadata"], raw["spec"], raw["status"]
    template = spec["template"]
    annotations = metadata.get("annotations", {})
    mode = annotations.get("run.googleapis.com/scalingMode", "automatic")
    return {
        "name": metadata["name"], "uid": metadata["uid"],
        "spec_sha256": digest(spec),
        "revision": status["latestReadyRevisionName"],
        "scaling": {key: value for key, value in annotations.items() if key in SCALING_KEYS},
        "restore_scaling": "auto" if mode == "automatic" else annotations.get("run.googleapis.com/manualInstanceCount"),
        "revision_scaling": {key: value for key, value in template.get("metadata", {}).get("annotations", {}).items() if key in SCALING_KEYS},
        "traffic": [{key: value for key, value in entry.items() if key in {"revisionName", "percent", "latestRevision", "tag"}} for entry in status.get("traffic", [])],
        "service_account": template["spec"].get("serviceAccountName"),
        "timeout_seconds": template["spec"].get("timeoutSeconds"),
        "concurrency": template["spec"].get("containerConcurrency"),
        "containers": [{"image": item.get("image"), "environment_variable_names": [env["name"] for env in item.get("env", [])]} for item in template["spec"]["containers"]],
    }


def assert_unchanged(original, current, disabled=False):
    for key in ("name", "uid", "spec_sha256", "revision", "traffic"):
        if original[key] != current[key]:
            raise RecoveryError(f"Service {key} changed; refuse automatic recovery")
    if disabled and not (
        current["scaling"].get("run.googleapis.com/scalingMode") == "manual"
        and current["scaling"].get("run.googleapis.com/manualInstanceCount") == "0"
    ):
        raise RecoveryError("Service is not confirmed disabled; refuse restoration")


class Cloud:
    def __init__(self, key_file, monitoring_account):
        self.key_file = key_file
        self.monitoring_account = monitoring_account
        self.monitor_token = None

    def command(self, args, monitoring=False):
        env = dict(os.environ, CLOUDSDK_CORE_DISABLE_PROMPTS="1")
        if monitoring:
            env.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
            prefix = ["gcloud", "--account=" + self.monitoring_account]
        else:
            env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = self.key_file
            prefix = ["gcloud", "--project=" + PROJECT]
        try:
            result = subprocess.run(prefix + args, env=env, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise RecoveryError("gcloud deadline exceeded; raw output suppressed") from None
        if result.returncode:
            raise RecoveryError(f"gcloud failed with exit code {result.returncode}; raw output suppressed")
        return result.stdout

    def config(self):
        return sanitized_config(json.loads(self.command([
            "run", "services", "describe", SERVICE, "--region=" + REGION, "--format=json",
        ])))

    def revision_created(self, revision):
        raw = json.loads(self.command(["run", "revisions", "describe", revision, "--region=" + REGION, "--format=json"]))
        return raw["metadata"]["creationTimestamp"]

    def scale(self, value):
        self.command(["run", "services", "update", SERVICE, "--region=" + REGION, "--scaling=" + value, "--quiet", "--format=json"])

    def lifecycle(self, since):
        query = (
            f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE}" '
            f'AND logName="projects/{PROJECT}/logs/run.googleapis.com%2Fvarlog%2Fsystem" '
            f'AND timestamp>="{since}"'
        )
        rows = json.loads(self.command(["logging", "read", query, "--limit=1001", "--order=asc", "--format=json"]))
        if len(rows) > 1000:
            raise RecoveryError("Lifecycle log result was truncated; refuse incomplete inventory")
        events = []
        for row in rows:
            message = row.get("textPayload") or row.get("jsonPayload", {}).get("message", "")
            kind = "start" if message.startswith("Starting new instance.") else "shutdown" if message.strip() == SHUTDOWN else "other"
            events.append({
                "timestamp": row["timestamp"], "instance_id": row.get("labels", {}).get("instanceId"),
                "revision": row.get("resource", {}).get("labels", {}).get("revision_name"),
                "kind": kind, "insert_id": row.get("insertId"),
            })
        return events

    def metrics(self):
        if self.monitor_token is None:
            self.monitor_token = self.command(["auth", "print-access-token"], monitoring=True).strip()
        now = utcnow()
        params = {
            "filter": f'metric.type="run.googleapis.com/container/instance_count" AND resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE}"',
            "interval.startTime": (now - dt.timedelta(minutes=20)).isoformat(),
            "interval.endTime": now.isoformat(), "view": "FULL", "pageSize": "100",
        }
        series = []
        for _ in range(10):
            url = f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/timeSeries?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(url, headers={"Authorization": "Bearer " + self.monitor_token})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = json.load(response)
            except urllib.error.HTTPError as error:
                return {"http_status": error.code, "series": []}
            except (urllib.error.URLError, TimeoutError):
                return {"http_status": None, "error": "monitoring transport failure", "series": []}
            for item in raw.get("timeSeries", []):
                series.append({"revision": item["resource"]["labels"]["revision_name"], "state": item["metric"]["labels"]["state"],
                               "points": [{"time": point["interval"]["endTime"], "count": int(point["value"]["int64Value"])} for point in item.get("points", [])]})
            if not raw.get("nextPageToken"):
                return {"http_status": 200, "series": series}
            params["pageToken"] = raw["nextPageToken"]
        raise RecoveryError("Monitoring result exceeded page bound")


def metric_buckets(metrics, revisions):
    if metrics.get("http_status") != 200:
        return {}
    wanted = {(revision, state) for revision in revisions for state in ("active", "idle")}
    buckets = {}
    for series in metrics["series"]:
        key = (series["revision"], series["state"])
        for point in series["points"]:
            buckets.setdefault(point["time"], {})[key] = point["count"]
    return {time: sum(values.values()) for time, values in buckets.items() if wanted <= values.keys()}


def active_inventory(events, revision):
    instances = {}
    for event in sorted(events, key=lambda row: timestamp(row["timestamp"])):
        if event["revision"] == revision and event["instance_id"]:
            if event["kind"] == "start":
                instances[event["instance_id"]] = event["timestamp"]
            elif event["kind"] == "shutdown":
                instances.pop(event["instance_id"], None)
    return instances


def termination_evidence(original, events, metrics, disabled_at, now):
    """Return positive evidence; never infer zero from an absent series or log."""
    disabled = timestamp(disabled_at)
    known = set(original["instance_ids"])
    shutdowns = {}
    latest_start = disabled
    for event in sorted(events, key=lambda row: timestamp(row["timestamp"])):
        when, identity = timestamp(event["timestamp"]), event["instance_id"]
        if when < disabled or not identity:
            continue
        if event["kind"] == "start":
            latest_start = max(latest_start, when)
            known.add(identity)
            shutdowns.pop(identity, None)
        elif event["kind"] == "shutdown":
            shutdowns[identity] = when
    # A platform shutdown signal plus the documented 10-second kill deadline,
    # with a 30-second margin measured after receipt, is an alternative when
    # Cloud Monitoring omits its zero series. The caller enforces receipt grace.
    logs_complete = bool(known) and known <= shutdowns.keys()
    log_evidence = {"kind": "all_known_instances_shutdown", "instance_ids": sorted(known),
                    "shutdown_times": {key: shutdowns[key].isoformat() for key in sorted(known)}} if logs_complete else None
    buckets = metric_buckets(metrics, {original["config"]["revision"]})
    eligible = sorted((time for time, count in buckets.items() if timestamp(time) >= latest_start and count == 0 and 0 <= (now - timestamp(time)).total_seconds() <= 180), key=timestamp)
    # Earlier positive drain samples are expected; no sample after the first
    # selected zero may show another live instance.
    if len(eligible) >= 2:
        first = timestamp(eligible[-2])
        positive_after = any(point["count"] > 0 and timestamp(point["time"]) >= first for series in metrics.get("series", []) for point in series["points"])
        if not positive_after:
            return {"kind": "two_explicit_zero_samples", "sample_times": eligible[-2:]}
    if log_evidence:
        last_shutdown = max(shutdowns[key] for key in known)
        if not any(point["count"] > 0 and timestamp(point["time"]) > last_shutdown + dt.timedelta(seconds=30) for series in metrics.get("series", []) for point in series["points"]):
            return log_evidence
    return None


def probe(cloud):
    config = cloud.config()
    if config["name"] != SERVICE or config["restore_scaling"] not in ("auto", "1"):
        raise RecoveryError("Unexpected service or original scaling configuration")
    if len(config["traffic"]) != 1 or config["traffic"][0].get("tag") or config["traffic"][0].get("percent") != 100:
        raise RecoveryError("Recovery requires one untagged traffic revision")
    created = cloud.revision_created(config["revision"])
    events = cloud.lifecycle(created)
    metrics = cloud.metrics()
    instances = active_inventory(events, config["revision"])
    buckets = metric_buckets(metrics, {config["revision"]})
    latest = max(buckets, key=timestamp, default=None)
    ready = bool(latest and (utcnow() - timestamp(latest)).total_seconds() <= 180 and buckets[latest] == len(instances) == 1)
    return {"captured_at": utcnow().isoformat(), "config": config, "revision_created": created,
            "lifecycle": events, "metrics": metrics, "instance_ids": sorted(instances), "ready_to_disable": ready,
            "reason": "One live instance is corroborated by fresh metrics and lifecycle identity" if ready else "Cannot corroborate exactly one old writer; no disable allowed"}


def wait_for_termination(cloud, original, state, output, timeout):
    deadline = time.monotonic() + timeout
    seen_shutdown_at = None
    seen_shutdown_ids = None
    while time.monotonic() < deadline:
        assert_unchanged(original["config"], cloud.config(), disabled=True)
        events = cloud.lifecycle(state["disable_requested_at"])
        metrics = cloud.metrics()
        proof = termination_evidence(original, events, metrics, state["disable_requested_at"], utcnow())
        save(output / "termination-observation.json", {"observed_at": utcnow().isoformat(), "lifecycle": events, "metrics": metrics, "evidence": proof})
        if proof and proof["kind"] == "all_known_instances_shutdown":
            identities = digest(proof)
            if identities != seen_shutdown_ids:
                seen_shutdown_at, seen_shutdown_ids = time.monotonic(), identities
            if time.monotonic() - seen_shutdown_at < 30:
                proof = None
        else:
            seen_shutdown_at, seen_shutdown_ids = None, None
        if proof:
            assert_unchanged(original["config"], cloud.config(), disabled=True)
            state.update(status="old_writer_terminated", evidence=proof, termination_confirmed_at=utcnow().isoformat())
            save(output / "state.json", state)
            return
        print(json.dumps({"status": "awaiting_positive_termination_evidence", "monitoring_http_status": metrics["http_status"]}), flush=True)
        time.sleep(20)
    raise RecoveryError("Termination evidence deadline exceeded; leave service disabled and retain state for resume")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="probe", choices=("probe", "recover", "resume"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-file", default=KEY)
    parser.add_argument("--monitoring-account", default=MONITOR_ACCOUNT)
    parser.add_argument("--approval", help="Supply only after actual user approval: " + APPROVAL)
    parser.add_argument("--termination-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.mode != "probe" and args.approval != APPROVAL:
        parser.error("Cloud mutation requires actual user approval and the explicit --approval acknowledgement")
    if not 60 <= args.termination_timeout <= 1800:
        parser.error("termination timeout must be 60..1800 seconds")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.mode == "resume")
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud = Cloud(args.key_file, args.monitoring_account)
        state = {"status": "read_only"}
        try:
            if args.mode == "resume":
                original = json.loads((output / "before.json").read_text())
                state = json.loads((output / "state.json").read_text())
                if state["status"] == "restored":
                    print(json.dumps({"status": "already_restored"})); return 0
                if "disable_requested_at" not in state:
                    raise RecoveryError("No recorded disable attempt to resume")
                assert_unchanged(original["config"], cloud.config(), disabled=True)
            else:
                original = probe(cloud)
                save(output / "before.json", original)
                if args.mode == "probe":
                    print(json.dumps({"status": "read_only", "ready_to_disable": original["ready_to_disable"], "instance_count": len(original["instance_ids"]), "monitoring_http_status": original["metrics"]["http_status"]})); return 0
                if not original["ready_to_disable"]:
                    raise RecoveryError(original["reason"])
                assert_unchanged(original["config"], cloud.config())
                state = {"status": "disable_requested", "disable_requested_at": utcnow().isoformat()}
                save(output / "state.json", state)
                cloud.scale("0")
                assert_unchanged(original["config"], cloud.config(), disabled=True)
            wait_for_termination(cloud, original, state, output, args.termination_timeout)
            state["status"] = "restore_requested"
            save(output / "state.json", state)
            cloud.scale(original["config"]["restore_scaling"])
            after = cloud.config()
            assert_unchanged(original["config"], after)
            if after["restore_scaling"] != original["config"]["restore_scaling"]:
                raise RecoveryError("Restored scaling does not match the original")
            save(output / "after.json", after)
            state.update(status="restored", restored_at=utcnow().isoformat(), note="Scaling restored; database recovery still requires a separate existing-row read")
            save(output / "state.json", state)
            print(json.dumps(state)); return 0
        except RecoveryError as error:
            state["last_error"] = str(error)
            save(output / "state.json", state)
            print(json.dumps({"status": state["status"], "error": str(error)}), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
