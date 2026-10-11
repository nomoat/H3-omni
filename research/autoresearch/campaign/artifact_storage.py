#!/usr/bin/env python3
"""Create-only, content-addressed GCS storage for frozen H3 evidence packages.

Credentials are passed to gcloud by path. This program never opens that file.
It does not change bucket/object ACLs or create signed download URLs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.parse import quote

BUCKET = "cj-dev-498907-autograd-training-lake"
PREFIX = "artifacts/h3-h200-cold-search-20261010-100/sha256"
CREDENTIAL_PATH = "/home/homo/.local/share/autograd-training-lake/secrets/gcs.json"


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def hash_file(path: Path) -> dict:
    before = path.stat()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ) or size != after.st_size:
        raise ValueError(f"Source changed during hashing: {path}")
    return {
        "sha256": sha.hexdigest(), "size_bytes": size,
        "local_md5_hex": md5.hexdigest(),
        "local_md5_base64": base64.b64encode(md5.digest()).decode("ascii"),
    }


class Gcloud:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.env_overrides = {
            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": CREDENTIAL_PATH,
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
            "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "false",
        }
        self.env = dict(os.environ, **self.env_overrides)

    def paths(self, arguments: list[str]) -> tuple[list[str], Path, Path, Path]:
        self.index += 1
        stem = self.directory / f"{self.index:02d}"
        command = ["gcloud", "--quiet", "--verbosity=error", *arguments]
        record = stem.with_suffix(".command.json")
        save_json(record, {
            "argv": command, "environment_overrides": self.env_overrides,
            "shell_command": "env " + " ".join(
                shlex.quote(f"{key}={value}") for key, value in self.env_overrides.items()
            ) + " " + shlex.join(command),
            "started_unix": time.time(),
        })
        return command, stem.with_suffix(".stdout.log"), stem.with_suffix(".stderr.log"), record

    def run(self, arguments: list[str]) -> subprocess.CompletedProcess:
        command, stdout_path, stderr_path, record = self.paths(arguments)
        result = subprocess.run(command, env=self.env, capture_output=True, text=True)
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        details = json.loads(record.read_text())
        details.update(returncode=result.returncode, finished_unix=time.time())
        save_json(record, details)
        return result

    def readback_sha256(self, uri: str) -> dict:
        command, stdout_path, stderr_path, record = self.paths(["storage", "cat", uri])
        digest, size = hashlib.sha256(), 0
        with stderr_path.open("wb") as errors:
            process = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=errors)
            assert process.stdout is not None
            with process.stdout:
                for chunk in iter(lambda: process.stdout.read(8 << 20), b""):
                    digest.update(chunk)
                    size += len(chunk)
            code = process.wait()
        # Never save or print the binary stdout; preserve its digest and length.
        result = {"sha256": digest.hexdigest(), "size_bytes": size}
        stdout_path.write_text(json.dumps({"stream_consumed_without_logging": True, **result}) + "\n")
        details = json.loads(record.read_text())
        details.update(returncode=code, finished_unix=time.time(), streamed_output=result)
        save_json(record, details)
        if code:
            raise RuntimeError(f"GCS readback failed; see {stderr_path}")
        return result


def describe(client: Gcloud, uri: str) -> dict | None:
    result = client.run(["storage", "objects", "describe", uri, "--raw", "--format=json"])
    if result.returncode:
        # Permission errors and unrelated CLI/configuration errors are not absence.
        if re.search(r"\b404\b|\bNo such object\b", result.stderr, re.IGNORECASE):
            return None
        raise RuntimeError(f"GCS object describe failed; see {client.directory}")
    metadata = json.loads(result.stdout)
    if not isinstance(metadata, dict):
        raise ValueError("GCS describe did not return object metadata")
    return metadata


def verify_object(client: Gcloud, uri: str, metadata: dict, local: dict) -> dict:
    expected_bucket, expected_name = uri.removeprefix("gs://").split("/", 1)
    if metadata.get("bucket") != expected_bucket or metadata.get("name") != expected_name:
        raise ValueError("GCS object identity mismatch")
    generation = str(metadata.get("generation", ""))
    if not generation.isdigit() or int(generation) < 1:
        raise ValueError("GCS object has no valid immutable generation")
    if int(metadata.get("size", -1)) != local["size_bytes"]:
        raise ValueError("GCS object size mismatch")
    custom_sha = metadata.get("metadata", {}).get("sha256")
    if custom_sha is not None and custom_sha != local["sha256"]:
        raise ValueError("GCS object custom SHA256 disagrees with local content")
    server_md5 = metadata.get("md5Hash")
    if server_md5:
        if server_md5 != local["local_md5_base64"]:
            raise ValueError("GCS object MD5 mismatch")
        return {
            "passed": True, "method": "server_md5_and_size",
            "local_sha256_matches_source_manifest": True,
            "server_md5_matches_local": True,
            "server_crc32c_recorded_not_independently_compared": True,
            "verified_generation": generation,
        }
    readback = client.readback_sha256(uri + "#" + generation)
    if readback["sha256"] != local["sha256"] or readback["size_bytes"] != local["size_bytes"]:
        raise ValueError("GCS generation-pinned readback SHA256/size mismatch")
    return {
        "passed": True, "method": "generation_pinned_sha256_readback",
        "local_sha256_matches_source_manifest": True,
        "readback": readback, "verified_generation": generation,
        "server_crc32c_recorded_not_independently_compared": True,
    }


def upload_artifact(item: dict, client: Gcloud) -> dict:
    path = Path(item["local_path"]).resolve(strict=True)
    local = hash_file(path)
    if local["sha256"] != item["sha256"] or local["size_bytes"] != item["size_bytes"]:
        raise ValueError(f"Local artifact differs from frozen package manifest: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", local["sha256"]):
        raise ValueError("Invalid SHA256")
    object_name = f"{PREFIX}/{local['sha256']}/evidence.tar"
    uri = f"gs://{BUCKET}/{object_name}"
    metadata = describe(client, uri)
    disposition = "existing_verified"
    if metadata is None:
        result = client.run([
            "storage", "cp", str(path), uri, "--if-generation-match=0",
            "--content-md5=" + local["local_md5_base64"],
            "--custom-metadata=sha256=" + local["sha256"],
            "--manifest-path=" + str(client.directory / "upload.csv"),
        ])
        if result.returncode:
            if not re.search(r"\b412\b|conditionNotMet|precondition", result.stderr, re.IGNORECASE):
                raise RuntimeError(f"GCS create-only upload failed; see {client.directory}")
            disposition = "concurrent_create_verified"
        else:
            disposition = "created"
        metadata = describe(client, uri)
        if metadata is None:
            raise RuntimeError("GCS object missing after upload")
    verification = verify_object(client, uri, metadata, local)
    if hash_file(path) != local:
        raise ValueError(f"Local artifact changed during upload/verification: {path}")
    save_json(client.directory / "object-metadata.json", metadata)
    receipt = {
        **item, **local, "local_path": str(path), "bucket": BUCKET,
        "object": object_name, "storage_uri": uri,
        "download_url": f"https://storage.cloud.google.com/{BUCKET}/{quote(object_name, safe='/')}",
        "download_authentication": "Google login; no signed URL",
        "generation": str(metadata["generation"]),
        "generation_storage_uri": uri + "#" + str(metadata["generation"]),
        "server_md5_base64": metadata.get("md5Hash"),
        "server_crc32c_base64": metadata.get("crc32c"),
        "disposition": disposition, "verification": verification,
        "receipt_directory": str(client.directory),
    }
    save_json(client.directory / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.input_manifest.read_text())
    artifacts = source["artifacts"]
    if not artifacts:
        raise ValueError("No artifacts requested")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "run": source["run"], "status": "in_progress",
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": hash_file(args.input_manifest)["sha256"],
        "uploader_path": str(Path(__file__).resolve()),
        "uploader_sha256": hash_file(Path(__file__))["sha256"],
        "artifacts": [],
    }
    save_json(args.output, report)
    attempt = args.output.parent / ("cloud-upload-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    attempt.mkdir()
    try:
        for index, item in enumerate(artifacts):
            print(f"Verifying/uploading {item['role']} candidate={item.get('candidate')} bytes={item['size_bytes']}", flush=True)
            client = Gcloud(attempt / f"{index:02d}")
            report["artifacts"].append(upload_artifact(item, client))
            save_json(args.output, report)
            print(f"PASS {report['artifacts'][-1]['storage_uri']}", flush=True)
        report["status"] = "PASS"
        save_json(args.output, report)
        return 0
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        save_json(args.output, report)
        print(report["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
