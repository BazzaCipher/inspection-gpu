#!/usr/bin/env python3
"""Destroys vast.ai instances that should no longer exist. Runs every 2 min from a systemd user
timer and from cron on this box (both, independently; linger is on so neither needs a login or a
Claude session), every 10 min from GitHub Actions (BazzaCipher/inspection-gpu, reaper.yml) for when
this box is frozen or off, and from worker.py before and after every run.

An instance is destroyed when any of these hold:
  - its label is inspect-<room>-<start>-<limit> and start + limit (+2 min grace) has passed
  - it is stopped or exited for more than 15 min (a stopped vast instance still bills its disk)
  - it is older than MAX_AGE_MIN, whatever its label (this account is only used by this pipeline)

Destroy, never stop. stdlib only; no output unless it acts or fails. Log: ~/.local/state/vast_reaper.log
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

MAX_AGE_MIN = 150    # worker.py caps a run at 120 min
API = "https://console.vast.ai/api"
LOG = os.path.expanduser("~/.local/state/vast_reaper.log")


def key():
    if os.environ.get("VAST_API_KEY"):   # GitHub Actions
        return os.environ["VAST_API_KEY"]
    for p in ("~/.config/vastai/vast_api_key", "~/.vast_api_key"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return open(p).read().strip()
    sys.exit("vast_reaper: no vast api key")


def call(method, path, k):
    req = urllib.request.Request(API + path, method=method, headers={"Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def why(i, now):
    label = i.get("label") or ""
    age = (now - float(i.get("start_date") or now)) / 60
    parts = label.split("-")
    if label.startswith("inspect-") and len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
        if now > int(parts[-2]) + int(parts[-1]) * 60 + 120:
            return f"past its {parts[-1]} min limit"
    status = (i.get("actual_status"), i.get("intended_status"))
    if ("stopped" in status or "exited" in status) and age > 15:
        return f"{status[0]}/{status[1]} for {age:.0f} min (still billing disk)"
    if age > MAX_AGE_MIN:
        return f"{age:.0f} min old (> {MAX_AGE_MIN})"
    return None


def reap(k=None, keep=None):
    """Destroy every instance that has a reason to go. `keep`: an id the caller is using right now."""
    k = k or key()
    now = time.time()
    left = []
    for i in call("GET", "/v1/instances/", k).get("instances", []):
        reason = why(i, now)
        if not reason or i["id"] == keep:
            continue
        for attempt in range(5):
            try:
                call("DELETE", f"/v0/instances/{i['id']}/", k)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
            time.sleep(3)
            if not any(x["id"] == i["id"] for x in call("GET", "/v1/instances/", k).get("instances", [])):
                break
        else:
            left.append(i["id"])
        log(f"destroyed {i['id']} ({i.get('label')}, ${i.get('dph_total') or 0:.3f}/hr): {reason}"
            + (" — STILL EXISTS" if i["id"] in left else ""))
    return left


if __name__ == "__main__":
    try:
        sys.exit(1 if reap() else 0)
    except Exception as e:  # noqa: BLE001
        log(f"reaper error: {e}")
        sys.exit(2)
