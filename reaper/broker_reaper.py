#!/usr/bin/env python3
"""Off-box backstop for Wolfgang's compute broker: runs from GitHub Actions every 10 min,
so a lease still dies when rauf (and the broker on it) is frozen or off.

The broker names every paid machine wbroker-<lease>-<deadline epoch>. This destroys:
  - a wbroker machine more than GRACE past its deadline
  - a stopped/exited machine (the broker destroys those too; a stopped vast box bills disk)
  - on vast, any other machine older than STRAY_MIN (the account is the broker's alone;
    with rauf up its sweep kills such strays within 2 min)
Destroy, never stop. Network volumes are left alone: cheap, and the broker expires them.
Keys: VAST_API_KEY / RUNPOD_API_KEY (either may be missing). stdlib only.
Self-check: python3 broker_reaper.py --selftest
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

GRACE, STRAY_MIN = 600, 60
LABEL = re.compile(r"^wbroker-L[0-9a-f]+-(\d{10})$")


def http(method, url, key):
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
        "User-Agent": "wolfgang-reaper/1"})  # RunPod's Cloudflare blocks urllib's UA
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def why(name, stopped, age_min, now, stray_ok):
    """The reason to destroy, or None."""
    m = LABEL.match(name or "")
    if m and now > int(m.group(1)) + GRACE:
        return f"{(now - int(m.group(1))) / 60:.0f} min past its deadline"
    if stopped:
        return "stopped"
    if not m and stray_ok and age_min is not None and age_min > STRAY_MIN:
        return f"not a broker lease, {age_min:.0f} min old"
    return None


def vast(key, now):
    found, tok = [], None
    while True:
        q = "?limit=25" + (f"&after_token={urllib.parse.quote(tok)}" if tok else "")
        r = http("GET", "https://console.vast.ai/api/v1/instances" + q, key)
        found += r.get("instances", [])
        tok = r.get("next_token")
        if not tok:
            break
    for i in found:
        st = f"{i.get('actual_status')} {i.get('intended_status')} {i.get('cur_state')}"
        age = (now - float(i["start_date"])) / 60 if i.get("start_date") else None
        yield (i["id"], i.get("label"), why(i.get("label"), "stop" in st or "exited" in st,
                                            age, now, True),
               f"https://console.vast.ai/api/v0/instances/{i['id']}/")


def runpod(key, now):
    pods = http("GET", "https://rest.runpod.io/v1/pods", key)
    for p in pods if isinstance(pods, list) else pods.get("pods", []):
        # RunPod gives no reliable creation time here, so strays are left to the broker
        yield (p["id"], p.get("name"), why(p.get("name"), p.get("desiredStatus") == "EXITED",
                                           None, now, False),
               f"https://rest.runpod.io/v1/pods/{p['id']}")


def main():
    now, failed = time.time(), False
    for env, lister in (("VAST_API_KEY", vast), ("RUNPOD_API_KEY", runpod)):
        key = os.environ.get(env)
        if not key:
            print(f"{env} not set, skipping")
            continue
        try:
            for pid, name, reason, url in lister(key, now):
                if not reason:
                    continue
                try:
                    http("DELETE", url, key)
                    print(f"destroyed {lister.__name__} {pid} ({name}): {reason}")
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        print(f"FAILED to destroy {lister.__name__} {pid}: {e}")
                        failed = True
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"{lister.__name__}: list failed: {e}")
            failed = True
    sys.exit(1 if failed else 0)  # a red run is the alert


def selftest():
    now = 2_000_000_000
    assert why(f"wbroker-L1a2b3c4d-{now - 601}", False, 5, now, True).endswith("deadline")
    assert why(f"wbroker-L1a2b3c4d-{now - 300}", False, 5, now, True) is None  # in grace
    assert why(f"wbroker-L1a2b3c4d-{now + 9999}", True, 5, now, True) == "stopped"
    assert why("by-hand", False, 61, now, True).startswith("not a broker")
    assert why("by-hand", False, 30, now, True) is None
    assert why("by-hand", False, None, now, False) is None  # RunPod strays: broker's job
    assert why(None, False, 99, now, False) is None
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
