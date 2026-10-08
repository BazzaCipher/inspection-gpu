#!/bin/bash
# Pod entrypoint: bring up sshd for the orchestrator, then self-destruct after MAX_LIFETIME_MIN
# so a crashed or forgotten worker can never leave a pod billing overnight.
set -u

if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd
echo "[start] sshd up; pod ${RUNPOD_POD_ID:-?}; gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

MAX_LIFETIME_MIN="${MAX_LIFETIME_MIN:-90}"
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  (
    sleep $((MAX_LIFETIME_MIN * 60))
    echo "[watchdog] ${MAX_LIFETIME_MIN} min reached — terminating pod ${RUNPOD_POD_ID}"
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
  ) &
fi

sleep infinity
