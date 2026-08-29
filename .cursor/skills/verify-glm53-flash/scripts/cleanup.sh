#!/usr/bin/env bash
# Tear down a serve THIS run started. Never kill a pre-existing lab instance.
# Never deletes artifacts/.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if [[ ! -f "$STATE_DIR/started" ]]; then
  echo "no verify-owned serve (missing $STATE_DIR/started); leaving containers alone"
  echo "artifacts kept under $ARTIFACTS"
  exit 0
fi

echo "stopping verify-owned serve via $REPO/stop.sh"
(
  cd "$REPO"
  CONTAINER_NAME="$CONTAINER_NAME" WORKER_HOST="$WORKER_HOST" ./stop.sh
)
rm -rf "$STATE_DIR"
echo "state removed; artifacts kept under $ARTIFACTS"
