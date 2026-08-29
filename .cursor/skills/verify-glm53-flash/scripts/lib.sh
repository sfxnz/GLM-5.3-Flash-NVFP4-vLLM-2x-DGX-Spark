# Shared paths and the same env defaults as run.sh. Source from the other helpers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$SKILL_DIR/../../.." && pwd)"
STATE_DIR="$SKILL_DIR/.run-state"
ARTIFACTS="$SKILL_DIR/artifacts"

CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-nvfp4}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-glm53-sm121-v11}"
SERVED_NAME="${SERVED_NAME:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
WORKER_HOST="${WORKER_HOST:-spark2}"
API="http://127.0.0.1:${PORT}/v1"
