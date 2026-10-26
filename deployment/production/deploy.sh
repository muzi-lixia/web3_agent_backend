#!/usr/bin/env bash
# Run on the server: ./deploy.sh backend|frontend ghcr.io/...@sha256:...
set -Eeuo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

component=${1:-}
image=${2:-}
case "$component" in
  backend) image_key=BACKEND_IMAGE ;;
  frontend) image_key=FRONTEND_IMAGE ;;
  *) echo 'Usage: deploy.sh backend|frontend IMAGE_DIGEST_OR_COMMIT_TAG' >&2; exit 2 ;;
esac
# Only this project's images and immutable CI-produced identifiers are accepted.
if [[ ! "$image" =~ ^ghcr\.io/muzi-lixia/web3_agent_${component}(@sha256:[a-f0-9]{64}|:sha-[a-f0-9]{40})$ ]]; then
  echo 'Expected this component image with a sha256 digest or full commit tag.' >&2
  exit 2
fi
[[ -f production.env && -f app.env ]] || { echo 'Configure production.env and app.env first.' >&2; exit 2; }

# Shared lock serializes deployments from the TWO independent repositories.
exec 9>.deploy.lock
flock -w 300 9
compose=(docker compose --env-file production.env -f compose.yml)
previous_image=$(python3 - "$image_key" <<'PY'
from pathlib import Path
import sys
key = sys.argv[1] + '='
values = [line[len(key):].strip() for line in Path('production.env').read_text().splitlines() if line.startswith(key)]
if len(values) != 1 or not values[0]:
    raise SystemExit('production.env must contain exactly one nonempty ' + key)
print(values[0])
PY
)

# Pull BEFORE changing the running service or saving the new version.
env "$image_key=$image" "${compose[@]}" pull "$component"

snapshot=$(mktemp .production.env.XXXXXX)
cp production.env "$snapshot"
cleanup() { rm -f -- "$snapshot"; }
trap cleanup EXIT
python3 - "$image_key" "$image" <<'PY'
import os
from pathlib import Path
import sys
p = Path('production.env')
key, image = sys.argv[1:]
lines = [key + '=' + image if line.startswith(key + '=') else line for line in p.read_text().splitlines()]
tmp = p.with_suffix('.env.next')
with tmp.open('w') as f:
    f.write('\n'.join(lines) + '\n')
    f.flush()
    os.fsync(f.fileno())
os.chmod(tmp, 0o600)
os.replace(tmp, p)
PY

if "${compose[@]}" up -d --no-deps --wait --wait-timeout 180 "$component"; then
  cp "$snapshot" production.env.previous
  echo "Deployed $component: $image"
  echo "Previous image: $previous_image"
else
  echo 'Health check failed; restoring the previous image configuration.' >&2
  cp "$snapshot" production.env
  if ! "${compose[@]}" up -d --no-deps --wait --wait-timeout 180 "$component"; then
    echo 'Rollback also failed; inspect container status and database compatibility.' >&2
  fi
  exit 1
fi
# This never deletes images or database volumes and never runs database migrations.
