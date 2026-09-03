#!/usr/bin/env bash
# vendored from sfxnz/forge kit @ 6285f70
# Vendor the kit into a recipe repo, or check that the vendored copy matches forge.
#   kit/sync.sh <recipe-path>           write kit/*.sh kit/*.py kit/probes/* into <recipe-path>/kit/
#   kit/sync.sh --check <recipe-path>   exit 1 if any vendored file differs from forge (stamp line ignored)
# Every vendored file starts with `# vendored from sfxnz/forge kit @ <sha>`.
# Run from a vendored copy (this file carries the stamp line), the forge kit is
# ${FORGE_KIT:-$HOME/projects/ai-lab/forge/kit}, not the directory this file sits in.
set -euo pipefail
kit="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if grep -q '^# vendored from sfxnz/forge kit @ ' "${BASH_SOURCE[0]}"; then
  kit="${FORGE_KIT:-$HOME/projects/ai-lab/forge/kit}"
  if [[ ! -f "$kit/sync.sh" ]]; then
    echo "vendored copy: forge kit not found at $kit (set FORGE_KIT)" >&2
    exit 2
  fi
  echo "vendored copy: forge kit is $kit"
fi
check=0
if [[ "${1:-}" == "--check" ]]; then
  check=1
  shift
fi
recipe="${1:?usage: kit/sync.sh [--check] <recipe-path>}"
dest="$recipe/kit"
sha="$(git -C "$kit" rev-parse --short HEAD)"

files=()
while IFS= read -r f; do
  files+=("$f")
done < <(cd "$kit" && find . -maxdepth 1 -type f \( -name '*.sh' -o -name '*.py' \) -printf '%P\n'; cd "$kit" && find probes -type f -printf '%p\n' | sort)

vendored() {
  local src="$kit/$1"
  # A shebang must stay on line 1; the stamp then goes on line 2.
  if [[ "$(head -c 2 "$src")" == "#!" ]]; then
    head -1 "$src"
    echo "# vendored from sfxnz/forge kit @ $sha"
    tail -n +2 "$src"
  else
    echo "# vendored from sfxnz/forge kit @ $sha"
    cat "$src"
  fi
}

strip_stamp() { grep -v '^# vendored from sfxnz/forge kit @ ' "$1"; }

rc=0
for f in "${files[@]}"; do
  if (( check )); then
    if [[ ! -f "$dest/$f" ]]; then
      echo "missing $dest/$f"
      rc=1
    elif ! diff -q <(strip_stamp "$dest/$f") "$kit/$f" >/dev/null; then
      echo "differs $dest/$f"
      rc=1
    fi
  else
    mkdir -p "$(dirname "$dest/$f")"
    vendored "$f" >"$dest/$f"
    chmod --reference="$kit/$f" "$dest/$f"
    echo "wrote $dest/$f"
  fi
done
if (( check )); then
  (( rc == 0 )) && echo "kit in sync with forge @ $sha"
  exit "$rc"
fi
