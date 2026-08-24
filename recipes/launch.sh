#!/usr/bin/env bash
# GLM-5.2 legacy launcher compatibility wrapper.
# This file remains MIT under the repository's REUSE policy.
set -Eeuo pipefail

printf '%s\n' 'recipes/launch.sh is deprecated; use bin/champion with a canonical preset' >&2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-launch}" in
  render) exec "$ROOT/bin/champion" render "${CHAMPION_PROFILE:-fast}" ;;
  launch) exec "$ROOT/bin/champion" launch "${CHAMPION_PROFILE:-fast}" ;;
  *) printf 'usage: CHAMPION_PROFILE={fast|balanced|large} %s {render|launch}\n' "$0" >&2; exit 64 ;;
esac
