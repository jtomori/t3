#!/usr/bin/env bash

set -o nounset
set -o errexit
trap 'echo "Aborting due to errexit on line $LINENO. Exit code: $?" >&2' ERR
set -o errtrace
set -o pipefail

# GME -> audio files
tttool media -d ogg 'gme/Mein Woerter-Bilderbuch Unser Zuhause.gme'

