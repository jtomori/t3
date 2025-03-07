#!/usr/bin/env bash

# Sane defaults for bash
set -o nounset
set -o errexit
trap 'echo "Aborting due to errexit on line $LINENO. Exit code: $?" >&2' ERR
set -o errtrace
set -o pipefail

# Create directories
mkdir gme ogg

# Download testing GME file
wget -c -O 'gme/Mein Woerter-Bilderbuch Unser Zuhause.gme' https://ravensburger.cloud/rvwebsite/rvDE/db/applications/Mein%20Woerter-Bilderbuch%20Unser%20Zuhause.gme

# GME -> OGG audio files
tttool media -d ogg 'gme/Mein Woerter-Bilderbuch Unser Zuhause.gme'

