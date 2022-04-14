#!/bin/bash

set -e

DOWNLOADS_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DOWNLOADS_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

DATA_DIR="${PROJECT_DIR}/data/"
CACHE_DIR="${PROJECT_DIR}/cache/"

mkdir -p "${DATA_DIR}"
mkdir -p "${CACHE_DIR}"

# download test-clean subset
echo "downloading LibriSpeech test-clean..."
wget http://www.openslr.org/resources/12/test-clean.tar.gz

# extract test-clean subset
echo "extracting LibriSpeech test-clean..."
tar -xf test-clean.tar.gz \
    -C "${DATA_DIR}"

# delete archive
rm -f "test-clean.tar.gz"
