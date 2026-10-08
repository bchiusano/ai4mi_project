#! /bin/bash
# Runs configs/test.yaml on top of configs/common.yaml, extra arguments override config values,
# e.g. ./docs/test.sh --fold 2

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUNDLE="$(cd "$homedir/.." && pwd)"

echo "Bundle root: $BUNDLE"

source "$BUNDLE/../../.venv/bin/activate"

export PYTHONPATH="$BUNDLE"

python -m monai.bundle run \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --logging_file "$BUNDLE/configs/logging.conf" \
    --config_file "['$BUNDLE/configs/common.yaml','$BUNDLE/configs/test.yaml']" \
    --bundle_root "$BUNDLE" \
    "$@"
