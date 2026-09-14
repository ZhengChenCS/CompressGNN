#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git submodule update --init src/offline/vendor/CompressGraph
cd src/offline
python setup.py install
cd ../runtime
python setup.py install
cd ../sample/
python setup.py install
cd ../../