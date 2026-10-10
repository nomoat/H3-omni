#!/bin/bash
# Fetch Ubuntu's ICU headers and runtime without modifying the container.
set -euo pipefail
H3_ICU=/home/cj/h3-cuda/icu
mkdir -p "$H3_ICU/lists/partial" "$H3_ICU/root"
cd "$H3_ICU"
apt-get -o Dir::State::lists="$H3_ICU/lists" update
apt-get -o Dir::State::lists="$H3_ICU/lists" download libicu-dev libicu70
for package in *.deb; do dpkg-deb -x "$package" root; done
sha256sum ./*.deb > SHA256SUMS
