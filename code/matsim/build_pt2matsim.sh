#!/bin/bash -l
# Rebuild pt2matsim-shaded.jar from source (only needed if the staged jar is
# missing or you want a different pt2matsim/MATSim version). The shaded jar
# bundles MATSim + geotools + the pt2matsim OSM/GTFS tools and can run the
# MATSim Controler directly.
#
# Requirements: a Java 17+ JDK and Maven (this script downloads a portable
# Apache Maven if `mvn` is not on PATH). Needs internet (repo.matsim.org + Maven
# Central). Produces ./pt2matsim-shaded.jar.
#
# Usage:  bash matsim/build_pt2matsim.sh [version]   (default v26.6)

set -euo pipefail
cd "$(dirname "$0")"

VER="${1:-v26.6}"
WORK="$(mktemp -d)"
echo "building pt2matsim ${VER} in ${WORK}"

MVN="$(command -v mvn || true)"
if [ -z "$MVN" ]; then
  echo "no mvn on PATH — downloading portable Apache Maven 3.9.9"
  curl -sL -o "$WORK/mvn.zip" "https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.zip"
  unzip -q -o "$WORK/mvn.zip" -d "$WORK"
  MVN="$WORK/apache-maven-3.9.9/bin/mvn"
fi

curl -sL -o "$WORK/src.zip" "https://github.com/matsim-org/pt2matsim/archive/refs/tags/${VER}.zip"
unzip -q -o "$WORK/src.zip" -d "$WORK"
SRC="$(echo "$WORK"/pt2matsim-*/)"

( cd "$SRC" && "$MVN" -q -DskipTests -Dmaven.test.skip=true \
    -Dmaven.repo.local="$WORK/.m2" package )

SHADED="$(ls "$SRC"/target/*-shaded.jar | head -1)"
cp "$SHADED" ./pt2matsim-shaded.jar
echo "wrote $(pwd)/pt2matsim-shaded.jar"
ls -lh pt2matsim-shaded.jar
