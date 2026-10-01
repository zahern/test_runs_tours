#!/bin/bash -l
# Rebuild the population-INDEPENDENT MATSim base inputs (network + GTFS transit)
# from the OSM extract and GTFS feed. These only change when the OSM/GTFS inputs
# change, so this is run rarely; the outputs are staged next to this script and
# reused by every job_matsim_scenario.pbs run.
#
# Produces (in this matsim/ dir):
#   network.xml.gz            multimodal (car + pt) network, EPSG:28356
#   transitSchedule.xml.gz    GTFS transit mapped onto the network
#   transitVehicles.xml.gz
#
# Requirements: the staged pt2matsim-shaded.jar, a Java 17+ runtime, and `osmium`
# (conda-forge osmium-tool) to convert the .pbf to OSM XML.
#
# Usage (from the code dir):
#   bash matsim/build_base_inputs.sh
#   OSM_PBF=... GTFS_ZIP=... CRS=EPSG:28356 bash matsim/build_base_inputs.sh

set -euo pipefail
cd "$(dirname "$0")"                          # -> matsim/

JAR="pt2matsim-shaded.jar"
CRS="${CRS:-EPSG:28356}"
OSM_PBF="${OSM_PBF:-../saved_data/osm_cache/seq_osm.pbf}"
GTFS_ZIP="${GTFS_ZIP:-../saved_data/osm_cache/seq_gtfs.zip}"
GTFS_DAY="${GTFS_DAY:-dayWithMostServices}"
HEAP="${HEAP:-12g}"
JAVA_BIN="${JAVA_HOME:+$JAVA_HOME/bin/}java"; command -v "$JAVA_BIN" >/dev/null 2>&1 || JAVA_BIN=java

[ -f "$JAR" ]      || { echo "ERROR: $JAR missing (build with matsim/build_pt2matsim.sh)"; exit 2; }
[ -f "$OSM_PBF" ]  || { echo "ERROR: OSM pbf not found: $OSM_PBF"; exit 2; }
[ -f "$GTFS_ZIP" ] || { echo "ERROR: GTFS zip not found: $GTFS_ZIP"; exit 2; }
command -v osmium >/dev/null 2>&1 || { echo "ERROR: need osmium-tool (conda install -c conda-forge osmium-tool)"; exit 2; }

echo "[1/5] pbf -> osm.gz (osmium)"
osmium cat "$OSM_PBF" -o seq.osm.gz --overwrite

echo "[2/5] OSM -> multimodal network ($CRS)"
"$JAVA_BIN" -cp "$JAR" org.matsim.pt2matsim.run.CreateDefaultOsmConfig osm_config.xml
sed -i "s#<param name=\"osmFile\" value=\"[^\"]*\" />#<param name=\"osmFile\" value=\"seq.osm.gz\" />#" osm_config.xml
sed -i "s#<param name=\"outputCoordinateSystem\" value=\"[^\"]*\" />#<param name=\"outputCoordinateSystem\" value=\"$CRS\" />#" osm_config.xml
sed -i "s#<param name=\"outputNetworkFile\" value=\"[^\"]*\" />#<param name=\"outputNetworkFile\" value=\"network_car.xml.gz\" />#" osm_config.xml
sed -i "s#<param name=\"writeCRS\" value=\"false\" />#<param name=\"writeCRS\" value=\"true\" />#" osm_config.xml
"$JAVA_BIN" -Xmx"$HEAP" -cp "$JAR" org.matsim.pt2matsim.run.Osm2MultimodalNetwork osm_config.xml

echo "[3/5] GTFS -> unmapped transit schedule ($CRS, $GTFS_DAY)"
rm -rf gtfs && mkdir -p gtfs && unzip -q -o "$GTFS_ZIP" -d gtfs
"$JAVA_BIN" -Xmx"$HEAP" -cp "$JAR" org.matsim.pt2matsim.run.Gtfs2TransitSchedule \
    "gtfs/" "$GTFS_DAY" "$CRS" transitSchedule_unmapped.xml.gz transitVehicles.xml.gz

echo "[4/5] map transit schedule onto network"
"$JAVA_BIN" -cp "$JAR" org.matsim.pt2matsim.run.CreateDefaultPTMapperConfig ptmapper_config.xml
sed -i 's#<param name="inputNetworkFile" value="" />#<param name="inputNetworkFile" value="network_car.xml.gz" />#' ptmapper_config.xml
sed -i 's#<param name="inputScheduleFile" value="" />#<param name="inputScheduleFile" value="transitSchedule_unmapped.xml.gz" />#' ptmapper_config.xml
sed -i 's#<param name="outputNetworkFile" value="" />#<param name="outputNetworkFile" value="network.xml.gz" />#' ptmapper_config.xml
sed -i 's#<param name="outputScheduleFile" value="" />#<param name="outputScheduleFile" value="transitSchedule.xml.gz" />#' ptmapper_config.xml
sed -i 's#<param name="numOfThreads" value="2" />#<param name="numOfThreads" value="6" />#' ptmapper_config.xml
"$JAVA_BIN" -Xmx"$HEAP" -cp "$JAR" org.matsim.pt2matsim.run.PublicTransitMapper ptmapper_config.xml

echo "[5/5] cleanup"
rm -f seq.osm.gz network_car.xml.gz transitSchedule_unmapped.xml.gz
ls -lh network.xml.gz transitSchedule.xml.gz transitVehicles.xml.gz
echo "Base inputs rebuilt in $(pwd)"
