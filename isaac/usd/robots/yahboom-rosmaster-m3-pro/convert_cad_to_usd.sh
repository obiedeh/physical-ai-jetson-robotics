#!/usr/bin/env bash
# Convert ROSMASTER-M3Pro.x_t (Parasolid) to USD using Isaac Sim's HOOPS Exchange converter
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_DIR="/home/oedeh/isaacsim"
KIT_BIN="$ISAACSIM_DIR/kit/kit"
EXTSCACHE="$ISAACSIM_DIR/extscache"

HOOPS_MAIN="$EXTSCACHE/omni.services.convert.cad-507.0.2+107.3.1.u353/omni/services/convert/cad/services/process/hoops_main.py"
INPUT_FILE="$SCRIPT_DIR/ROSMASTER-M3Pro.x_t"
OUTPUT_USD="$SCRIPT_DIR/rosmaster_m3pro_cad.usd"
CONFIG_JSON="$SCRIPT_DIR/cad_convert_config.json"

cat > "$CONFIG_JSON" << 'JSONEOF'
{
  "bInstancing": "false",
  "bOptimize": "true",
  "bConvertHidden": "false",
  "bConvertMetadata": "true",
  "materialType": "1",
  "iUpAxis": "0",
  "dMetersPerUnit": "0.001"
}
JSONEOF

echo "Converting $INPUT_FILE -> $OUTPUT_USD"
echo "This will take a few minutes for a 181 MB assembly..."

"$KIT_BIN" \
  --allow-root \
  --ext-folder "$EXTSCACHE" \
  --enable omni.kit.converter.hoops_core \
  --exec "'$HOOPS_MAIN' --input-path '$INPUT_FILE' --output-path '$OUTPUT_USD' --config-path '$CONFIG_JSON'" \
  --/app/fastShutdown=1 \
  --info 2>&1 | tee /tmp/cad_convert.log | grep -E "convert|error|warn|success|USD|HOOPS|progress" || true

echo ""
if [ -f "$OUTPUT_USD" ]; then
  echo "SUCCESS: $OUTPUT_USD created ($(du -sh "$OUTPUT_USD" | cut -f1))"
else
  echo "FAILED: USD not created. Check /tmp/cad_convert.log"
fi
