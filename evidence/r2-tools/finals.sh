#!/usr/bin/env bash
# finals.sh: the three published-defaults boots, back to back (R1 with Tier 0 + vision smoke).
cd "$(dirname "$0")/../.." || exit 1
bash evidence/r2-tools/final.sh evidence/r2/r1 tier0
bash evidence/r2-tools/final.sh evidence/r2/r2
bash evidence/r2-tools/final.sh evidence/r2/r3
