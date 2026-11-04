#!/usr/bin/env bash
echo "=== raw qstat -f 26665306 (head 40) ==="
qstat -f 26665306 2>&1 | head -40