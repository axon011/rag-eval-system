#!/usr/bin/env bash
# Stop the demo server cleanly.

set -euo pipefail

PORT="${PORT:-8002}"

if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -Command "
    \$c = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if (\$c) {
      foreach (\$conn in \$c) {
        Write-Output \"    killing PID \$(\$conn.OwningProcess)\"
        Stop-Process -Id \$conn.OwningProcess -Force -ErrorAction SilentlyContinue
      }
    } else {
      Write-Output \"    nothing listening on port $PORT\"
    }
  "
else
  pkill -f "uvicorn app.main:app" || echo "    no uvicorn process found"
fi
