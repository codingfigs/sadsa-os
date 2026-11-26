#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="$ROOT_DIR/requirements.txt"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"
LAUNCH=1

for arg in "$@"; do
  case "$arg" in
    --install-only) LAUNCH=0 ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Install Python 3.10+ and try again." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
  echo "Tkinter is missing from $PYTHON_BIN. Install python3-tk or the Tkinter package for your distribution." >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

REQ_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$REQ_HASH" ]; then
  echo "Installing dependencies..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
else
  echo "Dependencies are already installed."
fi

DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
if [ -d "$DESKTOP_DIR" ]; then
  cat > "$DESKTOP_DIR/sadsa.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SADSA
Comment=Software Application for Data Science and Analytics
Exec="$VENV_DIR/bin/python" "$ROOT_DIR/pyfda.py"
Path="$ROOT_DIR"
Icon=$ROOT_DIR/icon.ico
Terminal=false
Categories=Science;Education;
EOF
  chmod +x "$DESKTOP_DIR/sadsa.desktop"
fi

echo "SADSA setup is ready."

if [ "$LAUNCH" -eq 1 ]; then
  cd "$ROOT_DIR"
  exec "$VENV_DIR/bin/python" "$ROOT_DIR/pyfda.py"
fi
