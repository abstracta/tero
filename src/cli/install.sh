#!/bin/sh
set -e

INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
REPO="abstracta/tero"

detect_platform() {
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)

  case "$OS" in
    linux) OS="linux" ;;
    darwin) OS="darwin" ;;
    mingw*|msys*|cygwin*) OS="windows" ;;
    *)
      echo "Error: unsupported OS: $OS"
      exit 1
      ;;
  esac

  case "$ARCH" in
    x86_64|amd64) ARCH="x64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *)
      echo "Error: unsupported architecture: $ARCH"
      exit 1
      ;;
  esac

  BINARY="tero-${OS}-${ARCH}"
  if [ "$OS" = "windows" ]; then
    BINARY="${BINARY}.exe"
  fi
}

download() {
  URL="https://github.com/${REPO}/releases/latest/download/${BINARY}"

  echo "Downloading the latest tero (${OS}/${ARCH})..."

  curl -fL --progress-bar -o "${INSTALL_DIR}/tero" "$URL"
  chmod +x "${INSTALL_DIR}/tero"
}

main() {
  detect_platform
  mkdir -p "$INSTALL_DIR"
  download

  echo ""
  echo "tero installed to ${INSTALL_DIR}/tero"
  echo "  version: $(${INSTALL_DIR}/tero --version 2>/dev/null || echo latest)"
  echo ""
  echo "Run 'tero --help' to get started."
}

main
