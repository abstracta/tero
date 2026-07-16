#!/bin/sh
set -e

INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
TERO_VERSION="${TERO_VERSION:-latest}"

GITLAB_URL="https://gitlab.abstracta.us"
GITLAB_PROJECT_ID="350"

if [ -z "$GITLAB_TOKEN" ]; then
  echo "Error: GITLAB_TOKEN is required"
  exit 1
fi

AUTH_HEADER="PRIVATE-TOKEN: ${GITLAB_TOKEN}"

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
  URL="${GITLAB_URL}/api/v4/projects/${GITLAB_PROJECT_ID}/packages/generic/tero-cli/${TERO_VERSION}/${BINARY}"

  echo "Downloading tero v${TERO_VERSION} (${OS}/${ARCH})..."

  HTTP_CODE=$(curl -fL --progress-bar \
    -H "${AUTH_HEADER}" \
    -o /tmp/tero-download \
    -w "%{http_code}" \
    "$URL")

  if [ "$HTTP_CODE" != "200" ]; then
    echo "Error: download failed (HTTP $HTTP_CODE)"
    exit 1
  fi

  mv /tmp/tero-download "${INSTALL_DIR}/tero"
  chmod +x "${INSTALL_DIR}/tero"
}

main() {
  detect_platform
  mkdir -p "$INSTALL_DIR"
  download

  echo ""
  echo "tero installed to ${INSTALL_DIR}/tero"
  echo "  version: $(${INSTALL_DIR}/tero --version 2>/dev/null || echo "$TERO_VERSION")"
  echo ""
  echo "Run 'tero --help' to get started."
}

main
