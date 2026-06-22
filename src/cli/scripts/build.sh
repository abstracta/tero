#!/bin/sh
set -e

VERSION="${1:-dev}"
OUTDIR="dist/bin"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "Building tero CLI v${VERSION}..."

bunx esbuild src/index.ts --bundle --platform=node --format=esm --outfile=dist/tero-bundle.mjs --external:canvas --define:process.env.TERO_CLI_VERSION=\""${VERSION}"\"

bun build dist/tero-bundle.mjs --compile --target=bun-linux-x64 --outfile="${OUTDIR}/tero-linux-x64"
bun build dist/tero-bundle.mjs --compile --target=bun-linux-arm64 --outfile="${OUTDIR}/tero-linux-arm64"
bun build dist/tero-bundle.mjs --compile --target=bun-darwin-x64 --outfile="${OUTDIR}/tero-darwin-x64"
bun build dist/tero-bundle.mjs --compile --target=bun-darwin-arm64 --outfile="${OUTDIR}/tero-darwin-arm64"
bun build dist/tero-bundle.mjs --compile --target=bun-windows-x64 --outfile="${OUTDIR}/tero-windows-x64.exe"

rm -f dist/tero-bundle.mjs

echo "Done. Binaries in ${OUTDIR}/"
ls -lh "$OUTDIR"
