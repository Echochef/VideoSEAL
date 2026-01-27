#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for `scripts/extract_srt.py`.
#
# Usage:
#   ./extract_srt.sh --video /path/to/video.mp4 --output-dir /path/to/out
# or env-driven:
#   VIDEO=/path/to/video.mp4 OUTPUT_DIR=/path/to/out ./extract_srt.sh
#
# Notes:
# - This script is optional; `Videoseal` typically calls `scripts/extract_srt.py` directly.
# - All options are forwarded to `scripts/extract_srt.py`.

# Required: video path (absolute or relative)
VIDEO="${VIDEO:-}"

# Optional: OCR language code (e.g., ch/en/japan/korean/all)
LANGUAGE="all"

# Optional: mode fast|auto|accurate (default: auto)
MODE="auto"

# Optional: subtitle region (ymin, ymax, xmin, xmax)
AREA_YMIN=""
AREA_YMAX=""
AREA_XMIN=""
AREA_XMAX=""

# Optional: override drop-score threshold (default in config: 0.75)
DROP_SCORE=""
# Optional: subtitle-area deviation rate (e.g., 0.05~0.2)
SUB_AREA_DEVIATION_RATE="0.12"

# Optional: word segmentation (true/false)
WORD_SEGMENTATION=true

# Optional: generate txt (true/false)
GENERATE_TXT=false

# Optional: keep empty timestamps (true/false)
KEEP_EMPTY_TS=false

# Optional: interface language (logs only)
INTERFACE="English"

# Optional: VSF defaults
USE_BOTTOM_VSF=true

# Optional: bottom region ratio (0..1)
BOTTOM_RATIO="0.25"

# Optional: output directory (SRT/TXT). If empty, output to the video directory.
OUTPUT_DIR="${OUTPUT_DIR:-}"


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ARGS=(
  "${@}"
)

# Allow env-driven usage when not passing CLI args.
if [[ "${#ARGS[@]}" -eq 0 ]]; then
  if [[ -z "${VIDEO}" ]]; then
    echo "[ERR] Missing VIDEO. Usage: VIDEO=/path/to/video.mp4 OUTPUT_DIR=/path/to/out ./extract_srt.sh" >&2
    exit 2
  fi
  ARGS=(
    --video "$VIDEO"
    --language "$LANGUAGE"
    --mode "$MODE"
    --interface "$INTERFACE"
  )

  if [[ -n "${AREA_YMIN}" && -n "${AREA_YMAX}" && -n "${AREA_XMIN}" && -n "${AREA_XMAX}" ]]; then
    ARGS+=( --area "${AREA_YMIN}" "${AREA_YMAX}" "${AREA_XMIN}" "${AREA_XMAX}" )
  fi

  if [[ -n "${DROP_SCORE}" ]]; then
    ARGS+=( --drop-score "${DROP_SCORE}" )
  fi

  if [[ -n "${SUB_AREA_DEVIATION_RATE}" ]]; then
    ARGS+=( --sub-area-deviation-rate "${SUB_AREA_DEVIATION_RATE}" )
  fi

  if [[ "${WORD_SEGMENTATION}" != true ]]; then
    ARGS+=( --no-word-seg )
  fi

  if [[ "${GENERATE_TXT}" == true ]]; then
    ARGS+=( --gen-txt )
  fi

  if [[ "${KEEP_EMPTY_TS}" == true ]]; then
    ARGS+=( --keep-empty-ts )
  fi

  if [[ "${USE_BOTTOM_VSF}" == true ]]; then
    ARGS+=( --vsf-use-bottom-default true )
  else
    ARGS+=( --vsf-use-bottom-default false )
  fi

  if [[ -n "${BOTTOM_RATIO}" ]]; then
    ARGS+=( --vsf-bottom-ratio "${BOTTOM_RATIO}" )
  fi

  if [[ -n "${OUTPUT_DIR}" ]]; then
    ARGS+=( --output-dir "${OUTPUT_DIR}" )
  fi

  ARGS+=( --filter-by-area true )
fi

exec python3 "$SCRIPT_DIR/scripts/extract_srt.py" "${ARGS[@]}"
