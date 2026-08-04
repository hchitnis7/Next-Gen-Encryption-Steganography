#!/usr/bin/env bash
#
# Downloads the first N images from the ALASKA#2 uncompressed COLOR 512x512
# dataset (TIFF format, .tif extension), adapted from the official
# ALASKA_v2_RAWs_download_script.sh (see ALASKA2_documentation.pdf).
#
# Source: http://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_COLOR/
# License: CC BY-NC-ND (research/education only, no commercial use, credit
# required). Cite: Cogranne, Giboulot & Bas, "The ALASKA Steganalysis
# Challenge", ACM IH&MMSec 2019, or https://alaska.utt.fr
#
# Usage:
#   chmod +x download_alaska2_covers.sh
#   ./download_alaska2_covers.sh [N_IMAGES] [OUTPUT_DIR] [N_JOBS]
#
#   N_IMAGES   default 1000  (there are 80005 images total, 1..80005)
#   OUTPUT_DIR default ./alaska2_raw_color_512
#   N_JOBS     default 4     (concurrent download workers)

set -u

N_IMAGES="${1:-1000}"
Image_local_path="${2:-./alaska2_raw_color_512}/"
N_JOBS="${3:-4}"
DATASET_DIR="ALASKA_v2_TIFF_512_COLOR"   # swap to _256_COLOR or _VariousSize_COLOR if needed
BASE_URL="http://alaska.utt.fr/DATASETS/${DATASET_DIR}"

timeStart=$(date +%s)
mkdir -p "$Image_local_path"
mkdir -p ./tmp
LOG_FILE="./log_ALASKA_v2_downloads"
: > "$LOG_FILE"   # truncate log at start of run

echo "Downloading $N_IMAGES images from $BASE_URL"
echo "  -> $Image_local_path"
echo "  -> $N_JOBS concurrent workers"
echo "Log: $LOG_FILE"

MAX_ATTEMPTS=$((N_IMAGES * 2 + 500))   # allow for gaps in numbering, then give up

# ── Per-index download function, run in the background by the dispatch loop ──
# Each invocation gets its own tmp subdirectory so concurrent wget calls
# never collide on the same "./tmp/00042.tif" path.
_download_one() {
    local idx="$1"
    local i tmpdir imageName imageURL
    i=$(printf "%05d" "$idx")
    imageName="${i}.tif"
    imageURL="${BASE_URL}/${imageName}"

    if [ -f "${Image_local_path}${imageName}" ]; then
        return 0
    fi

    tmpdir="./tmp/worker_$$_$idx"
    mkdir -p "$tmpdir"

    # --no-check-certificate: alaska.utt.fr's TLS cert is currently expired
    # server-side (as of Jul 2026). Traffic is still HTTPS-encrypted, just
    # not validated against a CA. Fine for pulling public research files;
    # remove this flag once they renew their cert.
    if wget -c --no-check-certificate -P "$tmpdir/" "$imageURL" &>> "$LOG_FILE"; then
        mv "${tmpdir}/${imageName}" "${Image_local_path}${imageName}" 2>> "$LOG_FILE"
    fi
    rm -rf "$tmpdir"
    # Missing indices (404, gaps in numbering) are silently skipped — the
    # dataset numbering isn't perfectly contiguous (some raw conversions
    # failed for certain camera models per the ALASKA docs).
}
export -f _download_one
export Image_local_path BASE_URL LOG_FILE

n_collected=0
attempt_index=0
running_jobs=0

while [ "$n_collected" -lt "$N_IMAGES" ] && [ "$attempt_index" -lt "$MAX_ATTEMPTS" ]; do
    attempt_index=$((attempt_index + 1))

    # Skip dispatching a job for files already on disk (fast path, no subshell needed)
    i=$(printf "%05d" "$attempt_index")
    if [ -f "${Image_local_path}${i}.tif" ]; then
        n_collected=$((n_collected + 1))
        continue
    fi

    _download_one "$attempt_index" &
    running_jobs=$((running_jobs + 1))

    # Throttle to N_JOBS concurrent background downloads
    if [ "$running_jobs" -ge "$N_JOBS" ]; then
        wait -n            # wait for any one background job to finish
        running_jobs=$((running_jobs - 1))
    fi

    if [ $((attempt_index % 40)) -eq 0 ]; then
        wait                # sync up before recounting, avoids over/undercounting mid-flight jobs
        n_collected=$(find "$Image_local_path" -name "*.tif" | wc -l)
        currentTime=$(date +%s)
        echo "Collected $n_collected / $N_IMAGES (tried $attempt_index indices). Elapsed = $((currentTime - timeStart)) sec."
        running_jobs=0
    fi
done

wait   # let any still-running background jobs finish before final tally

n_ok=$(find "$Image_local_path" -name "*.tif" | wc -l)
if [ "$n_ok" -lt "$N_IMAGES" ]; then
    echo "WARNING: only found $n_ok images after trying $attempt_index indices (raised MAX_ATTEMPTS cap)."
fi

echo "Done. $n_ok / $N_IMAGES .tif files present in $Image_local_path"
echo "Point CFG['alaska2_dir'] at: $Image_local_path"