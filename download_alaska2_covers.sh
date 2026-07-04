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
#   ./download_alaska2_covers.sh [N_IMAGES] [OUTPUT_DIR]
#
#   N_IMAGES   default 1000  (there are 80005 images total, 1..80005)
#   OUTPUT_DIR default ./alaska2_raw_color_512

set -u

N_IMAGES="${1:-1000}"
Image_local_path="${2:-./alaska2_raw_color_512}/"
DATASET_DIR="ALASKA_v2_TIFF_512_COLOR"   # swap to _256_COLOR or _VariousSize_COLOR if needed
BASE_URL="http://alaska.utt.fr/DATASETS/${DATASET_DIR}"

timeStart=$(date +%s)
mkdir -p "$Image_local_path"
mkdir -p ./tmp

echo "Downloading $N_IMAGES images from $BASE_URL"
echo "  -> $Image_local_path"
echo "Log: ./log_ALASKA_v2_downloads"

n_collected=0
attempt_index=0
MAX_ATTEMPTS=$((N_IMAGES * 2 + 500))   # allow for gaps in numbering, then give up

while [ "$n_collected" -lt "$N_IMAGES" ] && [ "$attempt_index" -lt "$MAX_ATTEMPTS" ]; do
    attempt_index=$((attempt_index + 1))
    i=$(printf "%05d" "$attempt_index")
    imageName="${i}.tif"
    imageURL="${BASE_URL}/${imageName}"

    if [ -f "${Image_local_path}${imageName}" ]; then
        n_collected=$((n_collected + 1))
        continue
    fi

    # --no-check-certificate: alaska.utt.fr's TLS cert is currently expired
    # server-side (as of Jul 2026). Traffic is still HTTPS-encrypted, just
    # not validated against a CA. Fine for pulling public research files;
    # remove this flag once they renew their cert.
    if ( wget -c --no-check-certificate -P ./tmp/ "$imageURL" && mv "./tmp/${imageName}" "${Image_local_path}${imageName}" ) &>> ./log_ALASKA_v2_downloads; then
        n_collected=$((n_collected + 1))
    fi
    # Missing indices (404, gaps in numbering) are silently skipped — the
    # dataset numbering isn't perfectly contiguous (some raw conversions
    # failed for certain camera models per the ALASKA docs).

    if [ $((attempt_index % 10)) -eq 0 ]; then
        currentTime=$(date +%s)
        echo "Collected $n_collected / $N_IMAGES (tried $attempt_index indices). Elapsed = $((currentTime - timeStart)) sec."
    fi
done

if [ "$n_collected" -lt "$N_IMAGES" ]; then
    echo "WARNING: only found $n_collected images after trying $attempt_index indices (raised MAX_ATTEMPTS cap)."
fi

n_ok=$(find "$Image_local_path" -name "*.tif" | wc -l)
echo "Done. $n_ok / $N_IMAGES .tif files present in $Image_local_path"
echo "Point CFG['alaska2_dir'] at: $Image_local_path"