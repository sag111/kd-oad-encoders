#! /bin/bash
# Create directories for videos
mkdir -p processed/
mkdir -p processed/train
mkdir -p processed/val/
mkdir -p processed/test/

RAW_DIR="raw"

# File lists (names without extensions); source files are in raw/*.mkv
TRAIN_FILES=(
    "Breaking_Bad_ep1" "How_I_Met_Your_Mother_ep1" "How_I_Met_Your_Mother_ep2"
    "How_I_Met_Your_Mother_ep3" "How_I_Met_Your_Mother_ep4" "Modern_Family_ep1"
    "Modern_Family_ep6" "Modern_Family_ep2" "Modern_Family_ep3"
    "Mad_Men_ep1" "Sons_of_Anarchy_ep1" "24_ep1" "24_ep2"
)
VAL_FILES=(
    "Breaking_Bad_ep2" "How_I_Met_Your_Mother_ep5" "How_I_Met_Your_Mother_ep6"
    "Modern_Family_ep4" "Mad_Men_ep2" "Sons_of_Anarchy_ep2" "24_ep3"
)
TEST_FILES=(
    "Breaking_Bad_ep3" "How_I_Met_Your_Mother_ep7" "How_I_Met_Your_Mother_ep8"
    "Modern_Family_ep5" "Mad_Men_ep3" "Sons_of_Anarchy_ep3" "24_ep4"
)

total_processed=0
total_success=0
total_errors=0

process_list() {
    local -n names=$1
    local out_dir=$2
    local total=${#names[@]}
    local current=0
    for name in "${names[@]}"; do
        ((current++))
        local src="$RAW_DIR/${name}.mkv"
        if [ ! -f "$src" ]; then
            echo "Skipping [$current/$total]: file not found: $src"
            ((total_errors++))
            continue
        fi
        local dst="$out_dir/${name}.mp4"
        if [ -f "$dst" ]; then
            echo "Skipping [$current/$total]: ${name}.mp4 already exists"
            echo ""
            continue
        fi
        local tmp_dst="$out_dir/tmp_${name}.mp4"
        echo "[$current/$total] Converting: ${name}.mkv"
        if ffmpeg -i "$src" \
            -vf "scale=224:224:force_original_aspect_ratio=increase,crop=224:224,fps=24" \
            -c:v libx264 \
            -crf 23 \
            -preset medium \
            -an \
            "$tmp_dst" \
            -loglevel error -stats -y; then
            mv "$tmp_dst" "$dst"
            echo "Done: ${name}.mp4"
            ((total_success++))
        else
            rm -f "$tmp_dst"
            echo "Conversion error: ${name}.mkv"
            ((total_errors++))
        fi
        ((total_processed++))
        echo ""
    done
}

echo "=== Train ==="
process_list TRAIN_FILES "processed/train"
echo "=== Val ==="
process_list VAL_FILES "processed/val"
echo "=== Test ==="
process_list TEST_FILES "processed/test"

echo "Files converted! Processed: $total_processed, Successful: $total_success, Errors: $total_errors"
