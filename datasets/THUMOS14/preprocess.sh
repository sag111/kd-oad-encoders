#! /bin/bash
# Create directories for videos
mkdir -p processed/
mkdir -p processed/videos/
mkdir -p processed/validation/
mkdir -p processed/test/

# Directory arrays
INPUT_DIRS=("raw/validation" "raw/test" "raw/videos")
OUTPUT_DIRS=("processed/validation" "processed/test" "processed/videos/")

# Counters
total_processed=0
total_success=0
total_errors=0
total_skipped=0

# Iterate over directories
for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIR="${INPUT_DIRS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    echo "Processing directory: $INPUT_DIR, output directory: $OUTPUT_DIR"

    # Create the output directory if it does not exist
    mkdir -p "$OUTPUT_DIR"

    # Count the total number of files
    total_files=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.webm" \) | wc -l)
    current_file=0
    echo "Files found: $total_files"

    # Iterate over all video files
    for file in "$INPUT_DIR"/*.{avi,mov,mkv,mp4,flv,wmv,webm}; do
        [ -e "$file" ] || continue

        filename=$(basename "$file")
        name="${filename%.*}"

        ((current_file++))

        # Check whether the file has already been converted
        if [ -f "$OUTPUT_DIR/${name}.mp4" ]; then
            echo "[$current_file/$total_files] Skipping (already converted): $filename"
            ((total_skipped++))
            ((total_processed++))
            echo ""
            continue
        fi

        echo "[$current_file/$total_files] Converting: $filename"

        ffmpeg -i "$file" \
            -vf "scale=224:224:force_original_aspect_ratio=increase,crop=224:224,fps=24" \
            -c:v libx264 \
            -crf 23 \
            -preset medium \
            -an \
            "$OUTPUT_DIR/tmp_${name}.mp4" \
            -loglevel error -stats -y

        if [ $? -eq 0 ]; then
            mv "$OUTPUT_DIR/tmp_${name}.mp4" "$OUTPUT_DIR/${name}.mp4"
            echo "✓ Done: ${name}.mp4"
            ((total_success++))
        else
            rm -f "$OUTPUT_DIR/tmp_${name}.mp4"
            echo "✗ Conversion error: $filename"
            ((total_errors++))
        fi
        echo ""
        ((total_processed++))
    done
    echo "Finished processing: $INPUT_DIR"
done
echo "Files converted! Total: $total_processed, Successful: $total_success, Skipped: $total_skipped, Errors: $total_errors"