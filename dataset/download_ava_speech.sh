#!/bin/bash

OUT_DIR="ava_speech"
mkdir -p $OUT_DIR
LABEL_FILE="ava_speech_labels_v1.csv"

# make sure youtube-dl is installed
if ! command -v yt-dlp &> /dev/null
then
    echo "Installing youtube-dl"
    sudo -H pip install --upgrade yt-dlp
fi

# make sure labels are found
if ! [ -f "$OUT_DIR/$LABEL_FILE" ]; then
    echo "Downloading AVA-Speech labels"
    curl https://research.google.com/ava/download/ava_speech_labels_v1.csv --output $OUT_DIR/$LABEL_FILE
fi

# download files
yt_ids=($(cat $OUT_DIR/$LABEL_FILE | cut -d, -f1 | uniq))
echo "Downloading data from YouTube (only mono audio (00:15:00 till 00:30:00))"
for id in "${yt_ids[@]}"; do
    # download file if not found in $OUT_DIR
    if ! [ -f "$OUT_DIR/$id.wav" ]; then
        echo -e "\n\tDownloading $id..."
        video_URL="https://www.youtube.com/watch?v=$id"
        ffmpeg -ss 900 -to 1800 -i $(yt-dlp -f 139 --get-url $video_URL) -ac 1 $OUT_DIR/$id.wav
    else
        echo -e "\t$id exists."
    fi
done