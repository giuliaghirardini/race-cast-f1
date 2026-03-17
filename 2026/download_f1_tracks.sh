#!/bin/bash

# BASE="https://www.formula1.com/etc/designs/fom-website/images/racing/2026"

# Alternative: Download from media.formula1.com instead
BASE="https://media.formula1.com/image/upload/c_fit,h_704/q_auto/v1740000001/common/f1/2026/track"

countries=(
    "australia"
    "china"
    "japan"
    "bahrain"
    "saudi-arabia"
    "miami"
    "canada"
    "monaco"
    "spain-barcelona"
    "austria"
    "great-britain"
    "belgium"
    "hungary"
    "netherlands"
    "italy"
    "spain-madrid"
    "azerbaijan"
    "singapore"
    "usa-austin"
    "mexico"
    "brazil"
    "las-vegas"
    "qatar"
    "abu-dhabi"
)

cities=(
    melbourne
    shanghai
    suzuka
    sakhir
    jeddah
    miami
    montreal
    monaco
    barcelona
    spielberg
    silverstone
    spa
    budapest
    zandvoort
    monza
    madrid
    baku
    singapore
    austin
    mexicocity
    saopaulo
    lasvegas
    lusail
    abudhabi
)

echo "Downloading track images for 2026 F1 season..."
for i in "${!countries[@]}"; do

    num=$((i+1))
    folder="${num}-${countries[$i]}"
    city="${cities[$i]}"

    # img="2026track${city}detailed.avif"
    img="2026track${city}detailed.webp"

    echo "Downloading $img -> $folder"

    curl -L "$BASE/$img" -o "$folder/$img"

done

echo "Deleting avif files if they exist..."
for i in "${!countries[@]}"; do

    num=$((i+1))
    folder="${num}-${countries[$i]}"
    city="${cities[$i]}"

    echo "Deleting $img -> $folder"
    rm -f "$folder/2026track${city}detailed.avif"

done