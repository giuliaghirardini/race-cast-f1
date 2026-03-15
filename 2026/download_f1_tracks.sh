#!/bin/bash

BASE="https://www.formula1.com/etc/designs/fom-website/images/racing/2026"

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

for i in "${!countries[@]}"; do

    num=$((i+1))
    folder="${num}-${countries[$i]}"
    city="${cities[$i]}"

    img="2026track${city}detailed.avif"

    echo "Downloading $img -> $folder"

    curl -L "$BASE/$img" -o "$folder/$img"

done