#!/bin/bash

races=(
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

i=1

for race in "${races[@]}"; do
    folder="${i}-${race}"
    mkdir -p "$folder"
    echo "Created $folder"
    ((i++))
done