#!/bin/bash

read -rp "Enter a number between 1 and 50(default is seq01): " input
input=${input:-1}
# Check if input is a number between 1 and 50
if [[ $input =~ ^[1-50] ]]; then
    if [[ $input -lt 10 ]]; then
        res="0$input"
    else
        res="$input"
    fi
    echo "padded input is ${res}"
else
    echo "Input must be a number between 1 and 50."
fi 


build/bin/dso_dataset files=TUM-MONO/sequence_$res/images.zip \
	calib=TUM-MONO/sequence_$res/camera.txt \
	gamma=TUM-MONO/sequence_$res/pcalib.txt \
	vignette=TUM-MONO/sequence_$res/vignette.png \
	preset=0 \
	mode=0
