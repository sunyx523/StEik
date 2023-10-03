#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
cd $DIR/data/
# wget google drive file as per https://stackoverflow.com/a/58914589

# Get Scene from SIREN (56.2MB) (https://drive.google.com/file/d/13X1UlMsnbh3dcV4tJysVDgzg6kYyxHhb/view?usp=sharing)
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13X1UlMsnbh3dcV4tJysVDgzg6kYyxHhb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13X1UlMsnbh3dcV4tJysVDgzg6kYyxHhb" -O scene_reconstruction.tar.xz && rm -rf /tmp/cookies.txt
tar -xJvf scene_reconstruction.tar.xz