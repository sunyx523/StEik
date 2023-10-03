#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
cd $DIR/data/
# wget google drive file as per https://stackoverflow.com/a/58914589

# get SRB data (1.12GB) (https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view)
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe" -O deep_geometric_prior_data.zip && rm -rf /tmp/cookies.txt
unzip deep_geometric_prior_data.zip