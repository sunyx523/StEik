#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
cd $DIR/data/
# wget google drive file as per https://stackoverflow.com/a/58914589

# Get preprocessed subset of ShapeNet data (370.96MB) (https://drive.google.com/file/d/14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0/view?usp=sharing)
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0" -O NSP_data.tar.gz && rm -rf /tmp/cookies.txt
tar -xzvf NSP_data.tar.gz

# Get .ply point tree files for ShapeNet subset (412.80MB) (https://drive.google.com/file/d/1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf/view?usp=sharing)
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf" -O ShapeNetNSP.tar.gz && rm -rf /tmp/cookies.txt
tar -xzvf ShapeNetNSP.tar.gz