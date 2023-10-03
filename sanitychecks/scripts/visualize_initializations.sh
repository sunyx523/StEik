#!/bin/bash
# run from DiGS/sanitychecks/ , i.e. `./scripts/visualize_initializations.sh`
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"
cd $DIGS_DIR/sanitychecks/ # To call python scripts correctly

python3 initialization_visualization.py --nl 'sine' --init_type 'mfgi'
python3 initialization_visualization.py --nl 'sine' --init_type 'geometric_sine'
python3 initialization_visualization.py --nl 'relu' --init_type 'geometric_relu' --decoder_hidden_dim 512 --decoder_n_hidden_layers 8 #folowing IGR specifications
python3 initialization_visualization.py --nl 'sine' --init_type 'siren'