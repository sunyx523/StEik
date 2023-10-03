# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your path
DATASET_PATH=$DIR'/data/scene_reconstruction/' # change to your dataset path
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
echo "If $DATASET_PATH is not the correct path for the scene-recon dataset, change the variable DATASET_PATH"

cd $DIR/surface_reconstruction/ # To call python scripts correctly

LOGDIR='./log/scene/' #change to your desired log directory
IDENTIFIER='StEik'
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used
FILENAME='interior_room.ply'


### MODEL HYPER-PARAMETERS ###
##############################
LAYERS=8
DECODER_HIDDEN_DIM=256
NL='sine' # 'sine' | 'relu' | 'softplus'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='siren' #siren | geometric_sine | geometric_relu | mfgi
NEURON_TYPE='quadratic' #linear | quadratic
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(5e3 1e2 1e2 5e1 1e1)
DIV_TYPE='dir_l1' # 'dir_l1' | 'dir_l2' | 'full_l1' | 'full_l2'
DIV_DECAY='linear' # 'linear' | 'quintic' | 'step'
DECAY_PARAMS=(1e1 0.1 1e1 0.3 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000
### TRAINING HYPER-PARAMETERS ###
#################################
NITERATIONS=100000
GPU=0
LR=8e-6
GRAD_CLIP_NORM=10.0

python3 train_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --grid_res $GRID_RES --loss_type $LOSS_TYPE --gpu_idx $GPU --n_iterations $NITERATIONS --n_points $NPOINTS --lr $LR --nonmnfld_sample_type ${NONMNFLD_SAMPLE_TYPE} --dataset_path $DATASET_PATH --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim $DECODER_HIDDEN_DIM --div_decay $DIV_DECAY  --div_decay_params ${DECAY_PARAMS[@]} --div_type ${DIV_TYPE} --init_type ${INIT_TYPE} --neuron_type ${NEURON_TYPE} --nl ${NL} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}
python3 test_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --export_mesh 1 --dataset_path $DATASET_PATH --grid_res $TEST_GRID_RES --gpu_idx $GPU
