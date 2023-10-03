#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
cd $DIR/sanitychecks/ # To call python scripts correctly

LOGDIR='./log/' #change to your desired log directory
IDENTIFIER='my_experiment'
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used


### MODEL HYPER-PARAMETERS ###
##############################
LAYERS=4
DECODER_HIDDEN_DIM=128
NL='sine' # 'sine' | 'relu' | 'softplus'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='mfgi' #siren | geometric_sine | geometric_relu | mfgi
NEURON_TYPE='quadratic' #linear | quadratic
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e2)
DIV_TYPE='dir_l1' # 'dir_l1' | 'dir_l2' | 'full_l1' | 'full_l2'
DIVDECAY='linear' # 'linear' | 'quintic' | 'step'
DECAY_PARAMS=(1e2 0.2 1e2 0.4 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000
### TRAINING HYPER-PARAMETERS ###
#################################
NSAMPLES=10000
BATCH_SIZE=1
GPU=0
NEPOCHS=1
EVALUATION_EPOCH=0
LR=5e-5
GRAD_CLIP_NORM=10.0
### TESTING ARGUMENTS ###
#################################
EPOCHS_N_EVAL=($(seq 0 100 9900)) # use this to generate images of different iterations

for SHAPE in 'L'
do
  LOGDIR=${LOGDIRNAME}${NONMNFLD_SAMPLE_TYPE}'_sampling_'${GRID_RES}'/'${SHAPE}'/'${IDENTIFIER}'/'
  python3 train_basic_shape.py --logdir $LOGDIR --shape_type $SHAPE --grid_res $GRID_RES --loss_type $LOSS_TYPE --inter_loss_type 'exp' --num_epochs $NEPOCHS --gpu_idx $GPU --n_samples $NSAMPLES --n_points $NPOINTS --batch_size $BATCH_SIZE --lr ${LR} --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE --decoder_n_hidden_layers $LAYERS  --decoder_hidden_dim $DECODER_HIDDEN_DIM --div_decay $DIVDECAY --div_decay_params ${DECAY_PARAMS[@]} --div_type $DIV_TYPE --init_type ${INIT_TYPE} --neuron_type ${NEURON_TYPE} --nl ${NL} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}
  python3 test_basic_shape.py --logdir $LOGDIR --gpu_idx $GPU --epoch_n "${EPOCHS_N_EVAL[@]}"
done
