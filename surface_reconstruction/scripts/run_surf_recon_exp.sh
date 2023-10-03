# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your path
DATASET_PATH=$DIR'/data/deep_geometric_prior_data/' # change to your dataset path
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
echo "If $DATASET_PATH is not the correct path for the SRB dataset, change the variable DATASET_PATH"

cd $DIR/surface_reconstruction/ # To call python scripts correctly


LOGDIR='./log/SRB/' #change to your desired log directory
IDENTIFIER='StEik'
mkdir -p $LOGDIR
FILE=`basename "$0"`
SCAN_PATH=$DATASET_PATH'/scans/'
MODEL_DIR=$DIR'/models'

### MODEL HYPER-PARAMETERS ###
##############################
LAYERS=5
DECODER_HIDDEN_DIM=128
NL='sine' # 'sine' | 'relu' | 'softplus'
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='mfgi' #siren | geometric_sine | geometric_relu | mfgi
NEURON_TYPE='quadratic' #linear | quadratic
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(2e3 1e2 1e2 5e1 1e2)
DIV_TYPE='dir_l1' # 'dir_l1' | 'dir_l2' | 'full_l1' | 'full_l2'
DIV_DECAY='linear' # 'linear' | 'quintic' | 'step'
DECAY_PARAMS=(1e2 0.2 1e2 0.4 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000
### TRAINING HYPER-PARAMETERS ###
#################################
NITERATIONS=10000
GPU=0
LR=1e-4
GRAD_CLIP_NORM=10.0


for FILENAME in 'gargoyle.ply' 'daratech.ply' 'lord_quas.ply' 'anchor.ply' 'dc.ply'
do
  python3 train_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --grid_res $GRID_RES --loss_type $LOSS_TYPE --gpu_idx $GPU --n_iterations $NITERATIONS --n_points $NPOINTS  --lr ${LR} --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE --dataset_path $SCAN_PATH --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim ${DECODER_HIDDEN_DIM} --div_decay $DIV_DECAY --div_decay_params ${DECAY_PARAMS[@]} --div_type $DIV_TYPE --init_type ${INIT_TYPE} --neuron_type ${NEURON_TYPE} --nl ${NL}  --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}
  python3 test_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --export_mesh 1 --dataset_path $SCAN_PATH --grid_res $TEST_GRID_RES --gpu_idx $GPU
done

RESULTS_DIR=$LOGDIR$IDENTIFIER'/result_meshes'
python3 compute_metrics_srb.py --logdir $LOGDIR$IDENTIFIER --dataset_path $DATASET_PATH --results_path $RESULTS_DIR