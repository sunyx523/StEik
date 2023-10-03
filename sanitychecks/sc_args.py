# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import argparse
import torch
import os
import numpy as np

def add_args(parser):

    parser.add_argument('--gpu_idx', type=int, default=1, help='set < 0 to use CPU')
    parser.add_argument('--logdir', type=str, default='./log/siren_square', help='log directory')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--shape_type', type=str, default='L', help='shape dataset to load circle | square | L ')

    parser.add_argument('--n_samples', type=int, default=10000,
                        help='number of samples in the generated train and test set')
    parser.add_argument('--n_point_total', type=int, default=30000,
                        help='the total number of points sampled on the manifold')
    parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch, '
                                                                    '0 for training from scratch')
    parser.add_argument('--model_dirpath', type=str,
                        default='/mnt/3.5TB_WD/PycharmProjects/ZeroCurlShapes/models',
                        help='path to model directory for backup')
    parser.add_argument('--parallel', type=int, default=False, help='use data parallel')

    # training paramaeters
    parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--n_points', type=int, default=15000, help='number of points in each point cloud')
    # parser.add_argument('--delta', type=float, default=0.25, help='clamping value')

    # Network architecture and loss
    parser.add_argument('--decoder_hidden_dim', type=int, default=128, help='length of decoder hidden dim')
    parser.add_argument('--encoder_hidden_dim', type=int, default=128, help='length of encoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of layers in the decoder')
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu')
    parser.add_argument('--latent_size', type=int, default=0, help='number of elements in the latent vector')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 1.0],
                    help='radius and scaling')
    parser.add_argument('--neuron_type', type=str, default='quadratic', help='type of neuron')

    parser.add_argument('--normalize_normal_loss', type=int, default=False, help='normal loss normalization flag')
    parser.add_argument('--unsigned_n', type=int, default=True, help='flag for unsigned normal loss')
    parser.add_argument('--unsigned_d', type=int, default=False, help='flag for unsigned distance loss')

    parser.add_argument('--encoder_type', type=str, default='none', help='type of encoder None | pointnet')
    parser.add_argument('--export_vis', type=int, default=False, help='export levelset visualization while training')

    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_div', help='loss type to use: siren | siren_wo_n | siren_wo_n_w_div | siren_wo_n_w_div_w_reg')
    parser.add_argument('--n_loss_type', type=str, default='cos', help='type of normal los cos | ')
    parser.add_argument('--inter_loss_type', type=str, default='exp', help='type of inter los exp | unsigned_diff | signed_diff')
    parser.add_argument('--sampler_prob', type=str, default='none',
                        help='type of sampler probability for non manifold points on the grid none | div | curl')

    parser.add_argument('--div_type', type=str, default='dir_l1', help='divergence term norm dir_l1 | dir_l2 | full_l1 | full_l2')
    parser.add_argument('--grid_res', type=int, default=256,
                        help='uniform grid resolution')
    parser.add_argument('--div_clamp', type=float, default=50, help='divergence clamping value')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='grid',
                        help='how to sample points off the manifold - grid | gaussian | combined')
    parser.add_argument('--div_decay', type=str, default='none', help='divergence term importance decay none | step | linear')
    parser.add_argument('--div_decay_params', nargs='+', type=float, default=[0.0, 0.5, 0.75],
                        help='epoch number to evaluate')
    parser.add_argument('--init_type', type=str, default='mfgi',
                        help='initialization type siren | geometric_sine | geometric_relu |pt_circle | mfgi')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[3e3, 1e2, 1e2, 5e1, 1e1],
                        help='loss terms weights sdf | inter | normal | eikonal | div')
    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Local implicit functions experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser(description='Local implicit functions test experiment')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--logdir', type=str, default='./log/siren_square/', help='log directory')
    parser.add_argument('--n_points', type=int, default=0, help='number of points in each point cloud, '
                                                                  'if 0 use training options')
    parser.add_argument('--batch_size', type=int, default=0, help='number of samples in a minibatch, if 0 use training')
    parser.add_argument('--grid_res', type=int, default=256, help='evaluation grid resolution')
    parser.add_argument('--epoch_n', type=int, nargs='+', default=np.arange(0, 10000, 100).tolist(), help='epoch number to evaluate')
    # parser.add_argument('--epoch_n', type=int, default=29900, help='epoch to evaluate')
    parser.add_argument('--std', type=float, default=0.12, help='noise standard deviation')
    parser.add_argument('--noise_type', type=str, default='gaussian', help='type of noise to inject the data')
    parser.add_argument('--normal_vis', type=int, default=False, help='flag to visualize normals in plot')
    parser.add_argument('--n_example_plots', type=int, default=1, help='number of examples to plot')
    test_opt = parser.parse_args()

    param_filename = os.path.join(test_opt.logdir, 'trained_models/', 'params.pth')
    train_opt = torch.load(param_filename)

    test_opt.nl, test_opt.latent_size, test_opt.encoder_type, test_opt.n_samples, test_opt.seed, \
        test_opt.decoder_hidden_dim, test_opt.encoder_hidden_dim, test_opt.n_loss_type,\
        test_opt.normalize_normal_loss, test_opt.unsigned_n, test_opt.unsigned_d, test_opt.loss_type, \
        test_opt.model_dirpath, test_opt.inter_loss_type, test_opt.shape_type, \
        test_opt.div_decay,  test_opt.div_type, test_opt.div_clamp, test_opt.decoder_n_hidden_layers, \
         = train_opt.nl, train_opt.latent_size, train_opt.encoder_type,  train_opt.n_samples,  train_opt.seed,\
          train_opt.decoder_hidden_dim, train_opt.encoder_hidden_dim, train_opt.n_loss_type,  \
          train_opt.normalize_normal_loss, train_opt.unsigned_n, train_opt.unsigned_d, train_opt.loss_type, \
          train_opt.model_dirpath, train_opt.inter_loss_type, train_opt.shape_type,\
          train_opt.div_decay, train_opt.div_type, train_opt.div_clamp, train_opt.decoder_n_hidden_layers

    test_opt.n_point_total = train_opt.n_points

    if test_opt.n_points == 0:
        test_opt.n_points = train_opt.n_points
    if test_opt.batch_size == 0:
        test_opt.batch_size = train_opt.batch_size
    if "parallel" in train_opt:
        test_opt.parallel = train_opt.parallel
    else:
        test_opt.parallel = False
    if 'init_type' in train_opt:
        test_opt.init_type = train_opt.init_type
    else:
        test_opt.init_type = 'siren'
    test_opt.nonmnfld_sample_type = 'grid'
    return test_opt
