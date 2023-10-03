# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import argparse
import torch
import os


def get_train_args():
    parser = argparse.ArgumentParser(description='Local implicit functions experiment')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--logdir', type=str, default='./log/debug', help='log directory')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--dataset_path', type=str, default='../data/deep_geometric_prior_data', help='path to dataset folder')
    parser.add_argument('--raw_dataset_path', type=str, default='../data/deep_geometric_prior_data', help='path to dataset folder')
    parser.add_argument('--file_name', type=str, default='lord_quas.ply',
                        help='name of file to reconstruct (within the dataset path)')
    parser.add_argument('--n_iterations', type=int, default=5000,
                        help='number of interations in the generated train and test set')
    parser.add_argument('--model_dirpath', type=str,
                        default='./models',
                        help='path to model directory for backup')
    parser.add_argument('--parallel', type=int, default=False, help='use data parallel')
    parser.add_argument('--results_path', type=str, default='./log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes',
                        help='path to results directory')

    # training paramaeters
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')
    parser.add_argument('--n_points', type=int, default=30000, help='number of points in each point cloud')

    # Network architecture and loss
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--encoder_hidden_dim', type=int, default=128, help='length of encoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=8, help='number of decoder hidden layers')
    parser.add_argument('--nl', type=str, default='softplus', help='type of non linearity sine | relu')
    parser.add_argument('--latent_size', type=int, default=0, help='number of elements in the latent vector,'
                                                                   ' use 0 for reconstruction')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 1.0],
                    help='radius and scaling')
    parser.add_argument('--neuron_type', type=str, default='quadratic', help='type of neuron')


    parser.add_argument('--encoder_type', type=str, default='none', help='type of encoder none | pointnet')
    parser.add_argument('--loss_type', type=str, default='siren', help='loss type to use: siren | siren_wo_n | igr | igr_wo_n'
                                                                       'siren_wo_n_w_div | siren_wo_n_w_div_w_reg |...')
    parser.add_argument('--div_decay_params', nargs='+', type=float, default=[0.0, 0.5, 0.75],
                        help='epoch number to evaluate')
    parser.add_argument('--div_type', type=str, default='dir_l1', help='divergence term norm dir_l1 | dir_l2 | full_l1 | full_l2')
    parser.add_argument('--grid_res', type=int, default=128,
                        help='uniform grid resolution')
    parser.add_argument('--div_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='uniform',
                        help='how to sample points off the manifold - grid | gaussian | combined')
    parser.add_argument('--init_type', type=str, default='mfgi',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[3e3, 1e2, 1e2, 5e1, 1e1],
                        help='loss terms weights sdf | inter | normal | eikonal | div')
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser(description='Local implicit functions test experiment')
    parser.add_argument('--gpu_idx', type=int, default=1, help='set < 0 to use CPU')
    parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
    parser.add_argument('--file_name', type=str, default='daratech.ply', help='trained model name')
    parser.add_argument('--n_points', type=int, default=0, help='number of points in each point cloud, '
                                                                  'if 0 use training options')
    parser.add_argument('--batch_size', type=int, default=0, help='number of samples in a minibatch, if 0 use training')
    parser.add_argument('--grid_res', type=int, default=512, help='grid resolution for reconstruction')
    parser.add_argument('--export_mesh', type=bool, default=True, help='indicator to export mesh as ply file')
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset folder')

    test_opt = parser.parse_args()
    test_opt.logdir = os.path.join(test_opt.logdir, test_opt.file_name.split('.')[0])
    param_filename = os.path.join(test_opt.logdir, 'trained_models/params.pth')
    train_opt = torch.load(param_filename)

    test_opt.nl, test_opt.latent_size, test_opt.encoder_type, test_opt.n_iterations, test_opt.seed, \
        test_opt.decoder_hidden_dim, test_opt.encoder_hidden_dim, test_opt.decoder_n_hidden_layers, test_opt.init_type, test_opt.neuron_type \
        = train_opt.nl, train_opt.latent_size, train_opt.encoder_type, train_opt.n_iterations, train_opt.seed, \
          train_opt.decoder_hidden_dim, train_opt.encoder_hidden_dim, train_opt.decoder_n_hidden_layers, train_opt.init_type, train_opt.neuron_type
    test_opt.n_point_total = train_opt.n_points

    if test_opt.n_points == 0:
        test_opt.n_points = train_opt.n_points
    if "parallel" in train_opt:
        test_opt.parallel = train_opt.parallel
    else:
        test_opt.parallel = False
    return test_opt
