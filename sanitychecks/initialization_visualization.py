# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
# This code initializes a 2D SIREN and generates images of SDF (image + 3D surface), eikonal, curl, divergence and distance to GT difference
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset2d
import torch
import utils.visualizations as vis
import numpy as np
import models.Net as Net
import utils.utils as utils
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Initialization visualization test')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--output_dir', type=str, default='./log/initialization_vis/', help='path to store output images')
# parser.add_argument('--plot_second_derivs', type=bool, default=False, help='toggle on to plot second derivative images')
parser.add_argument('--plot_second_derivs', type=bool, default=True, help='toggle on to plot second derivative images')
parser.add_argument('--nl', type=str, default='sine', help='specify nonlinearity sine | relu')
parser.add_argument('--init_type', type=str, default='mfgi', help='specify type of initialization mfgi | geometric_relu | geometric_sine | siren')
# parser.add_argument('--init_type', type=str, default='geometric_sine', help='specify type of initialization mfgi | geometric_relu | geometric_sine | siren')
# parser.add_argument('--init_type', type=str, default='siren', help='specify type of initialization mfgi | geometric_relu | geometric_sine | siren')
parser.add_argument('--decoder_n_hidden_layers', type=int, default=4, help='number of MLP layers')
parser.add_argument('--decoder_hidden_dim', type=int, default=128, help='number units in the decoder')
parser.add_argument('--grid_res', type=int, default=256, help='evaluation grid resolution')
args = parser.parse_args()

seed = 0
# get data loaders
torch.manual_seed(seed)
np.random.seed(seed)
# [n_points, n_samples, res, sample_type, sapmling_std], shape_type
test_set = basic_shape_dataset2d.get2D_dataset(128, 1, args.grid_res, 'grid', 'circle')
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
                                              pin_memory=True)
# get model
device = torch.device("cuda:" + str(args.gpu_idx) if (torch.cuda.is_available()) else "cpu")

SINR = Net.Network(latent_size=0, in_dim=2, decoder_hidden_dim=args.decoder_hidden_dim,
                              nl=args.nl, encoder_type='none',
                              decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type)

os.makedirs(args.output_dir, exist_ok=True)

_, test_data = next(enumerate(test_dataloader))
SINR.eval()
nonmnfld_points = test_data['nonmnfld_points'].to(device)
SINR.to(device)

print("Visualizing initialization output {}".format(args.init_type))
# initialize and evaluate the model
nonmnfld_points.requires_grad_()
output_pred = SINR(nonmnfld_points, mnfld_pnts=None)

x_grid, y_grid, z_grid, z_diff, eikonal_term, grid_div, grid_curl = \
    utils.compute_deriv_props(SINR.decoder, output_pred["latent"], z_gt=test_set.dist_img, device=device)

contour_img, curl_img, eikonal_img, div_image, z_diff_img =\
    vis.plot_init_contour_div_props(x_grid, y_grid, z_grid, clean_points=None,
                            z_diff=z_diff, eikonal_term=eikonal_term, grid_div=grid_div, grid_curl=grid_curl,
                            example_idx=0, n_gt=None, n_pred=None,
                            nonmnfld_points=None, 
                            title_text="",
                            # title_text='initialization:  ' + str(args.init_type),
                            plot_second_derivs=args.plot_second_derivs)

# print(z_grid.min(), z_grid.max())

# Output visualizations
fig = vis.plot_sdf_surface(x_grid, y_grid, z_grid, show=False, show_ax=True, title_txt='sdf initialization surface')
fig.write_html(os.path.join(args.output_dir, "sdf_surface" + str(args.decoder_hidden_dim) + '_' + args.init_type + ".html"))
im = Image.fromarray(contour_img)
im.save(os.path.join(args.output_dir, "sdf_" + str(args.decoder_hidden_dim) + '_' + args.init_type + "_init.png"))
im = Image.fromarray(eikonal_img)
im.save(os.path.join(args.output_dir, "eikonal_" + str(args.decoder_hidden_dim) + '_' + args.init_type + "_init.png"))
im = Image.fromarray(z_diff_img)
im.save(os.path.join(args.output_dir, "zdiff_" + str(args.decoder_hidden_dim) + '_' + args.init_type + "_init.png"))
if args.plot_second_derivs:
    # im = Image.fromarray(curl_img)
    # im.save(os.path.join(args.output_dir, "curl_" + str(args.decoder_hidden_dim) + '_' + args.init_type + "_init.png"))
    im = Image.fromarray(div_image)
    im.save(os.path.join(args.output_dir, "div_" + str(args.decoder_hidden_dim) + '_' + args.init_type + "_init.png"))





