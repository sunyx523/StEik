# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import torch
import utils.visualizations as vis
import numpy as np
import models.Net as Net
import torch.nn.parallel
import utils.utils as utils
import surface_recon_args

# get training parameters
args = surface_recon_args.get_test_args()

file_path = os.path.join(args.dataset_path, args.file_name)
if args.export_mesh:
    outdir = os.path.join(os.path.dirname(args.logdir), 'result_meshes')
    os.makedirs(outdir, exist_ok=True)
    output_ply_filepath = os.path.join(outdir, args.file_name)
# get data loader
torch.manual_seed(args.seed)
np.random.seed(args.seed)
test_set = dataset.ReconDataset(file_path, args.n_points, n_samples=1, res=args.grid_res, sample_type='grid',
                                requires_dist=False)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
device = torch.device("cuda")

Net = Net.Network(latent_size=args.latent_size, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim,
                           nl=args.nl, encoder_type=args.encoder_type,
                           decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                            init_type=args.init_type, neuron_type=args.neuron_type)
if args.parallel:
    if (device.type == 'cuda'):
        Net = torch.nn.DataParallel(Net)

model_dir = os.path.join(args.logdir, 'trained_models')
trained_model_filename = os.path.join(model_dir, 'model.pth')
Net.load_state_dict(torch.load(trained_model_filename, map_location=device))
Net.to(device)
latent = None

print("Converting implicit to mesh for file {}".format(args.file_name))
cp, scale, bbox = test_set.cp, test_set.scale, test_set.bbox
test_set, test_dataloader, clean_points_gt, normals_gt,  nonmnfld_points, data = None, None, None, None, None, None  # free up memory
mesh_dict = utils.implicit2mesh(Net.decoder, latent, args.grid_res, translate=-cp, scale=1/scale,
                                get_mesh=True, device=device, bbox=bbox)
vis.plot_mesh(mesh_dict["mesh_trace"], mesh=mesh_dict["mesh_obj"], output_ply_path=output_ply_filepath, show_ax=False,
              title_txt=args.file_name.split('.')[0], show=False)

print("Conversion complete.")