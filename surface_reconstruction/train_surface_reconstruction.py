# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import numpy as np
import models.Net as Net
from models.losses import Loss
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import surface_recon_args


# get training parameters
args = surface_recon_args.get_train_args()

file_path = os.path.join(args.dataset_path, args.file_name)
logdir = os.path.join(args.logdir, args.file_name.split('.')[0])

# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)
os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/Net.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)
train_set = dataset.ReconDataset(file_path, args.n_points, args.n_iterations, args.grid_res, args.nonmnfld_sample_type)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4,
                                               pin_memory=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
device = torch.device("cuda")

Net = Net.Network(latent_size=args.latent_size, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
                  encoder_type=args.encoder_type, decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                  init_type=args.init_type, neuron_type=args.neuron_type, sphere_init_params=args.sphere_init_params)
if args.parallel:
    if (device.type == 'cuda'):
        Net = torch.nn.DataParallel(Net)

n_parameters = utils.count_parameters(Net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(Net.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0) # Does nothing
Net.to(device)
criterion = Loss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.div_decay, div_type=args.div_type)
num_batches = len(train_dataloader)

# For each batch in the dataloader
for batch_idx, data in enumerate(train_dataloader):
    # if batch_idx in [0, 1, 5, 10, 50, 100] or batch_idx % 500 == 0:
    #     try:
    #         shapename = args.file_name.split('.')[0]
    #         output_dir = os.path.join(logdir, '..', 'result_meshes')
    #         os.makedirs(output_dir, exist_ok=True)
    #         output_ply_filepath = os.path.join(output_dir, shapename+'_iter_{}.ply'.format(batch_idx))
    #         print('Saving to ', output_ply_filepath)
    #         cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
    #         mesh_dict = utils.implicit2mesh(DiGSNet.decoder, None, 128, translate=-cp, scale=1/scale,
    #                             get_mesh=True, device=device, bbox=bbox)
    #         mesh_dict["mesh_obj"].export(output_ply_filepath, vertex_normal=True)
    #     except:
    #         print(traceback.format_exc())
    #         print('Could not generate mesh')
    #         print()

    Net.zero_grad()
    Net.train()

    mnfld_points, mnfld_n_gt, nonmnfld_points = \
        data['points'].to(device), data['mnfld_n'].to(device),  data['nonmnfld_points'].to(device)

    mnfld_points.requires_grad_()
    nonmnfld_points.requires_grad_()

    output_pred = Net(nonmnfld_points, mnfld_points)

    loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt)
    lr = torch.tensor(optimizer.param_groups[0]['lr'])
    loss_dict["lr"] = lr
    utils.log_losses(log_writer_train, batch_idx, num_batches, loss_dict)

    loss_dict["loss"].backward()

    torch.nn.utils.clip_grad_norm_(Net.parameters(), 10.)

    optimizer.step()

    #Output training stats
    if batch_idx % 100 == 0:

        weights = criterion.weights
        utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
        utils.log_string('Iteration: {:4d}/{} ({:.0f}%) Loss: {:.5f} = L_Mnfld: {:.5f} + '
                'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f}'.format(
             batch_idx, len(train_set), 100. * batch_idx / len(train_dataloader),
                    loss_dict["loss"].item(), weights[0]*loss_dict["sdf_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                    weights[2]*loss_dict["normals_loss"].item(), weights[3]*loss_dict["eikonal_term"].item(),
                    weights[4]*loss_dict["div_loss"].item()), log_file)
        utils.log_string('Iteration: {:4d}/{} ({:.0f}%) Unweighted L_s : L_Mnfld: {:.5f},  '
                'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f}'.format(
             batch_idx, len(train_set), 100. * batch_idx / len(train_dataloader),
                    loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                    loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                    loss_dict["div_loss"].item()), log_file)
        utils.log_string('', log_file)

    criterion.update_div_weight(batch_idx, args.n_iterations,
                                args.div_decay_params)  # assumes batch size of 1
    scheduler.step()

    # save model
    # if batch_idx % 1000 == 0 :
        # utils.log_string("saving model to file :generator_model.pth", log_file)
        # torch.save(Net.state_dict(),
        #             os.path.join(model_outdir, 'model.pth' ))
# save model
utils.log_string("saving model to file :generator_model.pth", log_file)
torch.save(Net.state_dict(), os.path.join(model_outdir, 'model.pth' ))