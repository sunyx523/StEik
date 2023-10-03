# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset2d
import torch
import utils.visualizations as vis
import numpy as np
import models.Net as Net
from models.losses import Loss
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import sc_args

# get training parameters
args = sc_args.get_args()
gpu_idx, nl, n_points, batch_size, n_samples, latent_size, lr, num_epochs, logdir, \
n_loss_type, normalize_normal_loss, unsigned_n, unsigned_d, loss_type, seed, encoder_type,\
    model_dirpath, inter_loss_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.n_samples, args.latent_size, \
    args.lr, args.num_epochs, args.logdir, args.n_loss_type, \
    args.normalize_normal_loss, args.unsigned_n, args.unsigned_d, args.loss_type, args.seed, args.encoder_type, \
    args.model_dirpath, args.inter_loss_type

# set up backup and logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)
os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/Net.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)
train_set = basic_shape_dataset2d.get2D_dataset(n_points, n_samples, args.grid_res, args.nonmnfld_sample_type, shape_type=args.shape_type)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
torch.manual_seed(seed)
np.random.seed(seed)

# get model
device = torch.device("cuda:" + str(gpu_idx) if (torch.cuda.is_available()) else "cpu")

SINR = Net.Network(latent_size=latent_size, in_dim=2, decoder_hidden_dim=args.decoder_hidden_dim,
                              nl=args.nl, encoder_type='none', neuron_type=args.neuron_type,
                              decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type)
if args.parallel:
    if (device.type == 'cuda'):
        SINR = torch.nn.DataParallel(SINR)
n_parameters = utils.count_parameters(SINR)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)
# Setup Adam optimizers
optimizer = optim.Adam(SINR.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0) # Does nothing
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(2000, args.n_samples*num_epochs, 2000), gamma=0.5)  # milestones in number of optimizer iterations

if not args.refine_epoch == 0:
    refine_model_filename = os.path.join(model_outdir,
                                         'model_%d.pth' % (args.refine_epoch))
    SINR.load_state_dict(torch.load(refine_model_filename, map_location=device))
    optimizer.step()

SINR.to(device)

# get loss
criterion = Loss(weights=[3e3, 1e2, 1e2, 5e1, 1e2],
                               loss_type=loss_type, div_decay=args.div_decay, div_type=args.div_type)

num_batches = len(train_dataloader)
refine_flag = True
grid_points = train_set.grid_points

# Train the shape implicit neural representation
train_time_running_mean = 0
train_time_running_sum = 0
for epoch in range(num_epochs):
    if epoch <= args.refine_epoch and refine_flag and not args.refine_epoch == 0:
        scheduler.step()
        continue
    else:
        refine_flag = False

    for batch_idx, data in enumerate(train_dataloader):

        # save model before update
        if batch_idx % 50 == 0 :
            utils.log_string("saving model to file :{}".format('model_%d.pth' % (batch_idx)),
                             log_file)
            torch.save(SINR.state_dict(),
                       os.path.join(model_outdir, 'model_%d.pth' % (batch_idx)))
        # train model
        SINR.zero_grad()
        # with torch.cuda.amp.autocast():
        SINR.train()

        mnfld_points, normals_gt, nonmnfld_dist_gt, nonmnfld_points, nonmnfld_n_gt  =  \
            data['points'].to(device), data['mnfld_n'].to(device),  data['nonmnfld_dist'].to(device), \
            data['nonmnfld_points'].to(device), data['nonmnfld_n'].to(device),

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()

        output_pred = SINR(nonmnfld_points, mnfld_points)

        loss_dict, _ = criterion(output_pred=output_pred, mnfld_points=mnfld_points, 
                                    nonmnfld_points=nonmnfld_points, mnfld_n_gt=normals_gt)

        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        utils.log_losses(log_writer_train, batch_idx, num_batches, loss_dict)
        # utils.log_weight_hist(log_writer_train, epoch, batch_idx, num_batches, SINR.decoder.fc_block.net[:], batch_size)
        if batch_idx % 25 == 0:
            weights = criterion.weights
            utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                    'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["loss"].item(), weights[0]*loss_dict["sdf_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                        weights[2]*loss_dict["normals_loss"].item(), weights[3]*loss_dict["eikonal_term"].item(),
                        weights[4]*loss_dict["div_loss"].item()), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f},  '
                    'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                        loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                        loss_dict["div_loss"].item()), log_file)
            utils.log_string('', log_file)


        loss_dict["loss"].backward()
        optimizer.step()
        
        scheduler.step()
        criterion.update_div_weight(epoch*n_samples + batch_idx, num_epochs*n_samples, args.div_decay_params)

        # save last model
        if batch_idx == num_batches - 1 :
            utils.log_string("saving model to file :{}".format('model_%d.pth' % (num_batches)),
                             log_file)
            torch.save(SINR.state_dict(),
                       os.path.join(model_outdir, 'model_%d.pth' % (num_batches)))