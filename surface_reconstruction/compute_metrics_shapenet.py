# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
import numpy as np
import json
import os
import trimesh
from scipy.spatial import cKDTree as KDTree
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import surface_recon_args
# import utils.utils as utils

import torch
import recon_dataset as dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.Net as Net
device = torch.device("cuda")
args = surface_recon_args.get_train_args()
Net = Net.Network(latent_size=0, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl, encoder_type='none',
                   decoder_n_hidden_layers=args.decoder_n_hidden_layers, neuron_type=args.neuron_type, init_type='mfgi')


dataset_path = args.dataset_path
raw_dataset_path = args.raw_dataset_path
mesh_path = args.logdir

print(dataset_path)
print(raw_dataset_path)
print(mesh_path)

# Print metrics to file in logdir as well
out_path = os.path.join(args.logdir, 'metric_summary.txt')
import builtins as __builtin__
def print(*args, **kwargs):
    # Override print function to also print to file
    __builtin__.print(*args, **kwargs)
    with open(out_path, 'a') as fp:
        __builtin__.print(*args, file=fp, **kwargs)

print("Metrics on ShapeNet")

shape_class_name_dict = \
        {
            "04256520": "sofa",
            "02691156": "airplane",
            "03636649": "lamp",
            "04401088": "telephone",
            "04530566": "watercraft",
            "03691459": "loudspeaker",
            "03001627": "chair",
            "02933112": "cabinet",
            "04379243": "table",
            "03211117": "display",
            "02958343": "car",
            "02828884": "bench",
            "04090263": "rifle",
        }

shape_class_name2id = {v:k for k,v in shape_class_name_dict.items()}

eval_type = "DeepSDF"
# eval_type = "Default"

def compute_dists(recon_points, gt_points, eval_type="Default"):
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points, workers=4)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points, workers=4)
    if eval_type == 'DeepSDF':
        cd_re2gt = np.mean(re2gt_distances**2)
        cd_gt2re = np.mean(gt2re_distances**2)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = cd_re2gt + cd_gt2re
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    else:
        cd_re2gt = np.mean(re2gt_distances)
        cd_gt2re = np.mean(gt2re_distances)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = 0.5* (cd_re2gt + cd_gt2re)
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist, hausdorff_distance, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re

order = ["car","chair","airplane","display","table","rifle","cabinet","loudspeaker","telephone","bench","sofa","watercraft","lamp",]
order = sorted(order)

chamfers = {}
IoUs = {}
# for shape_class in os.listdir(mesh_path):
for shape_class in order:
    if shape_class not in os.listdir(mesh_path):
        continue
    shape_class_id = shape_class_name2id[shape_class]
    gt_shape_class_path = os.path.join(dataset_path, shape_class)
    gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)

    result_meshes_path = os.path.join(mesh_path, shape_class, 'result_meshes')
    saved_weights_class_path = os.path.join(mesh_path, shape_class)
    shape_files = [f for f in os.listdir(result_meshes_path) if '_iter_' not in f]
    print("Found {} files for {}".format(len(shape_files), shape_class))

    chamfers[shape_class] = []
    IoUs[shape_class] = []
    for shape_file in shape_files:
        shape = shape_file.replace(".ply","")
        recon_shape_path = os.path.join(result_meshes_path, shape_file)
        recon_mesh = trimesh.load(recon_shape_path)
        
        gt_shape_path = os.path.join(gt_shape_class_path, shape_file)
        gt_shape_weights_path = os.path.join(saved_weights_class_path, shape_file.replace('.ply',''), 'trained_models', 'model.pth')
        gt_pc = trimesh.load(gt_shape_path)
        gt_raw_shape_path = os.path.join(gt_raw_shape_class_path, shape)
        points = np.load(os.path.join(gt_raw_shape_path, 'points.npz')) # [('points', (100000, 3)), ('occupancies', (12500,)), ('loc', (3,)), ('scale', ())]
        pointcloud = np.load(os.path.join(gt_raw_shape_path, 'pointcloud.npz')) # [('points', (100000, 3)), ('normals', (100000, 3)), ('loc', (3,)), ('scale', ())]
        gen_points = points['points'] # (100000,3)
        occupancies = np.unpackbits(points['occupancies']) # (100000)

        n_points=15000; n_samples=10000; grid_res=512
        test_set = dataset.ReconDataset(gt_shape_path, n_points*n_samples, n_samples=1, res=grid_res, sample_type='grid',
                                requires_dist=False)
        cp, scale, bbox = test_set.cp, test_set.scale, test_set.bbox
        Net.load_state_dict(torch.load(gt_shape_weights_path, map_location=device))
        Net.to(device)

        eval_points = (gen_points - cp) / scale
        eval_points = torch.tensor(eval_points, device=device, dtype=torch.float32)
        res = Net.decoder(eval_points)

        pred_occupancies = (res.reshape(-1)<0).int().detach().cpu().numpy()
        iou = (occupancies & pred_occupancies).sum() / (occupancies | pred_occupancies).sum()
        IoUs[shape_class].append(iou)
        print("IoU, {:.4f}".format(iou))


        gt_points = gt_pc.vertices
        recon_points = trimesh.sample.sample_surface(recon_mesh, 30000)[0]
        chamfer_dist, hausdorff_distance,  cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re = compute_dists(recon_points, gt_points, eval_type="DeepSDF")
        chamfers[shape_class].append(chamfer_dist)
        print('\t{: <10} {:e} {:e} {:e} {:e}'.format(shape, chamfer_dist, hausdorff_distance, cd_re2gt, hd_re2gt))
    dists_np = np.array(chamfers[shape_class])
    print("{}: Mean: {:e}, Median: {:e}, Std: {:e}".format(shape_class,  dists_np.mean(), np.median(dists_np), dists_np.std()))
print('Final Chamfers')


# for key in chamfers:
for key in order:
    if key not in chamfers:
        continue
    dists_np = np.array(chamfers[key])
    print("{:<15} : Mean: {:e}, Median: {:e}, Std: {:e}".format(key, dists_np.mean(), np.median(dists_np), dists_np.std()))
all_chamfers = np.concatenate([chamfers[k] for k in chamfers]).reshape(-1)
print("Overall: Mean: {:e}, Median: {:e}, Std: {:e}".format(all_chamfers.mean(), np.median(all_chamfers), all_chamfers.std()))

print('Final IoUs')
for key in order:
    if key not in IoUs:
        continue
    IoUs_np = np.array(IoUs[key])
    print("{:<15} : Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(key, IoUs_np.mean(), np.median(IoUs_np), IoUs_np.std()))
all_IoUs = np.concatenate([IoUs[k] for k in IoUs]).reshape(-1)
print("Overall: Mean: {:.4f}, Median: {:.4f}, Std: {:.4f}".format(all_IoUs.mean(), np.median(all_IoUs), all_IoUs.std()))
