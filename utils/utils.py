# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
import os
from tensorboardX import SummaryWriter
import io
from PIL import Image
import numpy as np
import torch
from torch.autograd import grad
import warnings
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from skimage import measure
import plotly.graph_objs as go
import trimesh


def center_and_scale(points, cp=None, scale=None):
    # center a point cloud and scale it to unite sphere.
    if cp is None:
        cp = points.mean(axis=1)
    points = points - cp[:, None, :]
    if scale is None:
        scale = np.linalg.norm(points, axis=-1).max(-1)
    points = points / scale[:, None, None]
    return points, cp, scale


def log_losses(writer, batch_idx, num_batches, loss_dict):
    # helper function to log losses to tensorboardx writer
    iteration = batch_idx + 1
    for loss in loss_dict.keys():
        writer.add_scalar(loss, loss_dict[loss].item(), iteration)
    return iteration

def log_weight_hist(writer, batch_idx, num_batches, net_blocks):
    # helper function to log losses to tensorboardx writer
    iteration = batch_idx + 1
    for i, block in enumerate(net_blocks):
        writer.add_histogram('layer_weights_' + str(i), block[0].weight, iteration)
        writer.add_histogram('layer_biases_' + str(i), block[0].bias
                             , iteration)
    return iteration


def log_images(writer, iteration, contour_img, curl_img, eikonal_img, div_image, z_diff_img, example_idx):
    # helper function to log images to tensorboardx writer
    writer.add_image('implicit_function/' + str(example_idx), contour_img.transpose(2, 0, 1), iteration)
    writer.add_image('curl/' + str(example_idx), curl_img.transpose(2, 0, 1), iteration)
    writer.add_image('eikonal_term/' + str(example_idx), eikonal_img.transpose(2, 0, 1), iteration)
    writer.add_image('divergence/' + str(example_idx), div_image.transpose(2, 0, 1), iteration)
    writer.add_image('z_diff/' + str(example_idx), z_diff_img.transpose(2, 0, 1), iteration)


def log_string(out_str, log_file):
    # helper function to log a string to file and print it
    log_file.write(out_str+'\n')
    log_file.flush()
    print(out_str)


def setup_logdir(logdir, args=None):
    # helper function to set up logging directory

    os.makedirs(logdir, exist_ok=True)
    log_writer_train = SummaryWriter(os.path.join(logdir, 'train'))
    log_writer_test = SummaryWriter(os.path.join(logdir, 'test'))
    log_filename = os.path.join(logdir, 'out.log')
    log_file = open(log_filename, 'w')
    model_outdir = os.path.join(logdir, 'trained_models')
    os.makedirs(model_outdir, exist_ok=True)

    if args is not None:
        params_filename = os.path.join(model_outdir, 'params.pth')
        torch.save(args, params_filename)  # save parameters
        log_string("input params: \n" + str(args), log_file)
    else:
        warnings.warn("Training options not provided. Not saving training options...")


    return log_file, log_writer_train, log_writer_test, model_outdir


def backup_code(logdir, dir_list=[], file_list=[]):
    #backup models code
    code_bkp_dir = os.path.join(logdir, 'code_bkp')
    os.makedirs(code_bkp_dir, exist_ok=True)
    for dir_name in dir_list:
        print("copying directory {} to {}".format(dir_name, code_bkp_dir))
        os.system('cp -r %s %s' % (dir_name, code_bkp_dir))  # backup the current model code
    for file_name in file_list:
        print("copying file {} to {}".format(file_name, code_bkp_dir))
        os.system('cp %s %s' % (file_name, code_bkp_dir))


def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf).convert('RGB')
    img = np.asarray(img)
    return img


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True)[0]#[:, -3:]
    return points_grad

def get_cuda_ifavailable(torch_obj, device=None):
    # if cuda is available return a cuda obeject
    if (torch.cuda.is_available()):
        return torch_obj.cuda(device=device)
    else:
        return torch_obj

def compute_props(decoder, latent, z_gt, device):

    # compute derivative properties on a grid
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    #plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    return x, y, z_np, z_diff

def compute_deriv_props(decoder, latent, z_gt, device):

    # compute derivative properties on a grid
    res = z_gt.shape[1]
    x, y, grid_points = get_2d_grid_uniform(resolution=res, range=1.2, device=device)
    grid_points.requires_grad_()
    if latent is not None:
        grid_points_latent = torch.cat([latent.expand(grid_points.shape[0], -1), grid_points], dim=1)
    else:
        grid_points_latent = grid_points
    z = decoder(grid_points_latent)
    z_np = z.detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    # compute derivatives
    grid_grad = gradient(grid_points, z)
    dx = gradient(grid_points, grid_grad[:, 0], create_graph=False, retain_graph=True)
    dy = gradient(grid_points, grid_grad[:, 1], create_graph=False, retain_graph=False)

    grid_curl = (dx[:, 1] - dy[:, 0]).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute eikonal term (gradient magnitude)
    eikonal_term = ((grid_grad.norm(2, dim=-1) - 1) ** 2).cpu().detach().numpy().reshape(x.shape[0], x.shape[0])

    # compute divergence
    grid_div = (dx[:, 0] + dy[:, 1]).detach().cpu().numpy().reshape(x.shape[0], x.shape[0])

    #plot z difference image
    z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt)).reshape(x.shape[0], x.shape[0])

    return x, y, z_np, z_diff, eikonal_term, grid_div, grid_curl


def get_2d_grid_uniform(resolution=100, range=1.2, device=None):
    # generate points on a uniform grid within  a given range
    x = np.linspace(-range, range, resolution)
    y = x
    xx, yy = np.meshgrid(x, y)
    grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float),
                                             device=device)
    return x, y, grid_points


def get_3d_grid(resolution=100, bbox=1.2*np.array([[-1, 1], [-1, 1], [-1, 1]]), device=None, eps=0.1, dtype=np.float16):
    # generate points on a uniform grid within  a given range
    # reimplemented from SAL : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    # and IGR : https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py

    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])
    if (shortest_axis == 0):
        x = np.linspace(bbox[0, 0] - eps,  bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(bbox[1, 0] - eps,  bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype)) #
    # grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16),
    #                                          device=device)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def read_vnf_ply(filename):
    # read vertices and normal vectors from a ply file
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    nx = np.asarray(plydata.elements[0].data['nx'])
    ny = np.asarray(plydata.elements[0].data['ny'])
    nz = np.asarray(plydata.elements[0].data['nz'])
    return np.stack([x, y, z], axis=1), np.stack([nx, ny, nz], axis=1), plydata['face'].data


def scale_pc_to_unit_sphere(points, cp=None, s=None):
    if cp is None:
        cp = points.mean(axis=0)
    points = points - cp[None, :]
    if s is None:
        s = np.linalg.norm(points, axis=-1).max(-1)
    points = points / s
    return points, cp, s

def recon_metrics(pc1, pc2, one_sided=False, alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
                  percentiles=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], k=[10, 25, 50], return_all=False):
    # Compute reconstruction benchmarc evaluation metrics :
    # chamfer and hausdorff distance metrics between two point clouds pc1 and pc2 [nx3]
    # percentage of distance points metric (not used in the paper)
    # and the meal local chamfer variance
    # pc1 is the reconstruction and pc2 is the gt data

    scale = np.abs(pc2).max()

    # compute one side
    pc1_kd_tree = KDTree(pc1)
    one_distances, one_vertex_ids = pc1_kd_tree.query(pc2, n_jobs=4)
    cd12 = np.mean(one_distances)
    hd12 = np.max(one_distances)
    cdmed12 = np.median(one_distances)
    cd21 = None
    hd21 = None
    cdmed21 = None
    pods2 = None
    cdp2 = None
    chamfer_distance = cd12
    hausdorff_distance = hd12

    # compute chamfer distance percentiles cdp
    cdp1 = np.percentile(one_distances, percentiles, interpolation='lower')

    # compute PoD
    pod1 = []
    for alpha in alphas:
        pod1.append((one_distances < alpha * scale).sum() / one_distances.shape[0])


    if not one_sided:
        # compute second side
        pc2_kd_tree = KDTree(pc2)
        two_distances, two_vertex_ids = pc2_kd_tree.query(pc1, n_jobs=4)
        cd21 = np.mean(two_distances)
        hd21 = np.max(two_distances)
        cdmed21 = np.median(two_distances)
        chamfer_distance = 0.5*(cd12 + cd21)
        hausdorff_distance = np.max((hd12, hd21))
        # compute chamfer distance percentiles cdp
        cdp2 = np.percentile(two_distances, percentiles)
        # compute PoD
        pod2 = []
        for alpha in alphas:
            pod2.append((two_distances < alpha*scale).sum() / two_distances.shape[0])

     # compute double sided pod
    pod12 = []
    for alpha in alphas:
        pod12.append( ((one_distances < alpha * scale).sum() + (two_distances < alpha*scale).sum()) /
                      (one_distances.shape[0] + two_distances.shape[0]))
    cdp12 = np.percentile(np.concatenate([one_distances, two_distances]), percentiles) # compute chamfer distance percentiles cdp

    nn1_dist, local_idx2 = pc1_kd_tree.query(pc1, max(k), n_jobs=-1)
    nn1_dist_2pc2 = two_distances[local_idx2]
    malcv = [(nn1_dist_2pc2[:, :k0]/ nn1_dist.mean(axis=1, keepdims=True)).var(axis=1).mean() for k0 in k]


    if return_all:
        return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
            (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv, (one_distances, two_distances)


    return chamfer_distance, hausdorff_distance, (cd12, cd21, cdmed12, cdmed21, hd12, hd21), (pod1, pod2, pod12), \
           (cdp1.tolist(), cdp2.tolist(), cdp12.tolist()), malcv

def load_reconstruction_data(file_path, n_points=30000, sample_type='vertices'):
    extension = file_path.split('.')[-1]
    if extension == 'xyz':
        points = np.loadtxt(file_path)
    elif extension == 'ply':
        mesh = trimesh.load_mesh(file_path)

        if hasattr(mesh, 'faces') and not sample_type == 'vertices':
            # sample points if its a triangle mesh
            points = trimesh.sample.sample_surface(mesh, n_points)[0]
        else:
            # use the vertices if its a point cloud
            points = mesh.vertices
    # Center and scale points
    # cp = points.mean(axis=0)
    # points = points - cp[None, :]
    # scale = np.abs(points).max()
    # points = points / scale
    return np.array(points).astype('float32')


def implicit2mesh(decoder, latent, grid_res, translate=[0., 0., 0.], scale=1, get_mesh=True, device=None,
                  bbox=np.array([[-1, 1], [-1, 1], [-1, 1]])):
    # compute a mesh from the implicit representation in the decoder.
    # Uses marching cubes.
    # reimplemented from SAL get surface trace function : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    print('in implicit2mesh')
    print(grid_res, translate, scale, bbox)
    mesh = None
    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    print('Finished getting grid_dict')
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"]

    z = []
    # for point in tqdm(torch.split(pnts, 100000, dim=0)):
    for point in tqdm(torch.split(pnts, 100000, dim=0)):
        # point: (100000, 3)
        if (not latent is None):
            point = torch.cat([point, latent.unsqueeze(0).repeat(point.shape[0], 1), ], dim=1)
        point = get_cuda_ifavailable(point, device=device)
        z.append(decoder(point.type(torch.float32)).detach().cpu().numpy())
    z = np.concatenate(z, axis=0).reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                                          grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2]).astype(np.float64)
    # import pdb; pdb.set_trace()
    print(z.min(), z.max())

    verts, faces, normals, values = measure.marching_cubes(volume=z, level=0.0,
                                                                   spacing=(cell_width, cell_width, cell_width))

    verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
    verts = verts * (1/scale) - translate

    if get_mesh:
        mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)
    trace = None
    # I, J, K = ([triplet[c] for triplet in faces] for c in range(3))

    # trace = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
    #                       i=I, j=J, k=K, name='',
    #                       color='orange', opacity=0.5)]

    return { "mesh_trace": trace,
            "mesh_obj": mesh }


def convert_xyz_to_ply_with_noise(file_path, noise=None):
    #convert ply file in xyznxnynz format to ply file
    points = np.loadtxt(file_path)
    if noise is None:
        mesh = trimesh.Trimesh(points[:, :3], [], vertex_normals=points[:, 3:])
        mesh.export(file_path.split('.')[0] + '.ply', vertex_normal=True)
    else:
        for std in noise:
            bbox_scale = np.abs(points).max()
            var = std*std
            cov_mat = bbox_scale * np.array([[var, 0., 0.], [0., var, 0.], [0., 0., var]])
            noise = np.random.multivariate_normal([0., 0., 0.], cov_mat, size=points.shape[0], check_valid='warn', tol=1e-8)
            mesh = trimesh.Trimesh(points[:, :3] + noise, [], vertex_normals=points[:, 3:])
            mesh.export(file_path.split('.')[0] + '_' +str(std) + '.ply', vertex_normal=True)

def count_parameters(model):
    #count the number of parameters in a given model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    file_path = '/home/sitzikbs/Datasets/Reconstruction_IKEA_sample/interior_room.xyz'
    convert_xyz_to_ply_with_noise(file_path, noise=[0.01])
