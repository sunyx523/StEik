# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import os.path
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from abc import ABC, abstractmethod
import torch
import open3d as o3d


class ReconDataset(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path, n_points, n_samples=128, res=128, sample_type='grid', sapmling_std=0.005,
                 requires_dist=False, requires_curvatures=False, grid_range=1.1):

        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples
        self.grid_res = res
        self.sample_type = sample_type #grid | gaussian | combined
        self.sampling_std = sapmling_std
        self.requires_dist = requires_dist
        self.nonmnfld_dist, self.nonmnfld_n, self.mnfld_curvs = None, None, None
        self.requires_curvatures = requires_curvatures # assumes a subdirectory names "estimated props" in dataset path
        # load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)


        # extract center and scale points and normals
        self.points, self.mnfld_n = self.get_mnfld_points()
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()

        # generate grid points and find distance to closest point on the line
        self.grid_range = grid_range  # larger than 1 to avoid boundary issues
        x, y, z = np.linspace(-grid_range, grid_range, self.grid_res).astype(np.float32), \
                  np.linspace(-grid_range, grid_range, self.grid_res).astype(np.float32),\
                  np.linspace(-grid_range, grid_range, self.grid_res).astype(np.float32)
        xx, yy, zz = np.meshgrid(x, y, z)
        xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
        self.grid_points = np.stack([xx, yy, zz], axis=1).astype('f')
        self.nonmnfld_points = self.get_nonmnfld_points()

        self.point_idxs = np.arange(self.points.shape[0], dtype=np.int32)
        self.nonmnfld_points_idxs = np.arange(self.nonmnfld_points.shape[0], dtype=np.int32)
        self.generate_batch_indices()


    def generate_batch_indices(self):
        mnfld_idx = []
        nonmnfld_idx = []
        for i in range(self.n_samples):
            mnfld_idx.append(np.random.choice(self.point_idxs, self.n_points))
            nonmnfld_idx.append(np.random.choice(self.nonmnfld_points_idxs, self.n_points))
        self.mnfld_idx = np.array(mnfld_idx)
        self.nonmnfld_idx = np.array(nonmnfld_idx)


    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        # center and scale point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        # self.scale = np.linalg.norm(points, axis=-1).max(-1)
        self.scale = np.abs(points).max()
        points = points / self.scale

        return points, normals

    def get_nonmnfld_points(self):
        if self.sample_type == 'grid':
            nonmnfld_points = self.grid_points
        elif self.sample_type == 'uniform':
            nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                                size=(self.grid_res * self.grid_res * self.grid_res, 3)).astype(np.float32)
        elif self.sample_type == 'gaussian':
            nonmnfld_points = self.sample_gaussian_noise_around_shape()
            idx = np.random.choice(range(nonmnfld_points.shape[1]), self.grid_res * self.grid_res)
            sample_idx = np.random.choice(range(nonmnfld_points.shape[0]), self.grid_res * self.grid_res)
            nonmnfld_points = nonmnfld_points[sample_idx, idx]
        elif self.sample_type == 'combined':
            nonmnfld_points1 = self.sample_gaussian_noise_around_shape()
            nonmnfld_points2 = self.grid_points
            idx1 = np.random.choice(range(nonmnfld_points1.shape[1]), int(np.ceil(self.grid_res * self.grid_res / 2)))
            idx2 = np.random.choice(range(nonmnfld_points2.shape[0]), int(np.floor(self.grid_res * self.grid_res / 2)))
            sample_idx = np.random.choice(range(nonmnfld_points1.shape[0]), int(np.ceil(self.grid_res * self.grid_res / 2)))

            nonmnfld_points = np.concatenate([nonmnfld_points1[sample_idx, idx1], nonmnfld_points2[idx2]], axis=0)
        else:
            raise Warning("Unsupported non manfold sampling type")
        return nonmnfld_points

    def sample_gaussian_noise_around_shape(self):
        n_noisy_points = int(np.round(self.grid_res * self.grid_res / self.n_points))
        noise = np.random.multivariate_normal([0, 0, 0], [[self.sampling_std, 0, 0], [0, self.sampling_std, 0],
                                                          [0, 0, self.sampling_std]],
                                              size=(self.points.shape[0], n_noisy_points)).astype(np.float32)
        nonmnfld_points = np.tile(self.points[:, None, :], [1, n_noisy_points, 1]) + noise
        nonmnfld_points = nonmnfld_points.reshape([nonmnfld_points.shape[0], -1, nonmnfld_points.shape[-1]])
        return nonmnfld_points

    @abstractmethod
    def get_points_distances_and_normals(self, points):
        # implement a function that computes the distance and normal vectors of nonmanifold points.
        # default implementation finds the nearest neighbor and return its normal and the distance projected onto it.
        # which is a coarse approximation

        # compute distance and normal (general case - closest point)
        kdtree = spatial.cKDTree(self.points)
        distances, nn_idx = kdtree.query(points, k=1, n_jobs=-1)
        normals = self.mnfld_n[nn_idx]
        p1p2 = points - self.points[nn_idx]
        sings = np.sign(np.einsum('ij,ij->i', p1p2, self.mnfld_n[nn_idx]))
        distances = sings * distances
        return distances.astype('f'), normals.astype('f')


    def __getitem__(self, index):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:self.n_points]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        manifold_normals = self.mnfld_n[mnfld_idx] # (n_points, 3)

        nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                            size=(self.n_points, 3)).astype(np.float32) # (n_points, 3)

        return {'points': manifold_points,  'mnfld_n': manifold_normals,
                'nonmnfld_points': nonmnfld_points}

    def __len__(self):
        return self.n_samples




