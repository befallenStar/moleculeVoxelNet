# -*- encoding: utf-8 -*-
import numpy as np

atoms_vector = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


class VoxelLoader:
    def __init__(self, data, xyz_index=(0, 1, 2)):
        self.d = np.array(data)
        self.x, self.y, self.z = xyz_index

    def __call__(self, *size, sigma=1, squeeze=False):
        xyz_range = self.__xyz_range__()
        threeD_xyz = self.__get_3D_matrix__(self.d, xyz_range, *size)
        full_mat = self.__gaussian_smoothing__(threeD_xyz, sigma)
        if squeeze:
            full_mat = self.__squeeze__(full_mat)
        return full_mat

    def __get_3D_matrix__(self, full_list, xyz_range, *size):
        lx, ly, lz, hx, hy, hz = xyz_range
        assert len(size) == 3
        D, H, W = size
        threeD_xyz = np.zeros(shape=[D, H, W, len(atoms_vector)],
                              dtype=np.float32)
        for itm in full_list:
            x, y, z = itm[[self.x, self.y, self.z]]
            idx = atoms_vector.index(int(itm[-1]))
            pos_x = int(
                (x - lx) * (D - 1) / (hx - lx)) if hx - lx != 0 else int(D / 2)
            pos_y = int(
                (y - ly) * (H - 1) / (hy - ly)) if hy - ly != 0 else int(H / 2)
            pos_z = int(
                (z - lz) * (W - 1) / (hz - lz)) if hz - lz != 0 else int(W / 2)
            threeD_xyz[pos_x][pos_y][pos_z][idx] = 1.
        return threeD_xyz

    def __gaussian_smoothing__(self, full_mat, sigma=2):
        # print(full_mat.shape)
        omega = 1 / sigma
        full = np.zeros(
            [len(full_mat) + 2 * sigma, len(full_mat[0]) + 2 * sigma,
             len(full_mat[0][0]) + 2 * sigma, len(full_mat[0][0][0])])
        full[sigma:len(full_mat) + sigma, sigma:len(full_mat[0]) + sigma,
        sigma:len(full_mat[0][0]) + sigma] = full_mat
        xx, yy, zz, sub = np.where(full_mat != 0)
        for count in range(len(xx)):
            for x in range(-sigma, sigma + 1):
                for y in range(-sigma, sigma + 1):
                    for z in range(-sigma, sigma + 1):
                        if abs(x) + abs(y) + abs(z) > sigma or abs(x) + abs(
                                y) + abs(z) == 0:
                            continue
                        atom = full_mat[xx[count]][yy[count]][zz[count]].copy()
                        idx = sub[count]
                        gau = np.exp(
                            -(x ** 2 + y ** 2 + z ** 2) / (2 * (sigma ** 2)))
                        cos = np.cos(2 * np.pi * omega * np.sqrt(
                            x ** 2 + y ** 2 + z ** 2))
                        atom[idx] *= gau * cos

                        full[xx[count] + x + sigma][yy[count] + y + sigma][
                            zz[count] + z + sigma] += atom
                        # p = full[xx[count] + x + sigma][yy[count] + y + sigma][
                        #     zz[count] + z + sigma][idx]
                        # if p > 1:
                        #     full[xx[count] + x + sigma][yy[count] + y + sigma][
                        #         zz[count] + z + sigma][idx] = self.__sigmoid__(
                        #         p)
        full = self.__sigmoid__(full)
        return full

    def __xyz_range__(self):
        x_min = self.d[:, self.x].min()
        x_max = self.d[:, self.x].max()
        y_min = self.d[:, self.y].min()
        y_max = self.d[:, self.y].max()
        z_min = self.d[:, self.z].min()
        z_max = self.d[:, self.z].max()
        return x_min, y_min, z_min, x_max, y_max, z_max

    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))

    def __squeeze__(self, full_mat, scale=2):
        D, H, W, C = full_mat.shape
        D_new = (D // scale) + 1
        H_new = (H // scale) + 1
        W_new = (W // scale) + 1
        voxel = np.zeros([D_new, H_new, W_new, C])
        for i in range(D_new):
            for j in range(H_new):
                for k in range(W_new):
                    for x in range(scale):
                        for y in range(scale):
                            for z in range(scale):
                                if i * scale + x < D and j * scale + y < H and k * scale + z < W:
                                    voxel[i][j][k] += \
                                        full_mat[i * scale + x][j * scale + y][
                                            k * scale + z]
        return voxel
