# -*- encoding: utf-8 -*-
import numpy as np

atoms_vector = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


class VoxelLoader:
    def __init__(self, data, xyz_index=(0, 1, 2)):
        self.d = np.array(data)
        self.x, self.y, self.z = xyz_index

    def __call__(self, mag_coeff=100, sigma=1):
        xyz_range = self.__xyz_range__()
        threeD_xyz = self.__get_3D_matrix__(self.d, xyz_range, mag_coeff)
        full_mat = self.__gaussian_smoothing__(threeD_xyz, sigma)
        return full_mat

    def __get_3D_matrix__(self, full_list, xyz_range, mag_coeff):
        lx, ly, lz, hx, hy, hz = xyz_range
        magnifier = int(mag_coeff)
        if hx - lx != 0:
            magnifier = min(int(mag_coeff) / (hx - lx), magnifier)
        if hy - ly != 0:
            magnifier = min(int(mag_coeff) / (hy - ly), magnifier)
        if hz - lz != 0:
            magnifier = min(int(mag_coeff) / (hz - lz), magnifier)
        dx, dy, dz = list(map(int,
                              [(hx - lx) * magnifier + 1,
                               (hy - ly) * magnifier + 1,
                               (hz - lz) * magnifier + 1]))
        threeD_xyz = np.zeros(shape=[dx, dy, dz, len(atoms_vector)],
                              dtype=np.float32)
        for itm in full_list:
            x, y, z = itm[[self.x, self.y, self.z]]
            idx = atoms_vector.index(int(itm[-1]))
            threeD_xyz[int((x - lx) * magnifier)][
                int((y - ly) * magnifier)][
                int((z - lz) * magnifier)][idx] = 1.
        return threeD_xyz

    def __gaussian_smoothing__(self, full_mat, sigma=1):
        # print(full_mat.shape)
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
                        atom[idx] *= gau
                        full[xx[count] + x + sigma][yy[count] + y + sigma][
                            zz[count] + z + sigma] += atom
                        p = full[xx[count] + x + sigma][yy[count] + y + sigma][
                            zz[count] + z + sigma][idx]
                        if p > 1:
                            full[xx[count] + x + sigma][yy[count] + y + sigma][
                                zz[count] + z + sigma][idx] = self.__sigmoid__(
                                p)
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
