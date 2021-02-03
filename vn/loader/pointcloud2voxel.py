# -*- encoding: utf-8 -*-
from ase import Atoms
from ase.data.pubchem import pubchem_conformer_search

from vn.loader.pointclouds_to_voxelgrid import *
from vn.loader.visualize_npy import visualization

colors = {1: (1, 1, 1), 6: (.5, .5, .5), 7: (0, 0, 1),
          8: (1, 1, 0), 15: (1, 0, 1), 16: (1, 0, 0), 17: (0, 1, 0)}
atoms_vector = [1, 6, 7, 8, 15, 16, 17]


def load_atoms(cid=None, smiles=None) -> Atoms:
    inputs = [cid, smiles]
    input_check = [a is not None for a in inputs]
    if input_check.count(True) == 1:
        index = input_check.index(True)
        if index == 0:
            compounds = pubchem_conformer_search(cid=cid)
        else:
            compounds = pubchem_conformer_search(smiles=smiles)
    else:
        return Atoms()
    return compounds[0].atoms


def load(atoms: Atoms):
    # print(atoms.symbols)
    pointcloud = []
    for step, number in enumerate(atoms.numbers):
        point = atoms.positions[step].tolist()
        # point.extend(colors[number])
        point.append(number)
        pointcloud.append(point)
    return pointcloud


def load_voxel(atoms, *size, sigma=1):
    """
    turn an Atoms object to a voxel
    :param atoms: Atoms object
    :param mag_coeff: size of grids of voxel
    :param sigma: the kernel of gaussian smooth
    :return: a voxel represent the molecule
    """
    pointcloud = load(atoms)
    data_loader = VoxelLoader(pointcloud)
    full_mat = data_loader(*size,sigma=sigma)
    # D, H, W, C = full_mat.shape
    # D_new = ((D >> 3) + 1) << 3
    # H_new = ((H >> 3) + 1) << 3
    # W_new = ((W >> 3) + 1) << 3
    # voxel = np.zeros([D_new, H_new, W_new, C])
    # voxel[(D_new - D) >> 1:((D_new - D) >> 1) + D,
    # (H_new - H) >> 1:((H_new - H) >> 1) + H,
    # (W_new - W) >> 1:((W_new - W) >> 1) + W, :] = full_mat
    return full_mat


def main():
    cid = 31525001
    atoms = load_atoms(cid=cid)
    # smiles = "CNO"
    # atoms=load_atoms(smiles=smiles)
    pointcloud = load(atoms)
    data_loader = VoxelLoader(pointcloud)
    # full_mat=data_loader(mag_coeff=100,sigma=1)
    # visualization(full_mat)
    for mag_coeff, sigma in zip([20], [2]):
        full_mat = data_loader(mag_coeff=mag_coeff, sigma=sigma)
        visualization(full_mat)


if __name__ == '__main__':
    main()
