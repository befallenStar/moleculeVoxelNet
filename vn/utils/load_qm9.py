# -*- encoding: utf-8 -*-
import os
import tarfile

import numpy as np
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Hartree, eV, Bohr, Ang


class IsolatedAtomException(Exception):
    pass


def load_atomrefs(at_path):
    print('Downloading GDB-9 atom references...')
    # at_url = 'https://ndownloader.figshare.com/files/3195395'
    tmpdir = '..\\data'
    tmp_path = os.path.join(tmpdir, 'atomref.txt')

    # try:
    #     urllib.request.urlretrieve(at_url, tmp_path)
    #     logging.info("Done.")
    # except HTTPError as e:
    #     logging.error("HTTP Error:", e.code, at_url)
    #     return False
    # except URLError as e:
    #     logging.error("URL Error:", e.reason, at_url)
    #     return False

    atref = np.zeros((100, 6))
    labels = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
    with open(tmp_path) as f:
        lines = f.readlines()
        for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
            atref[z, 0] = float(l.split()[1])
            atref[z, 1] = float(l.split()[2]) * Hartree / eV
            atref[z, 2] = float(l.split()[3]) * Hartree / eV
            atref[z, 3] = float(l.split()[4]) * Hartree / eV
            atref[z, 4] = float(l.split()[5]) * Hartree / eV
            atref[z, 5] = float(l.split()[6])
    np.savez(at_path, atom_ref=atref, labels=labels)
    return True


def load_data(dbpath):
    print('Downloading GDB-9 data...')
    tmpdir = '..\\data'
    tar_path = os.path.join(tmpdir, 'dsgdb9nsd.xyz.tar.bz2')
    raw_path = os.path.join(tmpdir, 'gdb9_xyz')
    # url = 'https://ndownloader.figshare.com/files/3195389'

    # try:
    #     urllib.request.urlretrieve(url, tar_path)
    #     logging.info("Done.")
    # except HTTPError as e:
    #     logging.error("HTTP Error:", e.code, url)
    #     return False
    # except URLError as e:
    #     logging.error("URL Error:", e.reason, url)
    #     return False

    # tar = tarfile.open(tar_path)
    # tar.extractall(raw_path)
    # tar.close()

    basic_atoms = ['Atom_H', 'Atom_C', 'Atom_N', 'Atom_O', 'Atom_F', 'Atom_P',
                   'Atom_S', 'Atom_Cl', 'Atom_Br', 'Atom_I']
    prop_names = ['rcA', 'rcB', 'rcC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'energy_U0', 'energy_U', 'enthalpy_H',
                  'free_G', 'Cv']
    conversions = [1., 1., 1., 1., Bohr ** 3 / Ang ** 3,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Bohr ** 2 / Ang ** 2, Hartree / eV,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Hartree / eV, 1.]

    print('Parse xyz files...')
    with connect(dbpath) as con:
        for i, xyzfile in enumerate(os.listdir(raw_path)):
            '''
            structure of the xyz files:
            number:int, the amounts of atoms in the molecule
            properties:list, properties respond to the list builtin from third to the end
            
            the following lines represent atoms
            an atom a line with the formatter of (symbol, x, y, z, initial charge)
            '''
            xyzfile = os.path.join(raw_path, xyzfile)

            if i % 10000 == 0:
                print('Parsed: ' + str(i) + ' / 133885')
            properties = {}
            charges = {a: 0 for a in basic_atoms}
            # put the content into a temp file
            tmp = os.path.join(tmpdir, 'tmp.xyz')

            # parse the XYZ files
            # get the number of atoms
            # get the properties
            # get the numbers and the charges of each atom
            with open(xyzfile, 'r') as f:
                lines = f.readlines()
                # read the properties
                l = lines[1].split()[2:]

                # do preprocessing
                for pn, p, c in zip(prop_names, l, conversions):
                    properties[pn] = float(p) * c
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace('*^', 'e'))

            with open(tmp, 'r') as f:
                lines = f.readlines()
                # get the number
                cnt = int(lines[0])

                # get the numbers and the charges
                atoms = lines[2:cnt + 2]
                for atom in atoms:
                    a, _, _, _, c = atom.split()
                    a = 'Atom_' + a
                    charges[a] += float(c)
                properties.update(charges)

                # a function from ase module, which can read from XYZ formatter
                ats = list(read_xyz(f, 0))[0]

            # idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = \
            #     collect_neighbors(ats, 20.)

            # data = {'_idx_ik': idx_ik, '_idx_jk': idx_jk, '_idx_j': idx_j,
            #         '_seg_i': seg_i, '_seg_j': seg_j, '_offset': offset,
            #         '_ratio_j': ratio_j}
            con.write(ats, key_value_pairs=properties)
    print('Done.')

    return True


if __name__ == '__main__':
    # load_atomrefs('..\\data\\atomref.npz')
    load_data('..\\data\\qm9.db')
