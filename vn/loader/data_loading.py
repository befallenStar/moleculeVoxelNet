# -*- encoding: utf-8 -*-
import os

from ase.db import connect

from vn.loader.pointcloud2voxel import load_atoms


def load_db(path='..\\data', ase_db='ase.db'):
    ase_path = os.path.join(path, ase_db)
    conn = connect(ase_path)
    atomses, propertieses,datas = [], [],[]
    for row in conn.select():
        atoms = row.toatoms()
        properties = row.key_value_pairs
        data=row.data
        atomses.append(atoms)
        propertieses.append(properties)
        datas.append(data)
    return atomses, propertieses,datas


def merge(datapath='..\\data', destination='..\\ase_data', ase_db='ase.db'):
    atomses = []
    for file in os.listdir(datapath):
        # print(file)
        filepath = os.path.join(datapath, file)
        conn = connect(filepath)
        for row in conn.select():
            atoms = row.toatoms()
            atomses.append(atoms)

    for atoms in atomses:
        ase_path = os.path.join(destination, ase_db)
        conn = connect(ase_path)
        conn.write(atoms)
    print("{} atoms have been merged".format(len(atomses)))


def main():
    path = '../data'
    # cids = [3, 7, 11, 12, 13, 19, 21, 22, 29, 33, 34, 35, 44, 45, 49]
    for cid in range(1000):
        try:
            atoms = load_atoms(cid=cid + 1)
            # print('cid: ' + str(cid))
            name = atoms.symbols
            # print('atoms: ' + str(name))
            filepath = os.path.join(path, str(name) + '.db')
            print('filepath: ' + filepath)
            if not os.path.exists(filepath):
                conn = connect(filepath)
                conn.write(atoms)
            print(str(cid + 1) + " " + str(name) + ' done')
        except ValueError as e:
            print(cid + 1)
            print(str(e))


if __name__ == '__main__':
    # main()

    # merge(ase_db='ase-1000.db')

    atomses,_,_ = load_db(path='../ase_data', ase_db='qm9_25_mini.db')
    for step,atoms in enumerate(atomses):
        print(step,atoms.symbols,atoms.positions)

    # ase_path = "..\\ase_data\\qm9_15.db"
    # conn = connect(ase_path)
    # cnt = 1
    # with connect("..\\ase_data\\qm9_15_medium.db") as con:
    #     for row in conn.select():
    #         if cnt > 10000:
    #             break
    #         atoms = row.toatoms()
    #         properties = row.key_value_pairs
    #         data = row.data
    #         con.write(atoms, key_value_pairs=properties, data=data)
    #         cnt += 1
