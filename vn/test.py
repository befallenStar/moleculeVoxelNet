# -*- encoding: utf-8 -*-
from loader.data_loading import load_db


def main():
    qm9, propertieses = load_db(path='ase_data', ase_db='qm9.db')
    for i in range(10):
        print(qm9[i].positions, propertieses[i])


if __name__ == '__main__':
    main()
