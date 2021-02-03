# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd


def main():
    QM9_nano=np.load('../data/QM9_nano_USTC/QM9_nano.npz', allow_pickle=True)
    print(QM9_nano.files)
    print(QM9_nano['ID'])
    print(QM9_nano['Atoms'])
    print(QM9_nano['Distance'])
    print(QM9_nano['U0'])
    # BOB=pd.read_csv('../data/QM9_nano_USTC/BOB.tsv',sep='\t',names=['ID',*[i for i in range(2209)]])
    # print(BOB.head())
    CM=pd.read_csv('../data/QM9_nano_USTC/CM.tsv', sep='\t')
    print(CM.head())
    ECFP4=pd.read_csv('../data/QM9_nano_USTC/ECFP4.tsv', sep='\t')
    print(ECFP4.head())

if __name__ == '__main__':
    main()