# -*- encoding: utf-8 -*-
import mayavi.mlab
import numpy as np
import traits.trait_errors as traits

atoms_vector = [1, 6, 7, 8, 16]
# colors = [x/10 for x in range(-10,10,20//len(atoms_vector))]
colors = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualization(data):
    dict = {}
    try:
        xx, yy, zz, sub = np.where(data != 0)
        # print(xx.shape)

        cnt = -1
        for count in range(len(xx)):
            c=np.zeros([3])
            if cnt != -1:
                cnt -= 1
                continue
            item = data[xx[count]][yy[count]][zz[count]]
            idx = np.where(item != 0)
            if len(idx[0]) == 1:
                p = data[xx[count]][yy[count]][zz[count]][sub[count]]
                c = (1 - p) * np.ones([3])+p*np.array(colors[sub[count]])
            else:
                for i in idx[0]:
                    if item[i] != 0:
                        c += (1 - item[i]) * np.ones([3]) + item[i] * np.array(
                            colors[i])
                        cnt += 1
                cnt -= 1
            c = tuple(c.tolist())
            # print(c)
            if c in dict.keys():
                dict[c].append([xx[count], yy[count], zz[count]])
            else:
                dict[c] = [[xx[count], yy[count], zz[count]]]
        # print(len(dict.keys()))
        keys = []
        for k in dict.keys():
            keys.append(k)
            print(k)
            print(dict[k])
        keys = np.array(keys)
        # print(keys)
        # print(keys.max())
        max = keys.max()

        for k in dict.keys():
            # print(dict[k])
            positions = np.array(dict[k])
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            k = np.array(k) / max
            k = tuple(k)
            nodes = mayavi.mlab.points3d(x, y, z, color=k, scale_mode='none',
                                         scale_factor=1)

        print('data prepared...')

        mayavi.mlab.show()
    except traits.TraitError as e:
        print(dict.keys())
        print(str(e))
