#!/usr/bin/env python
import os
import numpy as np
import h5py


def read_hdf5(filename, columns=None, disp=1):

    F = h5py.File(filename, 'r')

    head = sorted(F.keys())
    data = np.array([F[_] for _ in head]).T

    F.close()

    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, str) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, str):
                    columns[i] = h.index(item.lower())

        if head is not None:
            head = [head[i] for i in columns]
        data = data[:, columns]

    if not disp:
        return data
    return head, data


def write_hdf5(filename, head, data, columns=None):
    #
    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, str) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, str):
                    columns[i] = h.index(item.lower())

        head = [head[i] for i in columns]
        data = data[:, columns]

    F = h5py.File(filename, 'w')
    for idx, key in enumerate(head):
        F.create_dataset(key, data=data[:, idx])
    F.close()
