#!/usr/bin/env python
import os
import numpy as np
import pickle


def read_pickle(filename, columns=None, disp=1):
    with open(filename, 'rb') as F:
        head, data = pickle.load(F)

    data = np.array(data)

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


def write_pickle(filename, head, data, columns=None):
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

    with open(filename, 'wb') as F:
        pickle.dump([head, data], F)
