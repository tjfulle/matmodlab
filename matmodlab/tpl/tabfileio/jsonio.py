#!/usr/bin/env python
import numpy as np
import json


def read_json(filename, columns=None, disp=1):

    # Cannot read as bytes, must be string
    with open(filename, 'r') as F:
        head, data = json.load(F)

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


def write_json(filename, head, data, columns=None):

    # If specific columns are requested, filter the data
    if columns is not None:
        if any(isinstance(x, str) for x in columns):
            h = [s.lower() for s in head]
            for (i, item) in enumerate(columns):
                if isinstance(item, str):
                    columns[i] = h.index(item.lower())

        head = [head[i] for i in columns]
        data = data[:, columns]

    if type(head) == np.ndarray:
        head = head.tolist()
    if type(data) == np.ndarray:
        data = data.tolist()

    # Cannot write as bytes, must be string
    with open(filename, 'w') as F:
        json.dump([head, data], F)
