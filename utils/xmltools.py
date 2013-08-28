import os
import re
import sys
import xml.dom.minidom as xdom

def str2list(string, dtype=str):
    string = re.sub(r"[, ]", " ", string)
    return [dtype(x) for x in string.split()]


def child2list(item_list, action=None):
    child_list = []
    for item in item_list:
        child = item.firstChild.data.split("\n")
        for data in child:
            child_list.extend([str(s.strip()) for s in data.split()])
    if action == "lower":
        child_list = [s.lower() for s in child_list]
    return child_list


def get_name_value(item):
    name = item.name.encode("utf-8").strip()
    value = fmtstr(item.value)
    return name, value


def uni2str(unistr):
    return unistr.encode("utf-8").strip()


def stringify(item, action=""):
    string = str(" ".join(item.split()))
    if action == "upper":
        return string.upper()
    if action == "lower":
        return string.lower()
    return string

def fmtstr(string):
    return " ".join(string.split())


if __name__ == "__main__":
    main()
