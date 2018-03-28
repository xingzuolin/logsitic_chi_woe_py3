#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 11:14
# @Author  : Jun
# @File    : func.py

import pandas as pd


def read_path(path, sep):
    data = pd.read_csv(path, sep=sep)
    return data


def convert_upper(data):
    if isinstance(data,list):
        data = [var.upper() for var in data]
    elif isinstance(data,str):
        data.upper()
    return data