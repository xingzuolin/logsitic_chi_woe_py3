#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 11:14
# @Author  : Jun
# @File    : binning.py

import pandas as pd
import numpy as np
import time
from func import *
from scipy import stats
import itertools
import math
from numpy import array
import re


def chi2_value(arr):
    assert(arr.ndim == 2)
    rows_sum = arr.sum(axis=1)
    col_sum = arr.sum(axis=0)
    tot_sum = arr.sum()
    E = np.ones(arr.shape)*col_sum/tot_sum
    E = (E.T * rows_sum).T
    square_tmp = (arr-E)**2
    E[E == 0] = 1
    square = square_tmp/E
    v = square.sum()
    return v


def chi2_cut(df, col, max_groups=None, threshold=None):
    freq_tab = df.copy()
    # freq_tab.set_index(col, inplace=True)
    freq_tab.drop(col, axis=1, inplace=True)
    freq = freq_tab.values
    cutoffs = freq_tab.index.values
    if max_groups is None:
        if threshold is None:
            cls_num = freq.shape[-1]
            threshold = stats.chi2.isf(0.05, df=cls_num - 1)

    while True:
        minvalue = None
        minidx = None
        for i in range(len(freq) - 1):
            v = chi2_value(freq[i:i + 2])
            if minvalue is None or minvalue > v:
                minvalue = v
                minidx = i

        if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):
            tmp = freq[minidx] + freq[minidx + 1]
            freq[minidx] = tmp
            freq = np.delete(freq, minidx + 1, 0)
            cutoffs = np.delete(cutoffs, minidx + 1, 0)
        else:
            break

    # index_cutoffs = []
    # for i in cutoffs:
    #     tmp = df[df[col] == i].index[0]
    #     index_cutoffs.append(tmp)
    # if len(cutoffs) != len(index_cutoffs):
    #     print('the index of {0} is not match the cutoffs'.format(col))
    #     return []
    cutoffs_list = list(cutoffs)
    cutoffs_list.extend([0, len(freq_tab)-1])
    cutoffs_list = sorted(list(set(cutoffs_list)))
    return cutoffs_list


def cutoff_combine(data_df, cutoff_list, groups):
    t1 = 0
    t2 = len(data_df)-1
    list_tmp = cutoff_list[1: len(cutoff_list)-1]
    combine = []
    if len(cutoff_list)-2 < groups:
        c = len(cutoff_list) - 2
    else:
        c = groups - 1
    list1 = list(itertools.combinations(list_tmp, c))
    if list1:
        combine = list(map(lambda x: sorted(x + (t1 - 1, t2)), list1))
    return combine


def cal_iv(date_df,items,bad_name,good_name,rate,total_all):
    iv0 = 0
    total_rate = [sum(date_df.ix[x[0]:x[1],bad_name]+date_df.ix[x[0]:x[1],good_name])*1.0/total_all for x in items]
    if [k for k in total_rate if k < rate]:
        return 0
    bad0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1],bad_name]),items)))
    good0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1],good_name]),items)))
    bad_rate0 = bad0*1.0/(bad0 + good0)
    if 0 in bad0 or 0 in good0:
        return 0
    good_per0 = good0*1.0/sum(date_df[good_name])
    bad_per0 = bad0*1.0/sum(date_df[bad_name])
    woe0 = list(map(lambda x: math.log(x,math.e), good_per0/bad_per0))
    if sorted(woe0, reverse=False) == list(woe0) and sorted(bad_rate0, reverse=True) == list(bad_rate0):
        iv0 = sum(woe0*(good_per0-bad_per0))
    elif sorted(woe0, reverse=True) == list(woe0) and sorted(bad_rate0, reverse=False) == list(bad_rate0):
        iv0 = sum(woe0*(good_per0-bad_per0))
    return iv0


def choose_best_combine(data_df, combine, bad_name, good_name, rate, total_all):
    z = [0]*len(combine)
    for i in range(len(combine)):
        item = combine[i]
        z[i] = list(zip(map(lambda x: x+1, item[0:len(item)-1]), item[1:]))
    iv_list = list(map(lambda x: cal_iv(data_df,x,bad_name,good_name,rate,total_all),z))
    iv_max = max(iv_list)
    if iv_max == 0:
        return ''
    index_max = iv_list.index(iv_max)
    combine_max = z[index_max]
    return combine_max


def verify_woe(x):
    if re.match('^\d*\.?\d+$', str(x)):
        return x
    else:
        return 0


def best_df(date_df, items, na_df, factor_name, bad_name, good_name,total_all,good_all,bad_all):
    df0 = pd.DataFrame()
    if items:
        piece0 = list(map(lambda x: '('+str(date_df.ix[x[0],factor_name])+','+str(date_df.ix[x[1],factor_name])+')',items))
        bad0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1],bad_name]),items))
        good0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1],good_name]),items))
        if len(na_df) > 0:
            piece0 = array(list(piece0) + list(map(lambda x: '('+str(x)+','+str(x)+')',list(na_df[factor_name]))))
            bad0 = array(list(bad0) + list(na_df[bad_name]))
            good0 = array(list(good0) + list(na_df[good_name]))
        else:
            piece0 = array(list(piece0))
            bad0 = array(list(bad0))
            good0 = array(list(good0))
        total0 = bad0 + good0
        total_per0 = total0*1.0/total_all
        bad_rate0 = bad0*1.0/total0
        good_rate0 = 1 - bad_rate0
        good_per0 = good0*1.0/good_all
        bad_per0 = bad0*1.0/bad_all
        df0 = pd.DataFrame(list(zip(piece0,total0,bad0,good0,total_per0,bad_rate0,good_rate0,good_per0,bad_per0)),columns=[factor_name,'Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Good_Rate','Good_Pcnt','Bad_Pcnt'])
        df0.sort_values(by='Bad_Rate', ascending=False, inplace=True)
        df0.index = range(len(df0))
        bad_per0 = array(list(df0['Bad_Pcnt']))
        good_per0 = array(list(df0['Good_Pcnt']))
        bad_rate0 = array(list(df0['Bad_Rate']))
        good_rate0 = array(list(df0['Good_Rate']))
        bad_cum = np.cumsum(bad_per0)
        good_cum = np.cumsum(good_per0)
        woe0 = list(map(lambda x: math.log(x, math.e), good_per0/bad_per0))
        if 'inf' in str(woe0):
            woe0 = list(map(lambda x: verify_woe(x), woe0))
        iv0 = woe0*(good_per0-bad_per0)
        gini = 1-pow(good_rate0,2)-pow(bad_rate0,2)
        df0['Bad_Cum'] = bad_cum
        df0['Good_Cum'] = good_cum
        df0["Woe"] = woe0
        df0["IV"] = iv0
        df0['Gini'] = gini
        df0['KS'] = abs(df0['Good_Cum'] - df0['Bad_Cum'])
        df0['test'] = df0[factor_name].apply(lambda x: float(x.split(',')[0][1:]))
        df0.sort_values(by='test', ascending=True, inplace=True)
        df0.drop('test', axis=1, inplace=True)
        df0.index = range(len(df0))
    return df0


def cal_bin(date_df, na_df, groups, rate, var_name, bad_name, good_name,total_all,total_good,total_bad):
    group_sort = list(range(groups + 1))
    group_sort.sort(reverse=True)
    cutoffs_list = chi2_cut(date_df, var_name, max_groups=10)
    if not cutoffs_list:
        df1 = pd.DataFrame()
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1
    df1 = pd.DataFrame()
    for c in group_sort[:groups - 1]:
        combine = cutoff_combine(date_df, cutoffs_list, c)
        best_combine = choose_best_combine(date_df, combine, bad_name, good_name, rate, total_all)
        df1 = best_df(date_df, best_combine, na_df, var_name, bad_name, good_name, total_all, total_good, total_bad)
        if len(df1) != 0:
            gini = sum(df1['Gini'] * df1['Total_Num'] / sum(df1['Total_Num']))
            print('piece_count:', str(len(df1)))
            print('IV_All_Max:', str(sum(df1['IV'])))
            print('Best_KS:', str(max(df1['KS'])))
            print('Gini_index:', str(gini))
            print(df1)
            return df1
    if len(df1) == 0:
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
    return df1


def ywgt_check(mst, var):
    if var not in mst:
        print('{0} is not in the input DataFrame!'.format(var))


def df_pivot_two(data_df, var_name, target, good_name, bad_name):
    if len(data_df) == 0:
        print('Error: the data is wrong')
        return data_df
    # if target != '':
    #     try:
    #         data_df[target] = data_df[target].astype(int)
    #     except:
    #         print('Error: the data is wrong')
    #         data_df = pd.DataFrame()
    #         return data_df
    date_df = data_df[target].groupby(
        [data_df[var_name], data_df[target]]).count().unstack().reset_index().fillna(0)
    date_df.columns = [var_name, good_name, bad_name]
    date_df.sort_values(by=[var_name], ascending=True, inplace=True)
    date_df.index = range(len(date_df))
    return date_df


def df_null_group(data_df, var_name, target, good_name, bad_name,missing_fill):
    data_df = data_df.copy()
    data_df[var_name].fillna(missing_fill, inplace=True)
    date_df = data_df[target].groupby(
        [data_df[var_name], data_df[target]]).count().unstack().reset_index().fillna(0)
    date_df.columns = [var_name, good_name, bad_name]
    return date_df


def find_best_bin(path='', data=pd.DataFrame(), sep=',', subset=[0, 1], y=None,var_name=None,groups=5,rate=0.05):
    start_time = time.time()
    good_name = 'good'
    total_name = 'total'
    bad_name = 'bad'
    if path != '':
        data = read_path(path, sep)
    if len(data) == 0:
        print('Error: there is no data')
        return data
    col_list = convert_upper(list(data.columns))
    y = convert_upper(y)
    var_name = convert_upper(var_name)
    ywgt_check(col_list, y)
    ywgt_check(col_list, var_name)
    data = data[data[y].isin(subset)]
    data.index = range(len(data))
    total_good = data.ix[data[y] == 0, y].count()
    total_bad = data.ix[data[y] == 1, y].count()
    total_all = total_good + total_bad
    data_df_null = data[data[var_name].isnull()]
    data_df_notnull = data[data[var_name].notnull()]
    if str(data_df_notnull[var_name].dtype).find('object') < 0:
        data_df = df_pivot_two(data_df_notnull, var_name, y, good_name, bad_name)
        if len(data_df_null) > 0:
            na_df = df_null_group(data_df_null,var_name,y,good_name,bad_name, missing_fill=-999)
    # if str(data_df_notnull.dtype).find('object') >= 0:
    #     continue
    if len(data_df) == 0:
        return data_df

    bin_df = cal_bin(data_df, na_df, groups, rate, var_name, bad_name, good_name, total_all, total_good, total_bad)









