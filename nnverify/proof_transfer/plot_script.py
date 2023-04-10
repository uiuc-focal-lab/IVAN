import argparse
import os

import pickle5 as pickle
import sys
sys.path.append('.')

from nnverify.common.result import Results

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nnverify import common
from nnverify.common import Status
from nnverify.proof_transfer import pt_util
from tabulate import tabulate


def get_accuracy_num(res):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output != Status.MISS_CLASSIFIED:
            count += 1
    return count


def get_ver_num(res):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output == Status.VERIFIED:
            count += 1
    return str(count)


def get_adv_num(res):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output == Status.ADV_EXAMPLE:
            count += 1
    return str(count)


def get_timeout_num(res):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output == Status.UNKNOWN:
            count += 1
    return str(count)


def get_avg_tree_size(res, res_ivan):
    sum_size = 0
    count = 0
    for i in range(len(res.results_list)):
        if res_ivan.results_list[i].ver_output != Status.UNKNOWN and res_ivan.results_list[i].ver_output != Status.MISS_CLASSIFIED:
            sum_size += res.results_list[i].tree_size
            count += 1
    return str(round(sum_size/count, 2))


def smallt_solved(res, res_base):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output != Status.UNKNOWN and res.results_list[i].ver_output != Status.MISS_CLASSIFIED\
                and res_base.results_list[i].tree_size <= 5:
            count += 1
    return count


def bigt_solved(res, res_base):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output != Status.UNKNOWN and res.results_list[i].ver_output != Status.MISS_CLASSIFIED\
                and res_base.results_list[i].tree_size > 5:
            count += 1
    return count


def smallt_time(res, res_base):
    count = 0
    for i in range(len(res.results_list)):
        if res.results_list[i].ver_output != Status.UNKNOWN and res.results_list[i].ver_output != Status.MISS_CLASSIFIED\
                and res_base.results_list[i].tree_size <= 5:
            count += res.results_list[i].time
    return str(round(count, 2))


def bigt_time(res, res_base, res_ivan):
    count = 0
    for i in range(len(res.results_list)):
        if res_ivan.results_list[i].ver_output != Status.UNKNOWN and res.results_list[i].ver_output != Status.MISS_CLASSIFIED\
                and res_base.results_list[i].tree_size > 5:
            count += res.results_list[i].time
    return str(round(count, 2))


def all_time(res, res_base):
    l1 = []
    l2 = []
    for i in range(len(res.results_list)):
        if res_base.results_list[i].ver_output != Status.UNKNOWN and res_base.results_list[i].ver_output != Status.MISS_CLASSIFIED:
                # and res_base.results_list[i].time > 10:
            l1.append(res.results_list[i].time)
            l2.append(res_base.results_list[i].time)
    return l1, l2


def generate_stats(i, plot=False):
    pt_args, res, res_pt = get_results(i)

    if plot:
        pt_util.plot_scatter(pt_args, res, res_pt, plot_name='scatter' + str(i+1))

    return [get_accuracy_num(res), get_ver_num(res) + '/' + get_adv_num(res) + '/' + get_timeout_num(res),
            get_ver_num(res_pt) + '/' + get_adv_num(res_pt) + '/' + get_timeout_num(res_pt),
            get_avg_tree_size(res, res_pt), get_avg_tree_size(res_pt, res_pt),
            smallt_solved(res, res), smallt_solved(res_pt, res), smallt_time(res, res), smallt_time(res_pt, res),
            bigt_solved(res, res), bigt_solved(res_pt, res), bigt_time(res, res, res_pt), bigt_time(res_pt, res, res_pt)
            ]


def get_results(i):
    file_name = common.RESULT_DIR + file_names[i]
    with open(file_name, 'rb') as inp:
        uu = pickle.load(inp)
        if len(uu) == 3:
            [pt_args, res, res_pt] = uu
        else:
            [res, res_pt] = uu
    return pt_args, res, res_pt


def stat_table(plot=None):
    rows = [['N', 'base_v/base_a/base_u', 'ivan_v/ivan_a/ivan_u', 'avg_tree\n_base', 'avg_tree\n_ivan',
             'smallt\n_base\nsolved', 'smallt\n_ivan_\nsolved', 'smallt\n_base_\ntime', 'smallt\n_ivan_\ntime',
             'bigt\n_base_\nsolved', 'bigt\n_ivan_\nsolved', 'bigt\n_base_\ntime', 'bigt\n_ivan_\ntime']]
    if args.num == -1:
        for i in range(len(file_names)):
            rows.append(generate_stats(i, plot=plot))
    else:
        rows.append(generate_stats(args.num, plot=plot))
    print(tabulate(rows, tablefmt="simple"))

    # Write to latex file
    dir_name = common.RESULT_DIR
    os.makedirs(dir_name, exist_ok=True)
    file_name = dir_name + 'table_stats.tex'
    with open(file_name, 'w') as f:
        f.write(tabulate(rows, tablefmt="latex"))


def sensitivity_plot():
    def get_float(element):
        try:
            float(element)
            return float(element)
        except ValueError:
            return None

    sns.set_style('darkgrid')
    ax = plt.subplot()
    ax.set_xlabel(r'$\alpha$')

    import re

    data = []
    with open("nnverify/proof_transfer/temp2.txt", 'r') as file:
        for line in file:
            flts = []
            for token in re.split(',|=|\n', line):
                if get_float(token) is not None:
                    flts.append(get_float(token))
            data.append(flts[2])
    x = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    y = [0, 0.25, 5, 0.75, 1]
    data = np.array(data)

    data = data.reshape((6, 5))
    s = sns.heatmap(data, annot=True, linewidths=.5, cmap="crest", yticklabels=x, xticklabels=y)
    s.set_xlabel(r'$\alpha$', fontsize=20)
    s.set_ylabel(r'$\theta$', fontsize=20)

    dir_name = common.RESULT_DIR + 'line_plot/'
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name + 'sensitivity', dpi=300)
    plt.close('all')


def query_stats(num):
    pt_args, res_base, res_ivan = get_results(num)

    query_cases = []

    for i in range(len(res_base.results_list)):
        if res_ivan.results_list[i].ver_output != Status.UNKNOWN and res_base.results_list[i].ver_output != Status.MISS_CLASSIFIED:
            # --------------Write the query here ------------------#
            spi = res_base.results_list[i].time/res_ivan.results_list[i].time
            if res_base.results_list[i].ver_output == Status.UNKNOWN:
                query_cases.append(i)

    res = Results(pt_args)
    res_pt = Results(pt_args)

    for q in query_cases:
        res.add_result(res_base.results_list[q])
        res_pt.add_result(res_ivan.results_list[q])

    return [get_accuracy_num(res), get_ver_num(res), get_adv_num(res), get_timeout_num(res),
            get_ver_num(res_pt), get_adv_num(res_pt), get_timeout_num(res_pt),
            get_avg_tree_size(res, res_pt), get_avg_tree_size(res_pt, res_pt),
            smallt_solved(res, res), smallt_solved(res_pt, res), smallt_time(res, res), smallt_time(res_pt, res),
            bigt_solved(res, res), bigt_solved(res_pt, res), bigt_time(res, res, res_pt), bigt_time(res_pt, res, res_pt)
            ]


def query():
    rows = [['N', 'base_v', 'base_a', 'base_u', 'ivan_v', 'ivan_a', 'ivan_u', 'avg_tree\n_base', 'avg_tree\n_ivan',
             'smallt\n_base\nsolved', 'smallt\n_ivan_\nsolved', 'smallt\n_base_\ntime', 'smallt\n_ivan_\ntime',
             'bigt\n_base_\nsolved', 'bigt\n_ivan_\nsolved', 'bigt\n_base_\ntime', 'bigt\n_ivan_\ntime']]
    if args.num == -1:
        for i in range(len(file_names)):
            rows.append(query_stats(i))
    else:
        rows.append(query_stats(args.num))
    print(tabulate(rows, tablefmt="simple"))


def avg_time():
    import statistics
    from scipy.stats import gmean
    if args.num == -1:
        speedups = []
        #for i in range(len(file_names)):
        for i in range(10, 12):
            pt_args, res_base, res_ivan = get_results(i)
            l1, l2 = all_time(res_ivan, res_base)
            # sum(l1)
            speedups += [sum(l2)/sum(l1)]
            # speedups += [l2[i]/l1[i] for i in range(len(l1))]

        print('Median: ', statistics.median(speedups))
        print('Arithmetic Mean:', statistics.mean(speedups))
        print('Geometric Mean: ', gmean(speedups))
    else:
        pt_args, res_base, res_ivan = get_results(args.num)
        print(all_time(res_ivan, res_base))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--sensitivity', type=bool, default=False)
    parser.add_argument('--query', type=bool, default=False)
    parser.add_argument('--avg_time', type=bool, default=False)
    args = parser.parse_args()

    if args.sensitivity:
        sensitivity_plot()
    elif args.query:
        query()
    elif args.avg_time:
        avg_time()
    else:
        stat_table(args.plot)
