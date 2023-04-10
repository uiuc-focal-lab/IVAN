import csv
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from time import gmtime, strftime
from nnverify import common, config
from nnverify.common import Status


def result_resolved(res):
    return res.ver_output != Status.UNKNOWN and res.ver_output != Status.MISS_CLASSIFIED


def compute_speedup(res, res_pt, pt_args):
    approx_time = 0
    prev_time = 0
    reduced_timeout = 0
    for i in range(len(res_pt.results_list)):
        # Not unknown on the vanilla approx network
        if result_resolved(res.results_list[i]):
            approx_time += res_pt.results_list[i].time
            prev_time += res.results_list[i].time

        elif result_resolved(res_pt.results_list[i]):
            reduced_timeout += 1

    print("Previous time: ", prev_time, "Approx time: ", approx_time)
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'proof_transfer.csv'
    speedup = np.float64(prev_time) / np.float64(approx_time)
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['Proof Transfer Result at', strftime("%Y-%m-%d %H:%M:%S", gmtime())])
        writer.writerow([pt_args.net, pt_args.dataset, pt_args.approximation, 'count:', pt_args.count])
        writer.writerow(['prev branches: ', res.avg_tree_size, 'approx branches:', res_pt.avg_tree_size])
        writer.writerow(['prev time: ', prev_time, 'approx time:', approx_time,
                         'speedup:', speedup, 'extra completed:', reduced_timeout])
    return speedup


def plot_verification_results(res, res_pt, pt_args):
    if pt_args.count <= 1:
        print("Not plotting since the results size is <=1")
        return

    plot_line_graph(pt_args, res, res_pt)
    plot_scatter(pt_args, res, res_pt)

    # Save results
    dir_name = common.RESULT_DIR + 'pickle/'
    os.makedirs(dir_name, exist_ok=True)
    with open(dir_name + res.get_file_name(pt_args), 'wb') as opfile:
        pickle.dump([pt_args, res, res_pt], opfile, pickle.HIGHEST_PROTOCOL)


def plot_line_graph(pt_args, res, res_pt):
    sns.set_style('darkgrid')
    ax = plt.subplot()
    ax.set_xlabel('# Solved')
    ax.set_ylabel('Time')
    total = len(res_pt.results_list)
    ax.set_xticks([int(total * i / 5) for i in range(5)])
    to_plot = []
    for i in range(len(res_pt.results_list)):
        if res_pt.results_list[i].ver_output != Status.UNKNOWN \
                and res_pt.results_list[i].ver_output != Status.MISS_CLASSIFIED:
            to_plot.append(i)
    h1 = get_line_plot(ax, res, config.baseline, to_plot)
    h2 = get_line_plot(ax, res_pt, config.tool_name, to_plot)
    ax.legend(handles=[h1, h2])
    dir_name = common.RESULT_DIR + 'line_plot/'
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(dir_name + res.get_plot_file_name(pt_args))
    plt.close('all')


def plot_scatter(pt_args, res, res_pt, plot_name=None):
    sns.set_style('darkgrid')
    ax = plt.subplot()
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Speedup', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.rcParams.update({'font.size': 17})

    ax.set_yscale('symlog', base=2)
    ax.set_xscale('symlog', base=2)

    from matplotlib.ticker import ScalarFormatter
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    x = []
    y = []

    for i in range(len(res_pt.results_list)):
        if result_resolved(res_pt.results_list[i]):

            speedup = (res.results_list[i].time / res_pt.results_list[i].time)
            ti = res.results_list[i].time

            x.append(ti)
            y.append(speedup)

    if len(y) <= 1 or len(x) <= 1:
        print("Not enough data to plot!")
        return

    y_ticks = [0]
    cur = 1
    while cur < 2*max(y):
        y_ticks.append(cur)
        cur *= 2
    ax.set_yticks(y_ticks)

    print(y_ticks)
    ax.set_ylim([0, 1.1*max(y_ticks)])

    ax.axline((pt_args.timeout, 0), (pt_args.timeout, 2*max(y_ticks)), color='C3', label='timeout')
    ax.axline((0, 1), (1.2*max(x), 1), linestyle='dashed', color='C2')

    plt.legend()
    plt.scatter(x, y, marker='x')
    # plt.scatter(x2, y2, marker='x')

    dir_name = common.RESULT_DIR + 'scatter_plot/'
    os.makedirs(dir_name, exist_ok=True)
    if plot_name is None:
        plot_name = res.get_file_name(pt_args)
    plt.savefig(dir_name + plot_name, dpi=300)
    plt.close('all')


def get_line_plot(ax, res, label, to_plot):
    x1 = []
    y1 = []
    cur_time1 = 0
    for i in to_plot:
        cur_time1 += res.results_list[i].time
        if result_resolved(res.results_list[i]):
            y1.append(cur_time1)
            x1.append(len(y1))
    h1, = ax.plot(x1, y1, label=label)
    return h1
