import csv
import os

from nnverify import common
from nnverify.common import RESULT_DIR, strip_name


class Result:
    def __init__(self, _time, _ver_output, tree_size=1):
        self.time = _time
        self.ver_output = _ver_output
        self.tree_size = tree_size


class Results:
    def __init__(self, args):
        self.avg_time = 0
        self.avg_tree_size = 0
        self.output_count = {}
        self.args = args
        self.results_list = []

    def add_result(self, result):
        self.results_list.append(result)

    # TODO: These stats may need to be computed differently with patch specification
    # ,or in general perturbations that generate multiple specifications per image
    def compute_stats(self):
        count = len(self.results_list)
        for res in self.results_list:
            self.avg_time += (res.time / count)
            self.avg_tree_size += (res.tree_size/count)
            if res.ver_output not in self.output_count:
                self.output_count[res.ver_output] = 0
            self.output_count[res.ver_output] += 1

        dir_name = RESULT_DIR + 'csv/'


        file_name = dir_name + self.get_csv_file_name()
        with open(file_name, 'a+') as f:
            writer = csv.writer(f)

            for i in range(len(self.results_list)):
                res = self.results_list[i]
                writer.writerow([i, res.ver_output, ' tree size:', res.tree_size, ' time:', res.time])

            writer.writerow(['Average time:', self.avg_time, ' Average tree size', self.avg_tree_size])
            writer.writerow([self.output_count])

    def get_csv_file_name(self):
        file_name = strip_name(self.args.domain) + '_' + strip_name(self.args.split) + '_' + strip_name(
            self.args.net, pos=-2) \
                    + '_' + str(self.args.eps) + '.csv'
        return file_name

    def get_plot_file_name(self, pt_args):
        file_name = 'plot_' + strip_name(self.args.domain) + '_' + strip_name(self.args.split) + '_' + strip_name(
            self.args.net, pos=-2) \
                    + '_' + str(self.args.eps) + str(pt_args.approximation) + '_' + str(pt_args.pt_method) + '.png'
        return file_name

    def get_file_name(self, pt_args):
        file_name = strip_name(self.args.domain) + '_' + strip_name(self.args.split) + '_' + strip_name(
            self.args.net, pos=-2) \
                    + '_' + str(self.args.eps) + '_' + str(pt_args.approximation) + '_' + str(pt_args.pt_method) + '.png'
        return file_name
