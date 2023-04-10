import copy
import torch

from enum import Enum

import nnverify.domains
from nnverify import parse
from nnverify.bnb import Split
from nnverify.domains.deepz import ZonoTransformer
from nnverify.specs.out_spec import OutSpecType


class InputSpecType(Enum):
    LINF = 1
    PATCH = 2
    GLOBAL = 3


class InputSpec(object):
    def __init__(self, input_lb, input_ub, out_constr, dataset, input=None):
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.out_constr = out_constr
        self.dataset = dataset
        if input is not None:
            self.input = input.flatten()
        else:
            self.input = None

    def __hash__(self):
        return hash((self.input_lb.numpy().tobytes(), self.input_ub.numpy().tobytes(), self.dataset))

    # After has collision Python dict check for equality. Thus, our definition of equality should define both
    def __eq__(self, other):
        if not torch.all(self.input_lb == other.input_lb) or not torch.all(self.input_ub == other.input_ub) \
                or self.dataset != other.dataset or not torch.all(
            self.out_constr.constr_mat[0] == other.out_constr.constr_mat[0]):
            return False
        return True

    def create_split_input_spec(self, input_lb, input_ub):
        return InputSpec(input_lb, input_ub, self.out_constr, self.dataset)

    def is_local_robustness(self):
        return self.out_constr.constr_type == OutSpecType.LOCAL_ROBUST

    def get_label(self):
        if self.out_constr.constr_type is not OutSpecType.LOCAL_ROBUST:
            raise ValueError("Label only for local robustness properties!")
        return self.out_constr.label

    def get_input_size(self):
        return self.input_lb.shape[0]

    def is_conjunctive(self):
        return self.out_constr.is_conjunctive

    def output_constr_mat(self):
        return self.out_constr.constr_mat[0]

    def output_constr_const(self):
        return self.out_constr.constr_mat[1]

    def split_spec(self, split, chosen_dim):
        if split == Split.INPUT or split == Split.INPUT_GRAD or split == Split.INPUT_SB:
            # Heuristic: Divide in 2 with longest width for now
            # choose a particular dimension of the input to split
            ilb1 = copy.deepcopy(self.input_lb)
            iub1 = copy.deepcopy(self.input_ub)

            iub1[chosen_dim] = (self.input_ub[chosen_dim] + self.input_lb[chosen_dim]) / 2

            ilb2 = copy.deepcopy(self.input_lb)
            iub2 = copy.deepcopy(self.input_ub)

            ilb2[chosen_dim] = (self.input_ub[chosen_dim] + self.input_lb[chosen_dim]) / 2

            return [self.create_split_input_spec(ilb1, iub1), self.create_split_input_spec(ilb2, iub2)]
        else:
            raise ValueError("Unsupported input split!")

    def multiple_splits(self, num_splits):
        all_splits = []
        new_ilb = copy.deepcopy(self.input_lb)
        new_iub = copy.deepcopy(self.input_ub)
        step_size = []
        # Assuming ACAS-XU for now
        for i in range(5):
            step_size.append((self.input_ub[i] - self.input_lb[i]) / num_splits[i])

        for i in range(num_splits[0]):
            new_ilb[0] = self.input_lb[0] + i * step_size[0]
            new_iub[0] = self.input_lb[0] + (i + 1) * step_size[0]
            for j in range(num_splits[1]):
                new_ilb[1] = self.input_lb[1] + j * step_size[1]
                new_iub[1] = self.input_lb[1] + (j + 1) * step_size[1]
                for k in range(num_splits[2]):
                    new_ilb[2] = self.input_lb[2] + k * step_size[2]
                    new_iub[2] = self.input_lb[2] + (k + 1) * step_size[2]
                    for l in range(num_splits[3]):
                        new_ilb[3] = self.input_lb[3] + l * step_size[3]
                        new_iub[3] = self.input_lb[3] + (l + 1) * step_size[3]
                        for m in range(num_splits[4]):
                            new_ilb[4] = self.input_lb[4] + m * step_size[4]
                            new_iub[4] = self.input_lb[4] + (m + 1) * step_size[4]

                            all_splits.append(
                                self.create_split_input_spec(copy.deepcopy(new_ilb), copy.deepcopy(new_iub)))
        return all_splits

    def get_zono_lb(self, net, s1):
        z1 = ZonoTransformer(s1)
        z1 = nnverify.domains.build_transformer(z1, net, self)
        lb1, _, _ = z1.compute_lb(complete=True)
        if lb1 is None:
            lb1 = 0
        return lb1


def merge_input_specs(input_spec_list):
    ilb = torch.stack([input_spec.input_lb for input_spec in input_spec_list])
    iub = torch.stack([input_spec.input_ub for input_spec in input_spec_list])
    return InputSpec(ilb, iub, input_spec_list[0].out_constr, input_spec_list[0].dataset)
