import copy
import random
import torch

import nnverify.domains
from nnverify import config
from nnverify.bnb import Split, is_relu_split, is_input_split
from nnverify.common import Status
from nnverify.domains.deepz import ZonoTransformer
from nnverify.specs.spec import Spec, SpecList


def branch_unsolved(spec_list, split, split_score=None, inp_template=None, args=None, net=None, transformer=None):
    new_spec_list = SpecList()
    verified_specs = SpecList()

    for spec in spec_list:
        if spec.status == Status.UNKNOWN:
            add_spec = split_spec(spec, split, split_score=split_score,
                                  inp_template=inp_template,
                                  args=args, net=net, transformer=transformer)
            new_spec_list += SpecList(add_spec)
        else:
            verified_specs.append(spec)
    return new_spec_list, verified_specs


def split_spec(spec, split_type, split_score=None, inp_template=None, args=None, net=None, transformer=None):
    if is_relu_split(split_type):
        spec.chosen_split = choose_relu(split_type, spec.relu_spec, spec=spec, split_score=split_score,
                                        inp_template=inp_template, args=args, transformer=transformer)
        split_relu_specs = spec.relu_spec.split_spec(split_type, spec.chosen_split)
        child_specs = [Spec(spec.input_spec, rs, parent=spec) for rs in split_relu_specs]
    elif is_input_split(split_type):
        spec.chosen_split = choose_split_dim(spec.input_spec, split_type, net=net)
        split_inp_specs = spec.input_spec.split_spec(split_type, spec.chosen_split)
        child_specs = [Spec(ins, spec.relu_spec, parent=spec) for ins in split_inp_specs]
    else:
        raise ValueError("Unknown split!")
    spec.children += child_specs
    return child_specs


def choose_relu(split, relu_spec, spec=None, split_score=None, inp_template=None, args=None, transformer=None):
    """
    Chooses the relu that is split in branch and bound.
    @param: relu_spec contains relu_mask which is a map that maps relus to -1/0/1. 0 here indicates that the relu
        is ambiguous
    """
    relu_mask = relu_spec.relu_mask
    if split == Split.RELU_RAND:
        all_relus = []

        # Collect all un-split relus
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu[0] == 2:
                all_relus.append(relu)

        return random.choice(all_relus)

    # BaBSR based on various estimates of importance
    elif split == Split.RELU_GRAD or split == Split.RELU_ESIP_SCORE or split == Split.RELU_ESIP_SCORE2:
        # Choose the ambiguous relu that has the maximum score in relu_score
        if split_score is None:
            raise ValueError("relu_score should be set while using relu_grad splitting mode")

        max_score, chosen_relu = 0, None

        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu in split_score.keys():
                if split_score[relu] >= max_score:
                    max_score, chosen_relu = split_score[relu], relu

        if chosen_relu is None:
            raise ValueError("Attempt to split should only take place if there are ambiguous relus!")

        config.write_log("Chosen relu for splitting: " + str(chosen_relu) + " " + str(max_score))
        return chosen_relu
    elif split == Split.RELU_KFSB:
        k = 3
        if split_score is None:
            raise ValueError("relu_score should be set while using kFSB splitting mode")
        if spec is None:
            raise ValueError("spec should be set while using kFSB splitting mode")

        candidate_relu_score_list = []
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu in split_score.keys():
                candidate_relu_score_list.append((relu, split_score[relu]))
        candidate_relu_score_list = sorted(candidate_relu_score_list, key=lambda x: x[1], reverse=True)
        candidate_relus = [candidate_relu_score_list[i][0] for i in range(k)]

        candidate_relu_lbs = {}
        for relu in candidate_relus:
            cp_spec = copy.deepcopy(spec)
            split_relu_specs = cp_spec.relu_spec.split_spec(split, relu)
            child_specs = [Spec(cp_spec.input_spec, rs, parent=cp_spec) for rs in split_relu_specs]

            candidate_lb = 0
            for child_spec in child_specs:
                transformer.update_spec(child_spec.input_spec, relu_mask=child_spec.relu_spec.relu_mask)
                lb, _, _ = transformer.compute_lb(complete=True)
                if lb is not None:
                    candidate_lb = min(candidate_lb, lb)

            candidate_relu_lbs[relu] = candidate_lb
        return max(candidate_relu_lbs, key=candidate_relu_lbs.get)
    else:
        # Returning just the first un-split relu
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0:
                return relu
    raise ValueError("No relu chosen!")


def choose_split_dim(input_spec, split, net=None):
    if split == Split.INPUT:
        chosen_dim = torch.argmax(input_spec.input_ub - input_spec.input_lb)
    elif split == Split.INPUT_GRAD:
        zono_transformer = ZonoTransformer(input_spec, complete=True)
        zono_transformer = nnverify.domains.build_transformer(zono_transformer, net, input_spec)

        center = zono_transformer.centers[-1]
        cof = zono_transformer.cofs[-1]
        cof_abs = torch.sum(torch.abs(cof), dim=0)
        lb = center - cof_abs

        if input_spec.out_constr.is_conjunctive:
            adv_index = torch.argmin(lb)
        else:
            adv_index = torch.argmax(lb)

        input_len = len(input_spec.input_lb)
        chosen_noise_idx = torch.argmax(torch.abs(cof[:input_len, adv_index])).item()
        # chosen_noise_idx = torch.argmax(torch.sum(torch.abs(cof[:input_len]), dim=1) * (self.input_ub - self.input_lb))
        chosen_dim = zono_transformer.map_for_noise_indices[chosen_noise_idx]
    elif split == Split.INPUT_SB:
        cp_spec = copy.deepcopy(input_spec)
        lb0 = input_spec.get_zono_lb(net, cp_spec)

        chosen_dim = -1
        best_score = -1e-3

        for dim in range(len(input_spec.input_lb)):
            s1, s2 = cp_spec.split_spec(split, dim)

            lb1 = input_spec.get_zono_lb(net, s1)
            lb2 = input_spec.get_zono_lb(net, s2)

            dim_score = min(lb1 - lb0, lb2 - lb0)

            if dim_score > best_score:
                chosen_dim = dim
                best_score = dim_score
    else:
        raise ValueError("Unknown splitting method!")
    return chosen_dim


def split_chosen_spec(spec, split_type, chosen_split):
    spec.chosen_split = chosen_split
    if is_relu_split(split_type):
        split_relu_specs = spec.relu_spec.split_spec(split_type, chosen_split)
        child_specs = [Spec(spec.input_spec, rs, parent=spec) for rs in split_relu_specs]
    elif is_input_split(split_type):
        split_inp_specs = spec.input_spec.split_spec(split_type, chosen_split)
        child_specs = [Spec(ins, spec.relu_spec, parent=spec) for ins in split_inp_specs]
    else:
        raise ValueError("Unknown split!")
    spec.children += child_specs
    return child_specs
