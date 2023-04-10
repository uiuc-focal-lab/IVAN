"""
A generic approach for the BnB based complete verification.
"""
import torch
import nnverify.domains
import nnverify.specs.spec as specs
import time

from nnverify.domains import get_domain_transformer
from nnverify import config
from nnverify.bnb import Split, is_relu_split, branch
from nnverify.bnb.proof_tree import ProofTree
from nnverify.domains.deepz import ZonoTransformer
from nnverify.common import Status
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from multiprocessing import Pool

from nnverify.specs.input_spec import merge_input_specs


class BnB:
    def __init__(self, net, transformer, init_prop, args, template_store, print_result=False):
        self.net = net
        self.transformer = transformer
        self.init_prop = init_prop
        self.split = args.split
        self.template_store = template_store
        self.args = args
        self.depth = 1
        self.init_time = time.time()
        self.global_status = Status.UNKNOWN
        self.print_result = print_result

        # Store proof tree for the BnB
        self.inp_template = self.template_store.get_template(self.init_prop)
        self.root_spec = None
        self.proof_tree = None

        self.cur_specs = self.get_init_specs(init_prop)
        self.tree_size = len(self.cur_specs)
        self.prev_lb = None
        self.cur_lb = None

    def get_init_specs(self, init_prop):
        tree_avail = self.template_store.is_tree_available(init_prop)

        if tree_avail and type(self.args.pt_method) == IVAN:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_pruned_leaves(self.args.pt_method.threshold, self.split)
        elif tree_avail and self.args.pt_method == ProofTransferMethod.REUSE:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_leaves()
        else:
            unstable_relus = self.get_unstable_relus()
            cur_specs = self.create_initial_specs(init_prop, unstable_relus)
        return cur_specs

    def get_unstable_relus(self):
        lb, is_feasible, adv_ex = self.transformer.compute_lb(complete=True)
        status = self.get_status(adv_ex, is_feasible, lb)

        if 'unstable_relus' in dir(self.transformer):
            unstable_relus = self.transformer.unstable_relus
        else:
            unstable_relus = None

        if status != Status.UNKNOWN:
            self.global_status = status
            if status == Status.VERIFIED and self.print_result:
                print(status)
        return unstable_relus

    def run(self):
        """
        It is the public method called from the analyzer. @param split is a string that chooses the mode for relu
        or input splitting.
        """
        if self.global_status != Status.UNKNOWN:
            return

        split_score = self.set_split_score(self.init_prop, self.cur_specs, inp_template=self.inp_template)

        while self.continue_search():
            self.update_depth()

            self.prev_lb = self.cur_lb
            self.reset_cur_lb()

            # Main verification loop
            if self.args.parallel:
                self.verify_specs_parallel()
            else:
                self.verify_specs()

            # Each spec should hold the prev lb and current lb
            self.cur_specs, verified_specs = branch.branch_unsolved(self.cur_specs, self.split, split_score=split_score,
                                                                    inp_template=self.inp_template, args=self.args,
                                                                    net=self.net, transformer=self.transformer)
            # Update the tree size
            self.tree_size += len(self.cur_specs)

        self.check_verified_status()
        self.store_final_tree()

    def verify_specs(self):
        for spec in self.cur_specs:
            self.update_transformer(spec.input_spec, relu_spec=spec.relu_spec)

            # Transformer is updated with new mask
            status, lb = self.verify_node(self.transformer, spec.input_spec)
            self.update_cur_lb(lb)
            spec.update_status(status, lb)

            if status == Status.ADV_EXAMPLE or self.is_timeout():
                self.global_status = status
                self.store_final_tree()
                return

    def verify_specs_parallel(self):
        cur_specs = [self.cur_specs[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range((len(self.cur_specs)+self.args.batch_size-1)//self.args.batch_size)]

        for batch_specs in cur_specs:
            # Create a batch specification
            batch_input_spec = merge_input_specs([spec.input_spec for spec in batch_specs])
            self.transformer = get_domain_transformer(self.args, self.net, batch_input_spec, complete=True)
            lb, is_feasible, adv_ex = self.transformer.compute_lb(complete=True)

            for i in range(len(batch_specs)):
                status = self.get_status(adv_ex, is_feasible, lb[i])
                self.update_cur_lb(lb[i])
                batch_specs[i].update_status(status, lb[i])

                if status == Status.ADV_EXAMPLE or self.is_timeout():
                    self.global_status = status
                    self.store_final_tree()
                    return

    def store_final_tree(self):
        self.proof_tree = ProofTree(self.root_spec)
        self.template_store.add_tree(self.init_prop, self.proof_tree)

    def verify_node(self, transformer, prop):
        """
        It is called from bnb_relu_complete. Attempts to verify (ilb, iub), there are three possible outcomes that
        are indicated by the status: 1) verified 2) adversarial example is found 3) Unknown
        """
        lb, is_feasible, adv_ex = transformer.compute_lb(complete=True)

        status = self.get_status(adv_ex, is_feasible, lb)
        return status, lb

    def get_status(self, adv_ex, is_feasible, lb):
        status = Status.UNKNOWN
        if adv_ex is not None:
            config.write_log("Found a counter example!")
            status = Status.ADV_EXAMPLE
        elif (not is_feasible) or (lb is not None and torch.all(lb >= 0)):
            status = Status.VERIFIED
        return status

    def update_transformer(self, prop, relu_spec=None):
        relu_mask = None
        if relu_spec is not None:
            relu_mask = relu_spec.relu_mask

        if 'update_spec' in dir(self.transformer) and is_relu_split(self.args.split):
            self.transformer.update_spec(prop, relu_mask=relu_mask)
        else:
            self.transformer = get_domain_transformer(self.args, self.net, prop, complete=True)

    def check_verified_status(self):
        # Verified
        if len(self.cur_specs) == 0:
            self.global_status = Status.VERIFIED
            if self.print_result:
                print(Status.VERIFIED)

    def reset_cur_lb(self):
        self.cur_lb = None

    def is_timeout(self):
        cur_time = (time.time() - self.init_time)
        ret = self.args.timeout is not None and cur_time > self.args.timeout
        return ret

    def continue_search(self):
        return self.global_status == Status.UNKNOWN and len(self.cur_specs) > 0 and (not self.is_timeout())

    def update_cur_lb(self, lb):
        # lb can be None if the LP is infeasible
        if lb is not None:
            if self.cur_lb is None:
                self.cur_lb = lb
            else:
                self.cur_lb = min(lb, self.cur_lb)

    def update_depth(self):
        #print('Depth :', self.depth, 'Specs size :', len(self.cur_specs), 'LB:', self.cur_lb)
        self.depth += 1

    def create_initial_specs(self, prop, unstable_relus):
        if is_relu_split(self.split):
            relu_spec = specs.create_relu_spec(unstable_relus)
            self.root_spec = specs.Spec(prop, relu_spec=relu_spec, status=self.global_status)
            cur_specs = specs.SpecList([self.root_spec])
            config.write_log("Unstable relus: " + str(unstable_relus))
        else:
            if self.args.initial_split:
                # Do a smarter initial split similar to ERAN
                # This works only for ACAS-XU
                zono_transformer = ZonoTransformer(prop, complete=True)
                zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)

                center = zono_transformer.centers[-1]
                cof = zono_transformer.cofs[-1]
                cof_abs = torch.sum(torch.abs(cof), dim=0)
                lb = center - cof_abs
                adv_index = torch.argmin(lb)
                input_len = len(prop.input_lb)
                smears = torch.abs(cof[:input_len, adv_index])
                split_multiple = 10 / torch.sum(smears)  # Dividing the initial splits in the proportion of above score
                num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]

                inp_specs = prop.multiple_splits(num_splits)
                cur_specs = specs.SpecList([specs.Spec(prop, status=self.global_status) for prop in inp_specs])
                # TODO: Add a root spec in this case as well
            else:
                self.root_spec = specs.Spec(prop, status=self.global_status)
                cur_specs = specs.SpecList([self.root_spec])

        return cur_specs

    def set_split_score(self, prop, relu_mask_list, inp_template=None):
        """
        Computes relu score for each relu if the split method needs it. Otherwise, returns None
        """
        split_score = None
        if self.split == Split.RELU_GRAD:
            # These scores only work for torch models
            split_score = specs.score_relu_grad(relu_mask_list[0], prop, net=self.net)
        elif self.split == Split.RELU_ESIP_SCORE or self.split == Split.RELU_KFSB:
            # These scores only work with deepz transformer
            zono_transformer = ZonoTransformer(prop, complete=True)
            zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)
            split_score = specs.score_relu_esip(zono_transformer)
        elif self.split == Split.RELU_ESIP_SCORE2:
            # These scores only work with deepz transformer
            zono_transformer = ZonoTransformer(prop, complete=True)
            zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)
            split_score = specs.score_relu_esip2(zono_transformer)

        # Update the scores based on previous scores
        if inp_template is not None and split_score is not None:
            if type(self.args.pt_method) == IVAN or type(self.args.pt_method) == REORDERING:
                # compute mean worst case improvements
                observed_split_scores = self.template_store.get_proof_tree(prop).get_observed_split_score()
                alpha = self.args.pt_method.alpha
                thr = self.args.pt_method.threshold

                for chosen_split in observed_split_scores:
                    if chosen_split in split_score and observed_split_scores[chosen_split] < self.args.pt_method.threshold:
                        split_score[chosen_split] = alpha * split_score[chosen_split] + (1 - alpha) * (
                                observed_split_scores[chosen_split] - thr)

        return split_score
