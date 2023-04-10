import torch

import nnverify.attack
import nnverify.domains
import nnverify.util as util
import nnverify.specs.spec as specs
import time
import nnverify.bnb.bnb as bnb

from nnverify import config
from nnverify.common import Status
from nnverify.specs.out_spec import OutSpecType
from nnverify.common.result import Result, Results
from nnverify.proof_transfer.template import TemplateStore
from nnverify.domains import build_transformer, get_domain_transformer


class Analyzer:
    def __init__(self, args, net=None, template_store=None):
        """
        @param args: configuration arguments for the analyzer such as the network, domain, dataset, attack, count, dataset,
            epsilon and split
        """
        self.args = args
        self.net = net
        self.template_store = template_store
        self.timeout = args.timeout
        self.device = config.DEVICE
        self.transformer = None
        self.init_time = None

        if self.net is None:
            self.net = util.get_net(self.args.net, self.args.dataset)
        if self.template_store is None:
            self.template_store = TemplateStore()

    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1

        # Check if classified correctly
        if nnverify.attack.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = bnb.BnB(self.net, self.transformer, prop, self.args, self.template_store)
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
        return status, tree_size

    def update_transformer(self, prop):
        if self.transformer is not None and 'update_input' in dir(self.transformer) \
                and prop.out_constr.constr_type == OutSpecType.LOCAL_ROBUST:
            self.transformer.update_input(prop)
        else:
            self.transformer = get_domain_transformer(self.args, self.net, prop, complete=True)

    def analyze_no_split_adv_ex(self, prop):
        # TODO: handle feasibility
        lb, _, adv_ex = self.transformer.compute_lb()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        elif adv_ex is not None:
            status = Status.ADV_EXAMPLE
        print(lb)
        return status

    def analyze_no_split(self):
        lb = self.transformer.compute_lb()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        print('LB: ', lb)
        return status

    def run_analyzer(self):
        """
        Prints the output of verification - count of verified, unverified and the cases for which the adversarial example
            was found
        """
        print('Running on the network: ', self.args.net)
        print('Number of verification instances: ', self.args.count)
        print('Timeout of verification: ', self.args.timeout)
        print('Using %s abstract domain' % self.args.domain)

        props, inputs = specs.get_specs(self.args.dataset, spec_type=self.args.spec_type, count=self.args.count,
                                        eps=self.args.eps)

        results = self.analyze_domain(props)

        results.compute_stats()
        print('Results: ', results.output_count)
        print('Average time:', results.avg_time)
        return results

    # There are multiple clauses in the inout specification
    # Property should hold on all the input clauses
    @staticmethod
    def extract_status(cl_status):
        for status in cl_status:
            if status != Status.VERIFIED:
                return status
        return Status.VERIFIED

    def analyze_domain(self, props):
        results = Results(self.args)
        for i in range(len(props)):
            print("************************** Proof %d *****************************" % (i+1))
            num_clauses = props[i].get_input_clause_count()
            clause_ver_status = []
            ver_start_time = time.time()

            for j in range(num_clauses):
                cl_status, tree_size = self.analyze(props[i].get_input_clause(j))
                clause_ver_status.append(cl_status)

            status = self.extract_status(clause_ver_status)
            print(status)
            ver_time = time.time() - ver_start_time
            results.add_result(Result(ver_time, status, tree_size=tree_size))

        return results

