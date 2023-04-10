import torch

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from nnverify import util
from nnverify.common import Domain
from nnverify.common.dataset import Dataset
from auto_LiRPA.operators import BoundLinear, BoundConv, BoundRelu


class LirpaTransformer:
    def __init__(self, prop, domain, dataset, complete=True):
        """"
        prop: Property for verification
        """
        self.domain = Domain
        self.dataset = dataset

        if domain == Domain.LIRPA_IBP:
            self.method = 'IBP'
        elif domain == Domain.LIRPA_CROWN_IBP:
            self.method = 'backward'
        elif domain == Domain.LIRPA_CROWN:
            self.method = 'CROWN'
        elif domain == Domain.LIRPA_CROWN_OPT:
            self.method = 'CROWN-Optimized'

        self.model = None
        self.ilb = None
        self.iub = None
        self.input = None
        self.out_spec = None
        self.prop = None

    def build(self, net, prop, relu_mask=None):
        self.ilb = util.reshape_input(prop.input_lb, self.dataset)
        self.iub = util.reshape_input(prop.input_ub, self.dataset)
        self.input = (self.ilb + self.iub) / 2
        batch_size = self.input.shape[0]
        if net.torch_net is None:
            raise ValueError("LiRPA only supports torch model!")
        self.model = BoundedModule(net.torch_net, torch.empty_like(self.input), device=prop.input_lb.device)
        self.out_spec = prop.out_constr.constr_mat[0].T.unsqueeze(0).repeat(batch_size, 1, 1)
        self.prop = prop

    def compute_lb(self, complete=False):
        ptb = PerturbationLpNorm(x_L=self.ilb, x_U=self.iub)
        lirpa_input_spec = BoundedTensor(self.input, ptb)
        olb, _ = self.model.compute_bounds(x=(lirpa_input_spec,), method=self.method, C=self.out_spec)
        olb = olb + self.prop.out_constr.constr_mat[1]

        if self.prop.is_conjunctive():
            lb = torch.min(olb, dim=1).values
        else:
            lb = torch.max(olb, dim=1).values

        if complete:
            return lb, True, None
        else:
            return lb

    def get_all_bounds(self):
        lbs, ubs = [], []
        lbs.append(self.ilb)
        ubs.append(self.iub)
        for node_name, node in self.model._modules.items():
            if type(node) in [BoundLinear, BoundConv] and node_name != 'last_final_node':
                lbs.append(node.lower)
                lbs.append(torch.clamp(node.lower, min=0))
                ubs.append(node.upper)
                ubs.append(torch.clamp(node.upper, min=0))
        return lbs, ubs
