import onnx
import torch
import nnverify.util as util
from autoattack import AutoAttack as AA
import torch.nn as nn

from nnverify.specs.input_spec import InputSpecType


class AutoAttack:
    def __init__(self, norm='Linf'):
        self.norm = norm

    def search_adversarial(self, net, prop, args):
        # If it is an ONNX network, convert it to torch
        if net.torch_net is None:
            net_onnx = onnx.load(net.net_name)
            net.torch_net, _ = util.onnx2torch(net_onnx)

        if args.spec_type == InputSpecType.LINF:
            return self.search_adversarial_linf(net, prop.input, prop.get_label(), args)
        else:
            # This should not require a label
            return self.search_adversarial_general(net, prop, args)

    def search_adversarial_linf(self, net, inp, label, args):
        torch_net = net.torch_net
        # norm_net = util.get_normalized_model(net, args.dataset)
        adversary = AA(torch_net, norm=self.norm, eps=args.eps, version='plus', device=inp.device)
        inp = util.reshape_input(inp, args.dataset)
        attack_images = adversary.run_standard_evaluation(inp, label.unsqueeze(0), bs=1)
        return attack_images

    # Works for general lb and ub
    def search_adversarial_general(self, net, prop, args):
        c = util.reshape_input(prop.input_lb, args.dataset)
        m = util.reshape_input((prop.input_ub-prop.input_lb), args.dataset)
        inp = torch.ones(m.shape)*0.5

        torch_net = net.torch_net

        norm_net = Normalization(torch_net, m, c, prop.out_constr)
        adversary = AA(norm_net, norm=self.norm, eps=0.5, version='standard', device=m.device)
        inp = util.reshape_input(torch.tensor(inp), args.dataset)
        counterexample = adversary.run_standard_evaluation(inp, torch.tensor([0]), bs=1)

        counterexample = counterexample*m + c
        # counterexample = counterexample*std + mean
        return counterexample


# class PGD:
#     def __init__(self, iterations=10, restarts=10):
#         self.iterations = iterations
#         self.restarts = restarts
#
#     def search_adversarial(self, net, inp, label, args):
#         torch_net = net.torch_net
#         inp = util.reshape_input(inp, args.dataset).clone().detach()
#
#         delta = ...
#
#         for i in range(self.iterations):
#             out = torch_net(inp+delta)
#             loss = ...
#
#             loss.backwards()
#             clamp
#
#         return attack_images


class Normalization(nn.Module):
    def __init__(self, model, m, c, out_constr):
        super(Normalization, self).__init__()
        self.m = nn.Parameter(m, requires_grad=False)
        self.c = nn.Parameter(c, requires_grad=False)
        self.out_constr_weight = nn.Parameter(out_constr.constr_mat[0], requires_grad=False)
        self.out_constr_bias = nn.Parameter(out_constr.constr_mat[1], requires_grad=False)
        self.model = model

    def forward(self, y):
        out = y*self.m + self.c
        out = self.model(out)
        out = out @ self.out_constr_weight +self.out_constr_bias
        return out
