import torch
import gurobipy as grb

from torch.nn import functional as F

import nnverify.attack
from nnverify import util
from nnverify.domains import grb_utils

device = 'cpu'


class ZonoTransformer:
    def __init__(self, prop, cof_constrain=None, bias_constrain=None, complete=False):
        """
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        """
        self.size = prop.get_input_size()
        self.prop = prop
        self.ilb = prop.input_lb
        self.iub = prop.input_ub
        self.complete = complete

        # Following fields are used for complete verification
        self.complete = complete
        self.map_for_noise_indices = {}

        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)

        self.ilb = self.ilb.to(device)
        self.iub = self.iub.to(device)

        center = (self.ilb + self.iub) / 2
        self.unstable_relus = []
        noise_ind = self.get_noise_indices()

        cof = ((self.iub - self.ilb) / 2 * torch.eye(self.size))[noise_ind]

        self.centers = []
        self.cofs = []
        self.relu_layer_cofs = []
        self.set_zono(center, cof)
        self.lp_model = None # This will be used for ReLU splits
        self.map_relu_layer_idx_to_layer_idx = {}
        self.masked = False # If the mask exist we use LP solver to find the bounds

    def get_noise_indices(self):
        num_eps = 1e-7
        noise_ind = torch.where(self.iub > (self.ilb + num_eps))
        if noise_ind[0].size() == 0:
            # add one dummy index in case there is no perturbation
            noise_ind = torch.tensor([0]).to(device)
        return noise_ind

    def compute_lb(self, adv_label=None, complete=False):
        """
        return the lower bound for the variables of the current layer
        """
        center = self.centers[-1]
        cof = self.cofs[-1]

        if complete:
            if self.masked:
                return self.compute_lb_masked()
            else:
                lb = self.get_zono_lb(adv_label, center, cof)
                return lb, True, None
        else:
            cof_abs = torch.sum(torch.abs(cof), dim=0)
            lb = center - cof_abs
            return lb

    def get_zono_lb(self, adv_label, center, cof):
        cof = cof[:, adv_label]
        cof_abs = torch.sum(torch.abs(cof), dim=0)
        lb = center[adv_label] - cof_abs
        if self.prop.is_conjunctive():
            lb = torch.min(lb)
        else:
            lb = torch.max(lb)
        return lb

    def update_spec(self, prop, relu_mask=None):
        if self.lp_model is None:
            # Create GUROBI model
            self.lp_model = grb_utils.get_gurobi_lp_model()

            # Add grb var for each noise symbol
            num_noise = self.cofs[-1].shape[0]
            self.grb_vars = [
                self.lp_model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'e{noise_idx}', lb=-1, ub=1)
                for noise_idx in range(num_noise)
            ]

        self.masked = False
        for relu, decision in relu_mask.items():
            if decision == 0:
                continue
            self.masked = True
            layer_idx = self.map_relu_layer_idx_to_layer_idx[relu[0]] - 1
            relu_idx = relu[1]
            pre_relu_zono_cof = self.cofs[layer_idx][:, relu_idx].numpy()
            pre_relu_zono_center = self.centers[layer_idx][relu_idx].numpy()
            grb_mvar = grb.MVar(self.grb_vars[:len(pre_relu_zono_cof)])

            linexpr = grb.LinExpr(pre_relu_zono_cof, self.grb_vars[:len(pre_relu_zono_cof)]) + pre_relu_zono_center
            if decision == -1:
                self.lp_model.addConstr(linexpr <= 0)
            elif decision == 1:
                # Using >= since gurobi does not support >
                self.lp_model.addConstr(linexpr >= 0)

    def compute_lb_masked(self):
        # Check objective for each prop
        # TODO: Handle disjunctive clause
        global_lb, adv_ex = None, None
        for i in range(len(self.centers[-1])):
            obj_cof = self.cofs[-1][:, i]
            obj_center = self.centers[-1][i].numpy()
            cof_abs = torch.sum(torch.abs(obj_cof), dim=0)
            unsplit_lb = self.centers[-1][i] - cof_abs

            if unsplit_lb >= 0:
                if global_lb is None:
                    global_lb = unsplit_lb
                continue

            optimize_var = self.lp_model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'out{i}', lb=unsplit_lb)
            linexpr = grb.LinExpr(obj_cof.numpy(), self.grb_vars) + obj_center
            self.lp_model.addConstr(optimize_var == linexpr)

            self.lp_model.setObjective(optimize_var, grb.GRB.MINIMIZE)
            self.lp_model.optimize()

            is_feasible, primal_sol = grb_utils.check_optimization_success(self.lp_model, optimize_var, self.grb_vars)
            adv_ex_cur = None
            if primal_sol is not None:
                adv_ex_cur = (primal_sol[:len(self.cofs[0])] * self.cofs[0]).diag() + self.centers[0]

            if not is_feasible:
                return None, False, None
            # TODO: Fix this. Net should be provided at the intialization of the domain analyzer
            elif nnverify.attack.check_adversarial(adv_ex_cur, self.net, self.prop):
                return None, True, adv_ex_cur

            cur_lb = torch.tensor(optimize_var.X)

            if global_lb is None or cur_lb < global_lb:
                global_lb = cur_lb

        return global_lb, True, adv_ex

    def compute_ub(self, test=True):
        """
        return the upper bound for the variables of the current layer
        """
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        ub = center + cof_abs

        return ub

    def bound(self):
        # This can be little faster by reusing the computation
        center = self.centers[-1]
        cof = self.cofs[-1]

        cof_abs = torch.sum(torch.abs(cof), dim=0)

        lb = center - cof_abs
        ub = center + cof_abs

        return lb, ub

    def get_zono(self):
        return self.centers[-1], self.cofs[-1]

    def set_zono(self, center, cof):
        self.centers.append(center)
        self.cofs.append(cof)

    def get_all_bounds(self):
        lbs, ubs = [], []

        for i in range(len(self.centers)):
            center = self.centers[i]
            cof = self.cofs[i]

            cof_abs = torch.sum(torch.abs(cof), dim=0)

            lb = center - cof_abs
            ub = center + cof_abs

            lbs.append(lb)
            ubs.append(ub)

        return lbs, ubs

    def handle_normalization(self, layer):
        '''
        only change the lower/upper bound of the input variables
        '''
        return
        # mean = layer.mean.view((1))
        # sigma = layer.sigma.view((1))
        #
        # prev_cent, prev_cof = self.get_zono()
        #
        # center = (prev_cent - mean) / sigma
        # cof = prev_cof / sigma
        #
        # self.set_zono(center, cof)
        #
        # return self

    def handle_addition(self, layer, last_layer=False):
        """
        handle addition layer
        """
        bias = layer.bias
        if last_layer:
            bias = bias @ self.prop.output_constr_mat()

        prev_cent, prev_cof = self.get_zono()

        center = prev_cent + bias
        cof = prev_cof

        self.set_zono(center, cof)
        return self

    def handle_linear(self, layer, last_layer=False):
        """
        handle linear layer
        """
        weight = layer.weight.T
        bias = layer.bias
        if last_layer:
            weight = weight @ self.prop.output_constr_mat()
            bias = bias @ self.prop.output_constr_mat() + self.prop.output_constr_const()

        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        prev_cent, prev_cof = self.get_zono()

        center = prev_cent @ weight + bias
        cof = prev_cof @ weight

        self.set_zono(center, cof)
        return self

    def handle_conv2d(self, layer):
        """
        handle conv2d layer
        first transform it to linear matrix
        then use absmul func
        """
        weight = layer.weight
        bias = layer.bias
        num_kernel = weight.shape[0]

        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding

        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        prev_cent, prev_cof = self.get_zono()

        prev_cent = prev_cent.reshape(1, shape[0], input_h, input_w)
        prev_cof = prev_cof.reshape(-1, shape[0], input_h, input_w)

        center = F.conv2d(prev_cent, weight, padding=layer.padding, stride=layer.stride, bias=bias).flatten()

        num_eps = prev_cof.shape[0]
        cof = F.conv2d(prev_cof, weight, padding=layer.padding, stride=layer.stride).reshape(num_eps, -1)

        self.set_zono(center, cof)

        return self

    def handle_relu(self, layer, optimize=True, relu_mask=None):
        """
        handle relu func
        """
        size = self.size

        prev_cent, prev_cof = self.get_zono()
        lb, ub = self.bound()

        relu_layer_idx = len(self.unstable_relus)
        self.map_relu_layer_idx_to_layer_idx[relu_layer_idx] = len(self.cofs)
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0))[0].tolist())

        num_eps = 1e-7
        lmbda = torch.div(ub, ub - lb + num_eps)
        mu = -(lb / 2) * lmbda

        active_relus = (lb > 0)
        passive_relus = (ub <= 0)
        ambiguous_relus = (~active_relus) & (~passive_relus)

        if self.complete:
            # Store the map from (unstable relu index -> index of the added noise)
            prev_error_terms = prev_cof.shape[0]
            unstable_relu_indices = torch.where(ambiguous_relus)[0]

            for i, index in enumerate(unstable_relu_indices):
                index_of_unstable_relu = prev_error_terms + i
                self.map_for_noise_indices[index_of_unstable_relu] = (relu_layer_idx, index.item())

            # Figure out how these should be used
            c1_decision = torch.zeros(size, dtype=torch.bool)
            c2_decision = torch.zeros(size, dtype=torch.bool)

            if relu_mask is not None:
                for relu in relu_mask.keys():
                    if relu[0] == relu_layer_idx:
                        if ambiguous_relus[relu[1]]:
                            if relu_mask[relu] == 1:
                                c1_decision[relu[1]] = 1
                            elif relu_mask[relu] == -1:
                                c2_decision[relu[1]] = 1

            ambiguous_relus = ambiguous_relus & (~c1_decision) & (~c2_decision)
            c1_mu = c1_decision*ub/2
            c2_mu = c2_decision*lb/2

        mult_fact = torch.ones(size, dtype=torch.bool)
        # mult_fact should have 1 at active relus, 0 at passive relus and lambda at ambiguous_relus
        mult_fact = mult_fact * (active_relus + ambiguous_relus * lmbda)

        if self.complete:
            new_noise_cofs = torch.diag(mu * ambiguous_relus + c1_mu + c2_mu)
        else:
            new_noise_cofs = torch.diag(mu * ambiguous_relus)

        non_empty_mask = new_noise_cofs.abs().sum(dim=0).bool()
        new_noise_cofs = new_noise_cofs[non_empty_mask, :]

        cof = torch.cat([mult_fact * prev_cof, new_noise_cofs])

        if self.complete:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus + c1_mu + c2_mu
        else:
            center = prev_cent * active_relus + (lmbda * prev_cent + mu) * ambiguous_relus

        self.set_zono(center, cof)
        self.relu_layer_cofs.append(cof)
        return self

    def verify_robustness(self, y, true_label):
        pass

# def absmul(lb, ub, weight, bias, down = True):
#     ''' 
#     Absdomain multiplication
#     '''
#     pos_wgt = F.relu(weight)
#     neg_wgt = -F.relu(-weight)

#     if down:
#         new_ilb = lb @ pos_wgt + ub @ neg_wgt
#         return new_ilb
#     else:
#         new_iub = ub @ pos_wgt + lb @ neg_wgt
#         return new_iub
