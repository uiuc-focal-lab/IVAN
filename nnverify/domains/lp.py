import gurobipy as grb
import torch
import copy
import time

import nnverify.attack
import nnverify.config as config
import numpy as np

from torch import nn
from torch.autograd import Variable

from nnverify.domains import grb_utils
from nnverify.domains.box import BoxTransformer
from nnverify.domains.deepz import ZonoTransformer
from nnverify.domains.deeppoly import DeeppolyTransformer
from nnverify.common.network import Network, LayerType
from nnverify.common import Domain
from nnverify.common.dataset import Dataset
from nnverify.domains.lirpa import LirpaTransformer
from nnverify.util import is_lirpa_domain


class LPTransformer:
    def __init__(self, prop, net, complete=True):
        """"
        prop: Property for verification
        """
        self.prop = prop

        #if net.torch_net is not None:
            # Since Lirpa only works with torch net
         #   self.init_domains = [Domain.DEEPZ, Domain.LIRPA_CROWN]
        #else:
        self.init_domains = [Domain.DEEPZ]

        self.set_shape(prop)

        self.lower_bounds, self.upper_bounds = [], []
        self.gurobi_vars, self.unstable_relus = [], []
        self.cur_relu_mask = None

        # Set the Gurobi model
        self.model = grb_utils.get_gurobi_lp_model()
        self.net = net

        if type(net) == Network:
            self.format = 'onnx'
        else:
            self.format = 'torch'

        self.relu_constraints_map = {}
        self.last_layer_constrs = None

    def build(self, net, prop, relu_mask=None):
        self.create_lp_model(net, prop, relu_mask=relu_mask)
        return self

    def set_shape(self, prop):
        self.dataset = prop.dataset
        if self.dataset == Dataset.MNIST:
            self.shape = (1, 28, 28)
        elif self.dataset == Dataset.CIFAR10:
            self.shape = (3, 32, 32)
        elif self.dataset == Dataset.ACAS:
            self.shape = (5)
        else:
            raise ValueError("Unsupported dataset!")

    def create_lp_model(self, net, prop, relu_mask=None):
        """
        This function builds the lp model parsing through each layer of @param layers
        """
        self.net = net
        self.cur_relu_mask = copy.deepcopy(relu_mask)
        self.choose_init_domains(net)

        # Find lower bounds prior to building the model
        # Create the input gurobi layer and update it
        self.create_input_layer()

        # Compute initial bounds using other domains such as deepz, deeppoly
        self.compute_init_bounds(self.init_domains, net, relu_mask)

        # Do the other layers, computing for each of the neuron, its upper
        # bound and lower bound
        self.set_shape(prop)

        # Create and update layers
        self.create_all_layers(prop, net, relu_mask)
        self.update_layers(relu_mask)

    def choose_init_domains(self, net):
        is_conv = False
        for layer in net:
            if layer.type == LayerType.Conv2D:
                is_conv = True
        if not is_conv:
            self.init_domains = [Domain.DEEPZ, Domain.DEEPPOLY]

    def create_all_layers(self, prop, net, relu_mask):
        model_create_start_time = time.time()
        layer_idx = 1
        for layer in net:
            new_layer_gurobi_vars = []
            layer_type = self.get_layer_type(layer)

            if layer_type == LayerType.Linear:
                self.create_linear_layer(prop, layer, layer_idx, net, new_layer_gurobi_vars)
            elif layer_type == LayerType.Conv2D:
                self.create_conv2d_layer(layer, layer_idx, new_layer_gurobi_vars)
            elif layer_type == LayerType.ReLU:
                self.create_relu_layer(layer_idx, new_layer_gurobi_vars, relu_mask)
            elif layer_type == LayerType.MaxPool1D:
                self.create_maxpool_layer(layer, layer_idx, new_layer_gurobi_vars)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                continue

            self.gurobi_vars.append(new_layer_gurobi_vars)
            layer_idx += 1
        model_create_end_time = time.time()
        print('Model creation time: ', model_create_end_time-model_create_start_time)

    def update_input_layer(self):
        for i in range(len(self.prop.input_lb)):
            v = self.gurobi_vars[0][i]
            v.lb = self.prop.input_lb[i].item()
            v.ub = self.prop.input_ub[i].item()

    def create_input_layer(self):
        inp_gurobi_vars = []
        for i in range(len(self.prop.input_lb)):
            v = self.model.addVar(obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'inp_{i}')
            inp_gurobi_vars.append(v)
        self.gurobi_vars.append(inp_gurobi_vars)

    def update_input(self, prop, relu_mask=None):
        """
        This function updates the underlying LP model based on the relu_mask
        """
        # reset all the fields related to the input
        self.reset_input(prop)

        # Update the Gurobi input layer
        self.compute_init_bounds(self.init_domains, self.net, relu_mask)
        self.update_layers(relu_mask, input_update=True)
        self.cur_relu_mask = copy.deepcopy(relu_mask)

    def reset_input(self, prop):
        self.prop = prop
        self.lower_bounds = []
        self.upper_bounds = []
        self.unstable_relus = []
        self.cur_relu_mask = None
        self.shape = None

    def update_spec(self, prop, relu_mask=None):
        """
        This function updates the underlying LP model based on the relu_mask
        """
        # This is not needed in case of relu splits since deeppoly and deepz
        # do not use masks yet
        # However, it is needed in the case of input splits
        self.prop = prop
        self.compute_init_bounds(self.init_domains, self.net, relu_mask)
        self.update_layers(relu_mask)
        self.cur_relu_mask = copy.deepcopy(relu_mask)

    def update_layers(self, relu_mask, input_update=False):
        model_update_start_time = time.time()
        # Do the other layers, computing for each of the neuron, its upper
        # bound and lower bound
        layer_idx = 1
        self.set_shape(self.prop)
        self.update_input_layer()

        layers = self.net
        for layer in layers:
            layer_type = self.get_layer_type(layer)

            if layer_type == LayerType.Linear:
                self.update_linear_layer(layer, layer_idx, last_layer=(layer == layers[-1]), input_update=input_update)
            elif layer_type == LayerType.Conv2D:
                self.update_conv2d(layer_idx)
            elif layer_type == LayerType.ReLU:
                self.update_relu_layer(layer_idx, relu_mask, input_update=input_update)
            elif layer_type == LayerType.Flatten:
                continue
            else:
                continue
            layer_idx += 1
        # Update the Gurobi model once all the constraints are added
        self.model.update()
        model_update_end_time = time.time()
        config.write_log('Time taken to update the model: ' + str(model_update_end_time-model_update_start_time))

    def compute_lb(self, complete=True):
        """
        Compute a lower bound of f(x)_{true_label}-f(x)_{adv_label} for all possible values of the adversarial label.
        There are three outcomes from this function.
        @return is_feasible
        1. If the encoding of the constraint is infeasible then  returns is_feasible=True, and other returns are None.
            Since the constraints include mask conditions where we make certain assumptions over some ReLU nodes this
            can lead to an infeasible model.

        @return adv_example
        2. If the optimal value of optimize_var <= 0 then we check if the final value of input as a potential adversarial
        example @adv_ex_candidate. However, this counter example can be spurious counter example in concrete domain.
        We only return non-None value if the counter-example is true.

        @return global_lb
        3. If the optimal value of each optimized variable is >=0 we have verified the property. In that case adv_ex is
        None

        """
        lb_start_time = time.time()

        # global_lb is lowest lower bound from all label
        global_lb = None

        for i in range(len(self.gurobi_vars[-1])):
            optimize_var = self.gurobi_vars[-1][i]

            # If previously computed lb >= 0 then we do not need to optimize this
            unsplit_lb = self.lower_bounds[-1][i]
            if unsplit_lb >= 0:
                if global_lb is None:
                    global_lb = unsplit_lb
                continue

            adv_ex_candidate, is_feasible = self.optimize_gurobi_model(optimize_var)

            # Immediate return if the LP is not possible
            if not is_feasible:
                return None, False, None
            elif nnverify.attack.check_adversarial(adv_ex_candidate, self.net, self.prop):
                return None, True, adv_ex_candidate

            cur_lb = torch.tensor(optimize_var.X)
            config.write_log("Initial lower bound: " + str(optimize_var.lb))
            config.write_log("LP optimized lower bound: " + str(cur_lb))
            config.write_log(str(i) + ' : ' + str(cur_lb))

            if global_lb is None or cur_lb < global_lb:
                global_lb = cur_lb

        config.write_log('Time taken for lb computation: ' + str(time.time() - lb_start_time))
        return global_lb, True, None

    def optimize_gurobi_model(self, optimize_var):
        # This should not be necessary
        # But getting False negatives without it!
        self.model.reset()
        # Minimize the lower bound
        self.model.setObjective(optimize_var, grb.GRB.MINIMIZE)

        self.model.optimize()
        is_feasible, adv_ex_cur = grb_utils.check_optimization_success(self.model, optimize_var, self.gurobi_vars[0])
        return adv_ex_cur, is_feasible

    def create_maxpool_layer(self, layer, layer_idx, new_layer_gurobi_vars):
        assert layer.padding == 0, "Non supported Maxpool option"
        assert layer.dilation == 1, "Non supported MaxPool option"
        nb_pre = len(self.gurobi_vars[-1])
        window_size = layer.kernel_size
        stride = layer.stride
        pre_start_idx = 0
        pre_window_end = pre_start_idx + window_size
        while pre_window_end <= nb_pre:
            lb = max(self.lower_bounds[-1][pre_start_idx:pre_window_end])
            neuron_idx = pre_start_idx // stride

            v = self.model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS,
                                  name=f'Maxpool{layer_idx}_{neuron_idx}')
            all_pre_var = 0
            for pre_var in self.gurobi_vars[-1][pre_start_idx:pre_window_end]:
                self.model.addConstr(v >= pre_var)
                all_pre_var += pre_var
            all_lb = sum(self.lower_bounds[-1][pre_start_idx:pre_window_end])
            max_pre_lb = lb
            self.model.addConstr(all_pre_var >= v + all_lb - max_pre_lb)

            pre_start_idx += stride
            pre_window_end = pre_start_idx + window_size
            new_layer_gurobi_vars.append(v)

    def create_relu_layer(self, layer_idx, new_layer_gurobi_vars, relu_mask):
        self.unstable_relus.append([])
        cur_relu_layer_idx = len(self.unstable_relus) - 1
        for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
            pre_lb = self.lower_bounds[layer_idx - 1][neuron_idx]
            pre_ub = self.upper_bounds[layer_idx - 1][neuron_idx]

            v = self.model.addVar(obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'ReLU{layer_idx}_{neuron_idx}')

            relu_decision = 0
            if relu_mask is not None and (cur_relu_layer_idx, neuron_idx) in relu_mask.keys():
                relu_decision = relu_mask[(cur_relu_layer_idx, neuron_idx)]

            self.relu_constraints_map[(layer_idx, neuron_idx)] = []

            if (pre_lb >= 0 and pre_ub >= 0) or relu_decision == 1:
                self.add_active_relu_constraints(layer_idx, neuron_idx, pre_var, relu_decision, v)
            elif (pre_lb <= 0 and pre_ub <= 0) or relu_decision == -1:
                self.add_passive_relu_constraints(layer_idx, neuron_idx, pre_var, relu_decision, v)
            else:
                self.add_ambiguous_relu_constraints(layer_idx, neuron_idx, pre_lb, pre_ub, pre_var, v)
            new_layer_gurobi_vars.append(v)

    def add_ambiguous_relu_constraints(self, layer_idx, neuron_idx, pre_lb, pre_ub, pre_var, v):
        self.unstable_relus[-1].append(neuron_idx)
        self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(v >= 0))
        self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(v >= pre_var))
        slope = pre_ub / (pre_ub - pre_lb)
        bias = - pre_lb * slope
        self.relu_constraints_map[(layer_idx, neuron_idx)].append(
            self.model.addConstr(v <= slope * pre_var + bias))

    def add_passive_relu_constraints(self, layer_idx, neuron_idx, pre_var, relu_decision, v):
        self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(v == 0))
        # Add this constraint to the previous var (is this necessary?)
        if relu_decision == -1:
            self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(pre_var <= 0))

    def add_active_relu_constraints(self, layer_idx, neuron_idx, pre_var, relu_decision, v):
        # The ReLU is always passing
        self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(v == pre_var))
        # Add this constraint to the previous node (is this necessary?)
        if relu_decision == 1:
            self.relu_constraints_map[(layer_idx, neuron_idx)].append(self.model.addConstr(pre_var >= 0))

    def create_conv2d_layer(self, layer, layer_idx, new_layer_gurobi_vars):
        weight = layer.weight
        bias = layer.bias
        assert layer.dilation == (1, 1)

        # ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        k_h, k_w = layer.kernel_size
        s_h, s_w = layer.stride
        p_h, p_w = layer.padding
        num_kernel = weight.shape[0]
        input_h, input_w = self.shape[1:]
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        pre_gvars = np.array(self.gurobi_vars[-1])
        pre_gvars = pre_gvars.reshape((weight.shape[1], input_h, input_w))

        # Updated shape
        self.shape = (num_kernel, output_h, output_w)
        for out_chan_idx in range(num_kernel):
            for out_row_idx in range(output_h):
                for out_col_idx in range(output_w):
                    # lin_expr1 = self.get_single_step_conv_expr_loop(bias, input_h, input_w, layer, out_chan_idx,
                    #                                                out_col_idx, out_row_idx, weight)

                    lin_expr = self.get_single_step_conv_expr(bias, input_h, input_w, layer, out_chan_idx, out_col_idx,
                                                          out_row_idx, pre_gvars, weight)

                    v = self.model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                    self.model.addConstr(v == lin_expr)

                    new_layer_gurobi_vars.append(v)

    def get_single_step_conv_expr_loop(self, bias, input_h, input_w, layer, out_chan_idx, out_col_idx, out_row_idx,
                                       weight):
        """
        This is unused and is replaced by faster implementation get_single_step_conv_expr. Keeping it here to check correctness.
        """
        lin_expr = bias[out_chan_idx].item()
        for in_chan_idx in range(weight.shape[1]):
            for ker_row_idx in range(weight.shape[2]):
                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                if (in_row_idx < 0) or (in_row_idx == input_h):
                    # This is padding -> value of 0
                    continue
                for ker_col_idx in range(weight.shape[3]):
                    in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                    if (in_col_idx < 0) or (in_col_idx == input_w):
                        # This is padding -> value of 0
                        continue
                    coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()

                    lin_expr += coeff * self.gurobi_vars[-1][
                        in_chan_idx * (input_h * input_w) + in_row_idx * (input_w) + in_col_idx]
        return lin_expr

    def get_single_step_conv_expr(self, bias, input_h, input_w, layer, out_chan_idx, out_col_idx, out_row_idx,
                                  pre_gvars, weight):
        # Set row start and end for convolution
        in_row_start = -layer.padding[0] + layer.stride[0] * out_row_idx
        in_row_end = in_row_start + weight.shape[2] - 1
        ker_row_start, ker_row_end = 0, weight.shape[2]
        if in_row_start < 0:
            ker_row_start = -in_row_start
        elif in_row_end >= input_h:
            ker_row_end = ker_row_end - in_row_end + input_h - 1
        in_row_start, in_row_end = max(in_row_start, 0), min(in_row_end, input_h - 1)

        # Set column start and end for convolution
        in_col_start = -layer.padding[1] + layer.stride[1] * out_col_idx
        in_col_end = in_col_start + weight.shape[3] - 1
        ker_col_start, ker_col_end = 0, weight.shape[3]
        if in_col_start < 0:
            ker_col_start = -in_col_start
        elif in_col_end >= input_w:
            ker_col_end = ker_col_end - in_col_end + input_w - 1
        in_col_start, in_col_end = max(in_col_start, 0), min(in_col_end, input_w - 1)

        # Extract gurobi variables
        exp_gvars = pre_gvars[:, in_row_start:in_row_end+1, in_col_start:in_col_end+1].reshape(-1)
        coeffs = layer.weight[out_chan_idx, :, ker_row_start:ker_row_end, ker_col_start:ker_col_end].reshape(-1)
        lexp = grb.LinExpr(coeffs, exp_gvars) + bias[out_chan_idx].item()
        return lexp

    def create_linear_layer(self, prop, layer, layer_idx, layers, new_layer_gurobi_vars):
        is_last_layer = layer == layers[-1]
        if is_last_layer:
            weight = prop.output_constr_mat().T @ layer.weight
            bias = prop.output_constr_mat().T @ layer.bias + prop.output_constr_const()
        else:
            weight, bias = layer.weight, layer.bias

        new_layer_gurobi_vars += [self.model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'lay{layer_idx}_{neuron_idx}')
                 for neuron_idx in range(weight.size(0))]

        gvars = grb.MVar(new_layer_gurobi_vars)
        pre_vars = grb.MVar(self.gurobi_vars[-1])

        if is_last_layer:
            self.add_and_store_last_layer_constraints(bias, gvars, pre_vars, weight)
        else:
            self.model.addConstr(weight.detach().numpy() @ pre_vars + bias.detach().numpy() == gvars)

    def add_and_store_last_layer_constraints(self, bias, gvars, pre_vars, weight):
        self.last_layer_constrs = self.model.addConstr(weight.detach().numpy() @ pre_vars + bias.detach().numpy() == gvars)

    def update_relu_layer(self, layer_idx, relu_mask, input_update=False):
        if input_update:
            self.unstable_relus.append([])
        cur_relu_layer_index = layer_idx // 2 - 1

        gvars = grb.MVar(self.gurobi_vars[layer_idx])
        gvars.lb = self.lower_bounds[layer_idx]
        gvars.ub = self.upper_bounds[layer_idx]

        for neuron_idx, pre_var in enumerate(self.gurobi_vars[layer_idx - 1]):
            relu_decision = 0
            if relu_mask is not None and (cur_relu_layer_index, neuron_idx) in relu_mask.keys():
                relu_decision = relu_mask[(cur_relu_layer_index, neuron_idx)]

            relu_decision_prev = 0
            if self.cur_relu_mask is not None and (cur_relu_layer_index, neuron_idx) in self.cur_relu_mask.keys():
                relu_decision_prev = self.cur_relu_mask[(cur_relu_layer_index, neuron_idx)]

            if input_update or relu_decision != relu_decision_prev:
                self.add_relu_constraints(layer_idx, neuron_idx, pre_var, relu_decision,
                                          self.gurobi_vars[layer_idx][neuron_idx])

    def add_relu_constraints(self, layer_idx, neuron_idx, pre_var, relu_decision, v):
        config.write_log("LP constraint updated at relu: " + str((layer_idx, neuron_idx)))
        pre_lb = self.lower_bounds[layer_idx - 1][neuron_idx]
        pre_ub = self.upper_bounds[layer_idx - 1][neuron_idx]

        # remove relu_decision_prev constraints
        for cons in self.relu_constraints_map[(layer_idx, neuron_idx)]:
            self.model.remove(cons)

        self.relu_constraints_map[(layer_idx, neuron_idx)] = []
        if (pre_lb >= 0 and pre_ub >= 0) or relu_decision == 1:
            self.add_active_relu_constraints(layer_idx, neuron_idx, pre_var, relu_decision, v)
        elif (pre_lb <= 0 and pre_ub <= 0) or relu_decision == -1:
            self.add_passive_relu_constraints(layer_idx, neuron_idx, pre_var, relu_decision, v)
        else:
            self.add_ambiguous_relu_constraints(layer_idx, neuron_idx, pre_lb, pre_ub, pre_var, v)

    def update_conv2d(self, layer_idx):
        gvar = grb.MVar(self.gurobi_vars[layer_idx])
        gvar.lb = self.lower_bounds[layer_idx]
        gvar.ub = self.upper_bounds[layer_idx]

    def update_linear_layer(self, layer, layer_idx, last_layer=False, input_update=False):
        if last_layer:
            weight = self.prop.output_constr_mat().T @ layer.weight
            bias = self.prop.output_constr_mat().T @ layer.bias
        else:
            weight, bias = layer.weight, layer.bias

        # Remove the previous constraints
        if input_update and last_layer:
            self.model.remove(self.last_layer_constrs)
            self.model.remove(self.gurobi_vars[layer_idx])
            self.gurobi_vars[layer_idx] = [
                self.model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'lay{layer_idx}_{neuron_idx}')
                for neuron_idx in range(weight.size(0))]
            gvars = grb.MVar(self.gurobi_vars[layer_idx])
            pre_vars = grb.MVar(self.gurobi_vars[layer_idx-1])
            self.add_and_store_last_layer_constraints(bias, gvars, pre_vars, weight)

        gvars = grb.MVar(self.gurobi_vars[layer_idx])
        gvars.lb = self.lower_bounds[layer_idx]
        gvars.ub = self.upper_bounds[layer_idx]

    def compute_init_bounds(self, init_domains, layers, relu_mask):
        domain_start_time = time.time()
        init_bounds = []
        for init_domain in init_domains:
            init_bounds.append(self.get_init_bounds(init_domain, layers, relu_mask))
        # Combine initial bounds from all domains in @param: init_domains
        self.lower_bounds = []
        self.upper_bounds = []
        for i in range(len(init_bounds[0][0])):
            self.lower_bounds.append(init_bounds[0][0][i].flatten())
            self.upper_bounds.append(init_bounds[0][1][i].flatten())

        for i in range(len(self.upper_bounds)):
            for j in range(len(self.upper_bounds[i])):
                if self.lower_bounds[i][j] < 0:
                    self.lower_bounds[i][j] -= 1e-4
                if self.upper_bounds[i][j] > 0:
                    self.upper_bounds[i][j] += 1e-4

        for init_bound_num in range(1, len(init_bounds)):
            for i in range(len(init_bounds[0][0])):
                self.lower_bounds[i] = torch.maximum(self.lower_bounds[i], init_bounds[init_bound_num][0][i].flatten().to('cpu'))
                self.upper_bounds[i] = torch.minimum(self.upper_bounds[i], init_bounds[init_bound_num][1][i].flatten().to('cpu'))

        config.write_log('Time taken by deepz: ' + str(time.time() - domain_start_time))
        config.write_log('Lower bound from deepz:' + str(self.lower_bounds[-1]))

    def get_init_bounds(self, domain, layers, relu_mask):
        if domain == Domain.BOX:
            return self.get_box_bounds(layers, relu_mask)
        elif domain == Domain.DEEPZ:
            return self.get_zono_bounds(layers, relu_mask)
        elif domain == Domain.DEEPPOLY:
            return self.get_dp_bounds(layers, relu_mask)
        elif is_lirpa_domain(domain):
            transformer = LirpaTransformer(self.prop, domain, self.dataset)
            transformer.build(layers, self.prop)
            transformer.compute_lb()
            return transformer.get_all_bounds()
        else:
            raise ValueError("Unknown domain!")

    #TODO: Merge following 3 methods
    def get_box_bounds(self, layers, relu_mask):
        """
        Returns the box bounds obtained from propagating input bounds through the layer.
        """
        transformers = BoxTransformer(self.prop)
        fb = forward_box
        return self.get_domain_bounds(fb, layers, relu_mask, transformers)

    def get_zono_bounds(self, layers, relu_mask):
        """
        Returns the deepz bounds obtained from propagating input bounds through the layer.
        """
        transformers = ZonoTransformer(self.prop)
        return self.get_domain_bounds(layers, relu_mask, transformers)

    def get_dp_bounds(self, layers, relu_mask):
        """
        Returns the deeppoly bounds obtained from propagating input bounds through the layer.
        """
        transformers = DeeppolyTransformer(self.prop)
        return self.get_domain_bounds(layers, relu_mask, transformers)

    def get_domain_bounds(self, layers, relu_mask, transformers):
        for layer in layers:
            layer_type = self.get_layer_type(layer)
            if layer_type == LayerType.ReLU:
                transformers.handle_relu(layer, optimize=False, relu_mask=relu_mask)
            elif layer_type == LayerType.Linear:
                if layer == layers[-1]:
                    transformers.handle_linear(layer, last_layer=True)
                else:
                    transformers.handle_linear(layer)
            elif layer_type == LayerType.Conv2D:
                transformers.handle_conv2d(layer)
        return transformers.get_all_bounds()

    def get_upper_bound(self):
        """
        Compute an upper bound of the minimum of the network on `spec`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        """
        # TODO: Fix this if used
        raise ValueError("get_upper_bound is not implemented for lp domain.")

        nb_samples = 1024
        nb_inp = spec.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        spec_lb = spec.select(1, 0).contiguous()
        spec_ub = spec.select(1, 1).contiguous()
        spec_width = spec_ub - spec_lb

        spec_lb = spec_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        spec_width = spec_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = spec_lb + spec_width * rand_samples

        var_inps = Variable(inps, volatile=True)
        outs = self.net(var_inps)

        upper_bound, idx = torch.min(outs, dim=0)

        upper_bound = upper_bound[0]
        ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_layer_type(self, layer):
        if self.format == "onnx":
            return layer.type

        if self.format == "torch":
            if type(layer) is nn.Linear:
                return LayerType.Linear
            elif type(layer) is nn.Conv2d:
                return LayerType.Conv2D
            elif type(layer) == nn.ReLU:
                return LayerType.ReLU
            elif type(layer) == nn.Flatten():
                return LayerType.Flatten
            else:
                return LayerType.NoOp
                # raise ValueError("Unsupported layer type for torch model!", type(layer))

        raise ValueError("Unsupported model format or model format not set!")

    def compute_ub(self):
        """
        Compute an upper bound of the function on `spec`
        """
        raise ValueError("Not used for now")
        olb = self.lower_bounds[-1][0]
        oub = self.upper_bounds[-1][0]
        for id in range(len(self.gurobi_vars[-1])):
            olb = min(olb, self.lower_bounds[-1][id])
            oub = max(oub, self.upper_bounds[-1][id])

        var = self.model.addVar(lb=olb, ub=oub, obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'op')

        for id in range(len(self.gurobi_vars[-1])):
            pre_var = self.gurobi_vars[-1][id]
            self.model.addConstr(var <= pre_var)

        self.model.setObjective(var, grb.GRB.MAXIMIZE)
        self.model.optimize()

        is_feasible, adv_ex = grb_utils.check_optimization_success(self.model, var, self.gurobi_vars[0])
        ub = None
        if is_feasible:
            ub = torch.tensor(var.X)

            for id in range(len(self.gurobi_vars[-1])):
                pre_var = self.gurobi_vars[-1][id]
                # self.model.addConstr(var <= pre_var)
                print(id, ' : ', pre_var.X)

        print('UB ', ub, '   Feasible:', is_feasible)
        return ub, is_feasible, adv_ex
