import torch
import copy

from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


class DeeppolyTransformer:
    def __init__(self, prop, cof_constrain=None, bias_constrain=None, complete=False):
        """
        lcof: the coefficients for the lower bound transformation (w.r.t. the input variables)
        ucof: the coefficients for the upper bound transformation (w.r.t. the input variables)
        lcst: the constants for the lower bound transformation (w.r.t. the input variables)
        ucst: the constants for the upper bound transformation (w.r.t. the input variables)
        ilb: the lower bound for the input variables
        iub: the upper bound for the input variables
        During the verification, we will iteratively update the lcf, ucf, lcst, ucst
        while fixing the lb, and ub after normalization.
        """
        #self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.prop = prop
        self.size = prop.get_input_size()
        cof = torch.eye(self.size, device=self.device)
        cst = torch.zeros(self.size, device=self.device)

        if self.size == 784:
            self.shape = (1, 28, 28)
        elif self.size == 3072:
            self.shape = (3, 32, 32)

        self.lcof = [cof, ]
        self.ucof = [cof, ]
        self.lcst = [cst, ]
        self.ucst = [cst, ]
        self.cur_lcof = None
        self.cur_lcst = None
        self.unstable_relus = []

        self.prop = prop
        self.cof_constrain = cof_constrain
        self.bias_constrain = bias_constrain
        self.optimize_lambda = False
        self.complete = complete

    def compute_lb(self, complete=False, adv_label=None):
        """
        return the lower bound for the variables of the current layer
        """
        lcof = self.lcof[-1]
        lcst = self.lcst[-1]

        lcof = lcof.to(self.device)
        lcst = lcst.to(self.device)

        for i in range(2, len(self.lcof) + 1):
            lcof, lcst = absmul(self.lcof[-i], self.ucof[-i], self.lcst[-i], self.ucst[-i], lcof, lcst, down=True)

        self.cur_lcof = lcof.detach()
        self.cur_lcst = lcst.detach()

        pos_cof = F.relu(lcof)
        neg_cof = -F.relu(-lcof)

        pos_lb = self.prop.input_lb.to(self.device) @ pos_cof
        neg_lb = self.prop.input_ub.to(self.device) @ neg_cof

        lb = pos_lb + neg_lb + lcst

        if self.cof_constrain is not None:
            k = torch.zeros(1, lcof.shape[1]).requires_grad_()
            optimizer = optim.Adam([k], lr=1)
            scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

            for i in range(150):
                optimizer.zero_grad()
                k.data.clamp_(0)
                new_cof = lcof.detach() + k * self.cof_constrain
                new_pos_cof = F.relu(new_cof)
                new_neg_cof = -F.relu(-new_cof)

                new_pos_lb = self.prop.input_lb @ new_pos_cof
                new_neg_lb = self.prop.input_ub @ new_neg_cof
                new_lb = new_pos_lb + new_neg_lb + lcst.detach() + k * self.bias_constrain
                loss = -new_lb.mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
            new_lb = torch.maximum(lb, new_lb.detach())

            return new_lb.squeeze()

        if complete:
            lb = lb.squeeze()
            if self.prop.is_conjunctive():
                lb = torch.min(lb)
            else:
                lb = torch.max(lb)
            return lb, True, None
        return lb.squeeze()

    def get_constrain(self, label_list):

        cof_constrain = self.cur_lcof[:, label_list]
        bias_constrain = self.cur_lcst[:, label_list]
        return cof_constrain, bias_constrain

    def compute_ub(self, test=True):
        """
        return the upper bound for the variables of the current layer
        """
        ucof = self.ucof[-1]
        ucst = self.ucst[-1]

        for i in range(2, len(self.ucof) + 1):
            ucof, ucst = absmul(self.lcof[-i], self.ucof[-i], self.lcst[-i], self.ucst[-i], ucof, ucst, down=False)

        pos_cof = F.relu(ucof)
        neg_cof = -F.relu(-ucof)

        pos_ub = self.prop.input_ub.to(self.device) @ pos_cof
        neg_ub = self.prop.input_lb.to(self.device) @ neg_cof

        ub = pos_ub + neg_ub + ucst

        if self.cof_constrain is not None and test:

            k = torch.zeros(1, ucof.shape[1]).requires_grad_()
            optimizer = optim.Adam([k], lr=1)
            scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)

            for i in range(150):
                optimizer.zero_grad()
                k.data.clamp_(0)
                new_cof = ucof.detach() - k * self.cof_constrain
                new_pos_cof = F.relu(new_cof)
                new_neg_cof = -F.relu(-new_cof)

                new_pos_ub = self.prop.input_ub @ new_pos_cof
                new_neg_ub = self.prop.input_lb @ new_neg_cof
                new_ub = new_pos_ub + new_neg_ub + ucst.detach() - k * self.bias_constrain

                loss = new_ub.mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            new_ub = torch.minimum(ub, new_ub.detach())

            return new_ub.squeeze()

        return ub.squeeze()

    def bound(self):
        return self.compute_lb(), self.compute_ub()

    def get_cof_cst(self):
        return self.lcof, self.ucof, self.lcst, self.ucst

    def set_cof_cst(self, lcof, ucof, lcst, ucst):
        self.lcof.append(lcof.to(self.device))
        self.ucof.append(ucof.to(self.device))
        self.lcst.append(lcst.to(self.device))
        self.ucst.append(ucst.to(self.device))

    def get_all_bounds(self):
        lbs = []
        ubs = []

        ucof = copy.copy(self.ucof)
        ucst = copy.copy(self.ucst)
        lcof = copy.copy(self.lcof)
        lcst = copy.copy(self.lcst)

        for i in range(len(ucof)):
            self.ucof = ucof[:i + 1]
            self.ucst = ucst[:i + 1]
            self.lcof = lcof[:i + 1]
            self.lcst = lcst[:i + 1]

            lbs.append(self.compute_lb())
            ubs.append(self.compute_ub())

        return lbs, ubs

    def handle_normalization(self, layer):
        """
        only change the lower/upper bound of the input variables
        """
        mean = layer.mean.view((1, 1))
        sigma = layer.sigma.view((1, 1))
        self.prop.input_lb = (self.prop.input_lb - mean) / sigma
        self.prop.input_ub = (self.prop.input_ub - mean) / sigma

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

        self.set_cof_cst(weight, weight, bias, bias)
        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        #     if self.old_lcof != None:
        #         old_lcof, old_ucof, old_lcst, old_ucst = absmul(*self.get_old_cof_cst(), weight, bias)
        #         self.set_old_cof_cst(old_lcof, old_ucof, old_lcst, old_ucst)

        return self

    def handle_addition(self, layer, true_label=None):
        """
        handle linear layer
        """
        # weight = layer.weight.T
        bias = layer.bias
        if true_label != None:
            # weight = weight[:,true_label].view(-1,1) - weight
            bias = bias[true_label] - bias

        weight = torch.eye(bias.shape[0])
        self.set_cof_cst(weight, weight, bias, bias)
        self.shape = (1, weight.shape[1])
        self.size = weight.shape[1]

        #     if self.old_lcof != None:
        #         old_lcof, old_ucof, old_lcst, old_ucst = absmul(*self.get_old_cof_cst(), weight, bias)
        #         self.set_old_cof_cst(old_lcof, old_ucof, old_lcst, old_ucst)

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

        size = self.size
        shape = self.shape

        input_h, input_w = shape[1:]

        ### ref. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d ###
        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        self.shape = (num_kernel, output_h, output_w)
        self.size = num_kernel * output_h * output_w

        ### pad cof, cst first ###

        cof = torch.eye(size).reshape(size, *shape)

        pad2d = (p_w, p_w, p_h, p_h)
        cof = F.pad(cof, pad2d)

        ### change to the linear matrix form ###
        linear_cof = []
        for i in range(output_h):
            w_cof = []
            for j in range(output_w):
                h_start = i * s_h
                h_end = h_start + k_h
                w_start = j * s_w
                w_end = w_start + k_w

                w_cof.append(cof[:, :, h_start: h_end, w_start: w_end])

            linear_cof.append(torch.stack(w_cof, dim=1))

        linear_cof = torch.stack(linear_cof, dim=1).reshape(size, output_h, output_w, -1)

        new_weight = weight.reshape(num_kernel, -1).T
        new_cof = linear_cof @ new_weight
        new_cof = new_cof.permute(0, 3, 1, 2).reshape(size, -1)
        new_cst = bias.view(-1, 1, 1).expand(num_kernel, output_h, output_w).reshape(1, -1)

        self.set_cof_cst(new_cof, new_cof, new_cst, new_cst)

        return self

    def handle_relu(self, layer, optimize=True, relu_mask=None):
        """
        handle relu func
        abs(lb) > abs(ub) => k = 0, otherwise k = 1
        """
        size = self.size

        lb, ub = self.bound()
        self.unstable_relus.append(torch.where(torch.logical_and(ub >= 0, lb <= 0)))

        new_lcof = torch.zeros(size).to(self.device)
        new_ucof = torch.zeros(size).to(self.device)
        new_lcst = torch.zeros(size).to(self.device)
        new_ucst = torch.zeros(size).to(self.device)

        ### case 1 ub <= 0 ###
        ### will be cleared ###
        clear = ub <= 0

        ### case 2 lb >= 0 ###
        ### will be saved ###
        noclear = ~ clear
        save = noclear & (lb >= 0)

        ### case 3 lb<0 & ub>0 ###
        ### need to be approximated ###
        approximate = (noclear & (lb < 0)).to(self.device)

        if self.optimize_lambda:
            if hasattr(layer, 'opt_lambda'):
                pass
            else:
                mask = ~(approximate & (abs(lb) > abs(ub)))
                layer.opt_lambda = torch.ones(size) * mask.int()
                layer.opt_lambda.requires_grad_()
            opt_lambda = layer.opt_lambda
        else:
            mask = ~(approximate & (abs(lb) > abs(ub)))
            opt_lambda = torch.ones(size, device=self.device) * mask.int()

        ####! optimize  ####
        new_lcof[save] = 1

        act_lambda = opt_lambda * approximate.int()
        new_lcof += act_lambda

        ####! optimize  ####

        ### handle lower bound first ###
        ### saved part ###
        #     new_lcof[save.expand_as(lcof)] = lcof[save.expand_as(lcof)]
        #     new_lcst[save] = lcst[save]

        #     ### k = 0 part ###
        #     kzero = approximate & (abs(lb) > abs(ub))
        #     new_lcof[kzero.expand_as(lcof)] = lcof[kzero.expand_as(lcof)]
        #     new_lcst[kzero] = lcst[kzero]

        #     ### k = 1 part ###
        #     kone = approximate & (abs(lb) <= abs(ub))
        #     new_lcof[kone.expand_as(lcof)] = lcof[kone.expand_as(lcof)]
        #     new_lcst[kone] = lcst[kone]

        ### upper bound ###
        ### saved part ###
        new_ucof[save] = 1

        ### change k ###
        denominator = ub - lb
        denominator[denominator == 0] = 1.
        tmp_ucof = ub / denominator
        tmp_ucst = - lb * ub / denominator
        new_ucof += tmp_ucof * approximate.int()
        new_ucst += tmp_ucst * approximate.int()

        self.set_cof_cst(torch.diag(new_lcof), torch.diag(new_ucof), new_lcst.reshape(1, -1),
                                 new_ucst.reshape(1, -1))
        return self

    # def handle_flatten(self, layer):
    #     size = self.size
    #     shape = self.shape

    #     if self.old_lcof != None:
    #         old_lcof, old_ucof, old_lcst, old_ucst = self.get_old_cof_cst()
    #         old_lcof = old_lcof.reshape(self.size,-1)
    #         old_ucof = old_ucof.reshape(self.size,-1)
    #         old_lcst = old_lcst.reshape(1,-1)
    #         old_ucst = old_ucst.reshape(1,-1)
    #         self.set_old_cof_cst(old_lcof, old_ucof, old_lcst, old_ucst)

    #     return self

    def verify_robustness(self, y, true_label):
        pass


def absmul(lcof, ucof, lcst, ucst, weight, bias, down=True):
    """
    Absdomain multiplication
    """
    pos_wgt = F.relu(weight)
    neg_wgt = -F.relu(-weight)

    if down:
        new_lcof = lcof @ pos_wgt + ucof @ neg_wgt
        new_lcst = lcst @ pos_wgt + ucst @ neg_wgt + bias
        return new_lcof, new_lcst
    else:
        new_ucof = ucof @ pos_wgt + lcof @ neg_wgt
        new_ucst = ucst @ pos_wgt + lcst @ neg_wgt + bias
        return new_ucof, new_ucst
