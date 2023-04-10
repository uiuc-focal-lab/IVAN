import copy

import gurobipy as grb


class NN:
    layers = []

    def __init__(self, layers):
        self.layers = layers


    def get_output(self, input, from_layer = 0, to_layer = -1):
        self.bounds = []
        if to_layer == -1:
            to_layer = len(self.layers)

        out = input
        for i in range(from_layer, to_layer):
            out = self.layers[i].apply(out)
            self.bounds.append(out)
        return out

    def get_point_output(self, input, from_layer = 0, to_layer = -1):
        if to_layer == -1:
            to_layer = len(self.layers)

        out = input
        for i in range(from_layer, to_layer):
            out = self.layers[i].apply_point(out)
        return out

    def create_gurobi(self, inp, model, relu_mask):
        self.gurobi_vars = []
        out = [inp[0], inp[1]]
        for i in range(0, len(self.layers)):
            if type(self.layers[i]) == ReluTransform:
                out = self.layers[i].apply_gurobi(out, model, self.bounds, i, relu_mask=relu_mask)
            else:
                out = self.layers[i].apply_gurobi(out, model, self.bounds, i)
            self.gurobi_vars.append(out)
        return out


class ReluTransform:
    def __init__(self):
        # DO nothing
        return

    def apply(self, input):
        out = []
        for i in range(len(input)):
            out.append((max(input[i][0], 0), max(input[i][1], 0)))
        return out

    def apply_gurobi(self, input, model, bounds, layer, relu_mask=None):
        out = []
        for i in range(len(input)):
            v = model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, lb=bounds[layer][i][0], ub=bounds[layer][i][1])

            pre_lb, pre_ub = bounds[layer-1][i][0], bounds[layer-1][i][1]

            relu_decision = 0
            if relu_mask is not None and (layer, i) in relu_mask.keys():
                relu_decision = relu_mask[(layer, i)]

            if (pre_lb >= 0 and pre_ub >= 0) or relu_decision == 1:
                model.addConstr(v == input[i])
                model.addConstr(input[i] >= 0)
            elif (pre_lb <= 0 and pre_ub <= 0) or relu_decision == -1:
                model.addConstr(v == 0)
                model.addConstr(input[i] <= 0)
            elif pre_lb <= 0 and pre_ub >= 0:
                slope = pre_ub / (pre_ub - pre_lb)
                bias = - pre_lb * slope
                model.addConstr(v <= slope * input[i] + bias)
                model.addConstr(v >= input[i])
                # model.addConstr(v >= 0)
            out.append(v)
        return out

    def apply_point(self, input):
        # print(input)
        out = []
        for i in range(len(input)):
            out.append(max(input[i][0], 0))
        return out


class AffineTransform:
    W = []

    def __init__(self, mat, bias=[0, 0]):
        self.W = mat
        self.bias = bias

    def apply(self, input):
        out = []
        for i in range(len(self.W)):
            val1 = 0
            val2 = 0
            for j in range(len(self.W[i])):
                add1 = self.W[i][j]*input[j][0]
                add2 = self.W[i][j]*input[j][1]
                val1 += min(add1, add2)
                val2 += max(add1, add2)
            out.append((min(val1, val2) + self.bias[i], max(val1, val2) + self.bias[i]))

        return out

    def apply_gurobi(self, input, model, bounds, layer):
        out = []
        for i in range(len(self.W)):
            v = model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, lb=bounds[layer][i][0], ub=bounds[layer][i][1])
            rhs = 0
            for j in range(len(self.W[i])):
                rhs += self.W[i][j]*input[j]
            out.append(v)
            model.addConstr(v == rhs)
        return out

    def apply_point(self, input):
        out = []
        for i in range(len(self.W)):
            val1 = 0
            for j in range(len(self.W[i])):
                add1 = self.W[i][j]*input[j]
                val1 += add1
            out.append(val1 + self.bias[i])

        return out


def split(relu_mask, relu_id):
    rm1 = copy.deepcopy(relu_mask)
    rm2 = copy.deepcopy(relu_mask)

    rm1[relu_id] = -1
    rm2[relu_id] = 1
    return rm1, rm2


def chooseRelu(rm, order=None):
    if order is None:
        # order = [(1, 0), (3, 1), (3, 0)]
        order = [(3, 1), (3, 0), (1, 0)]
        # order = [(3, 0), (3, 1), (1, 0)]

    for r in order:
        if r not in rm.keys():
            return r

def main():
    # Standard BaB
    Np = NN([AffineTransform([[2.1, -0.9], [4.2, -3.1]]), ReluTransform(), AffineTransform([[4.1, -5.9], [2.1, -2.9]]),
             ReluTransform(), AffineTransform([[-2, -2]])])

    _ = bab(Np)

    # Reuse
    N = NN([AffineTransform([[2, -1], [4, -3]]), ReluTransform(), AffineTransform([[4, -6], [2, -3]]),
            ReluTransform(), AffineTransform([[-2, -2]])])

    leaves = bab(N)

    Np = NN([AffineTransform([[2.1, -0.9], [4.2, -3.1]]), ReluTransform(), AffineTransform([[4.1, -5.9], [2.1, -2.9]]),
             ReluTransform(), AffineTransform([[-2, -2]])])

    _ = bab(Np, active=leaves)

    # Reorder
    N = NN([AffineTransform([[2, -1], [4, -3]]), ReluTransform(), AffineTransform([[4, -6], [2, -3]]),
            ReluTransform(), AffineTransform([[-2, -2]])])

    leaves = bab(N)

    Np = NN([AffineTransform([[2.1, -0.9], [4.2, -3.1]]), ReluTransform(), AffineTransform([[4.1, -5.9], [2.1, -2.9]]),
             ReluTransform(), AffineTransform([[-2, -2]])])

    order = [(3, 0), (3, 1), (1, 0)]
    _ = bab(Np, order=order)


def bab(N, active=None, order=None):
    if active is None:
        active = [{}]
    leaves = []
    psi = -14
    # Prove (cy - psi >= 0)
    node_cnt = 0
    while len(active) != 0:
        node_cnt += len(active)
        new_active = []
        for rm in active:
            cy = gurobi_bound(N, rm)
            if cy < psi:
                relu = chooseRelu(rm, order=order)
                rm1, rm2 = split(rm, relu)
                new_active.append(rm1)
                new_active.append(rm2)
            else:
                leaves.append(rm)
        active = new_active
        print(active)
    print("Total analyzer calls: ", node_cnt)
    return leaves


def gurobi_bound(N, relu_mask):
    # Awesome now we have 4 ambi relus with interval analysis

    input = [(0, 1), (0, 1)]
    output = N.get_output(input)

    # Use gurobi to get better bounds for the output now
    # Set the Gurobi model
    model = grb.Model()
    model.setParam('OutputFlag', False)
    x1 = model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'x1', lb=input[0][0], ub=input[0][1])
    x2 = model.addVar(obj=0, vtype=grb.GRB.CONTINUOUS, name=f'x2', lb=input[1][0], ub=input[1][1])
    grb_input = [x1, x2]
    model.update()
    grb_out = N.create_gurobi(grb_input, model, relu_mask)
    model.setObjective(grb_out[0], grb.GRB.MINIMIZE)
    model.optimize()
    print(grb_out[0].X+14)
    return grb_out[0].X


if __name__ == "__main__":
    main()