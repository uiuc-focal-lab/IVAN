import gurobipy as grb
import torch


def get_gurobi_lp_model():
    model = grb.Model()
    model.setParam('OutputFlag', False)
    model.setParam('Threads', 1)
    return model


def check_optimization_success(model, opt_var, input_vars):
    """
    @param var: Variable optimized
    @return: primal solution of the optimization
    """
    if model.status == 2:
        # Optimization successful, nothing to complain about
        adv_ex = None
        if opt_var.X < 0:
            adv_ex = torch.tensor([gvar.X for gvar in input_vars])
        return True, adv_ex
    elif model.status == 3:
        model.computeIIS()
        model.write("model.ilp")
        return False, None
    else:
        print('\n')
        print(f'model.status: {model.status}\n')
        raise NotImplementedError