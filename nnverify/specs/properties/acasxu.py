"""
Modified specs taken from https://github.com/stanleybak/vnncomp2021/blob/main/benchmarks/acasxu/generate.py
The verification problem is reduced to proving:
    (mat * Y + const) >= 0
"""
import torch

from math import pi

from nnverify import util
from nnverify.specs import spec
from nnverify.common.dataset import Dataset
from nnverify.specs.property import Property, InputSpecType, OutSpecType
from nnverify.specs.out_spec import Constraint


def get_acas_spec(id):
    """
    get the list of specification mat and const

    @mat and @const
    The verification problem is reduced to proving: (mat * Y + const) >= 0
    """

    # labels = ['Clear of Conflict (COC)', 'Weak Left', 'Weak Right', 'Strong Left', 'Strong Right']

    if id == 1:
        _ = 'Safe if COC < 1500. Output scaling is 373.94992 with a bias of 7.518884: (1500 - 7.518884) ' + \
            '/ 373.94992 = 3.991125'

        init_lb, init_ub = get_init_bounds([55947.691, -pi, -pi, 1145, 0], [60760, pi, pi, 1200, 60])

        output_scaling_mean = 7.5188840201005975
        output_scaling_range = 373.94992

        # (1500 - 7.518884) / 373.94992 = 3.991125
        threshold = (1500 - output_scaling_mean) / output_scaling_range
        mat, const \
            = get_out_constr([[-1.0, 0, 0, 0, 0]], [threshold])
        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              )
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 2:
        _ = 'Safe if COC is not maximal'

        init_lb, init_ub = get_init_bounds([55947.691, -pi, -pi, 1145, 0], [60760, pi, pi, 1200, 60])

        mat = [[-1, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0],
               [-1, 0, 0, 1, 0],
               [-1, 0, 0, 0, 1]]
        const \
            = [0, 0, 0, 0]

        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              , is_conjunctive=False)
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 3:
        _ = 'Safe if COC is not minimal'

        init_lb, init_ub = get_init_bounds([1500, -0.06, 3.1, 980, 960], [1800, 0.06, pi, 1200, 1200])

        mat = [[1, -1, 0, 0, 0],
               [1, 0, -1, 0, 0],
               [1, 0, 0, -1, 0],
               [1, 0, 0, 0, -1]]
        const \
            = [0, 0, 0, 0]

        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              , is_conjunctive=False)
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 4:
        _ = 'Safe if COC is not minimal'

        init_lb, init_ub = get_init_bounds([1500, -0.06, 0, 1000, 700], [1800, 0.06, 0, 1200, 800])

        mat = [[1, -1, 0, 0, 0],
               [1, 0, -1, 0, 0],
               [1, 0, 0, -1, 0],
               [1, 0, 0, 0, -1]]
        const \
            = [0, 0, 0, 0]

        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              , is_conjunctive=False)
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 5:
        _ = 'Safe if strong right is minimal'

        init_lb, init_ub = get_init_bounds([250, 0.2, -3.141592, 100, 0], [400, 0.4, -3.141592 + 0.005, 400, 400])
        mat = [[1, 0, 0, 0, -1],
               [0, 1, 0, 0, -1],
               [0, 0, 1, 0, -1],
               [0, 0, 0, 1, -1]]
        const \
            = [0, 0, 0, 0]

        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              )
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 6:
        _ = 'Safe if coc is minimal'
        init_lb, init_ub = get_init_bounds([12000, 0.7, -3.141592, 100, 0],
                                           [62000, 3.141592, -3.141592 + 0.005, 1200, 1200])
        mat = [[-1, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0],
               [-1, 0, 0, 1, 0],
               [-1, 0, 0, 0, 1]]
        const \
            = [0, 0, 0, 0]
        mat, const \
            = get_out_constr(mat, const)

        out_spec1 = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                               )
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec1], Dataset.ACAS)
    if id == 7:
        _ = 'safe if strong left is not minimal and strong right is not minimal'
        mat1 = [[-1, 0, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, 0, -1, 1, 0]]
        const1\
            = [0.0, 0, 0]
        mat1, const1\
            = get_out_constr(mat1, const1
                             )
        init_lb1, init_ub1 = get_init_bounds([0, -3.141592, -3.141592, 100, 0], [60760, 3.141592, 3.141592, 1200, 1200])

        out_spec1 = Constraint(OutSpecType.GLOBAL, constr_mat=(mat1, const1
                                                               ), is_conjunctive=False)

        mat2 = [[-1, 0, 0, 0, 1],
                [0, -1, 0, 0, 1],
                [0, 0, -1, 0, 1]]
        const2\
            = [0, 0, 0]
        mat2, const2\
            = get_out_constr(mat2, const2
                             )
        init_lb2, init_ub2 = get_init_bounds([0, -3.141592, -3.141592, 100, 0], [60760, 3.141592, 3.141592, 1200, 1200])

        out_spec2 = Constraint(OutSpecType.GLOBAL, constr_mat=(mat2, const2
                                                               ), is_conjunctive=False)
        return Property([init_lb1, init_lb2], [init_ub1, init_ub2], InputSpecType.GLOBAL, [out_spec1, out_spec2],
                        Dataset.ACAS)
    if id == 8:
        _ = 'safe if weak left is minimal or COC is minimal'
        init_lb1, init_ub1 = get_init_bounds([0, -3.141592, -0.1, 600, 600], [60760, -0.75 * 3.141592, 0.1, 1200, 1200])
        init_lb2, init_ub2 = get_init_bounds([0, -3.141592, -0.1, 600, 600], [60760, -0.75 * 3.141592, 0.1, 1200, 1200])

        mat1 = [[1, -1, 0, 0, 0],
                [0, -1, 1, 0, 0],
                [0, -1, 0, 1, 0],
                [0, -1, 0, 0, 1]]
        const1\
            = [0, 0, 0, 0]
        mat1, const1\
            = get_out_constr(mat1, const1
                             )
        out_spec1 = Constraint(OutSpecType.GLOBAL, constr_mat=(mat1, const1
                                                               ))

        mat2 = [[-1, 1, 0, 0, 0],
                [-1, 0, 1, 0, 0],
                [-1, 0, 0, 1, 0],
                [-1, 0, 0, 0, 1]]
        const2\
            = [0, 0, 0, 0]
        mat2, const2\
            = get_out_constr(mat2, const2
                             )
        out_spec2 = Constraint(OutSpecType.GLOBAL, constr_mat=(mat2, const2
                                                               ))
        return Property([init_lb1, init_lb2], [init_ub1, init_ub2], InputSpecType.GLOBAL, [out_spec1, out_spec2],
                        Dataset.ACAS)
    if id == 9:
        _ = 'strong left should be minimal'
        init_lb, init_ub = get_init_bounds([2000, -0.4, -3.141592, 100, 0], [7000, -0.14, -3.141592 + 0.01, 150, 150])
        mat = [[0, 1, 0, -1, 0],
               [0, 0, 1, -1, 0],
               [1, 0, 0, -1, 0],
               [0, 0, 0, -1, 1]]
        const \
            = [0, 0, 0, 0]
        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              )
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    if id == 10:
        _ = 'safe if coc is minimal'
        init_lb, init_ub = get_init_bounds([36000, 0.7, -3.141592, 900, 600],
                                           [60760, 3.141592, -3.141592 + 0.01, 1200, 1200])
        mat = [[-1, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0],
               [-1, 0, 0, 1, 0],
               [-1, 0, 0, 0, 1]]
        const \
            = [0, 0, 0, 0]
        mat, const \
            = get_out_constr(mat, const)

        out_spec = Constraint(OutSpecType.GLOBAL, constr_mat=(mat, const)
                              )
        return Property([init_lb], [init_ub], InputSpecType.GLOBAL, [out_spec], Dataset.ACAS)
    else:
        raise ValueError("Property not yer supported: ", id)


def get_out_constr(mat, const)\
        :
    mat = torch.tensor(mat).type(torch.float).T
    const \
        = torch.tensor(const)

    return mat, const



def get_init_bounds(init_lb, init_ub):
    init_lb = torch.tensor(init_lb).T
    init_ub = torch.tensor(init_ub).T
    mean, std = util.get_mean_std(Dataset.ACAS)
    mean, std = mean.flatten(), std.flatten()
    init_lb = (init_lb - mean) / std
    init_ub = (init_ub - mean) / std
    return init_lb, init_ub
