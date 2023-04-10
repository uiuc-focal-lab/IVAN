import torch

from nnverify import config
from nnverify.util import reshape_input, compute_output_tensor


def check_adversarial(adv_ex, net, prop):
    """
    returns true if adv_ex is an adversarial example if following conditions hold
    1. net does not classify adv_ex to true_label.
    2. adv_ex lies within the ilb and iub. i.e. ilb <= adv_ex <= iub

    if @param adv_label_to_check is not None then we only check if the adv_ex is adversarial for that particular label
    """
    if adv_ex is None:
        return False

    # sanity check adv example
    adv_ex = adv_ex.clone().detach()

    num_err = 1e-5
    assert torch.max(prop.input_lb - adv_ex.flatten() - num_err).item() <= 0
    assert torch.max(adv_ex.flatten() - prop.input_ub - num_err).item() <= 0

    adv_ex = reshape_input(adv_ex, prop.dataset)
    adv_label, out = compute_output_tensor(adv_ex, net)

    if prop.is_local_robustness():
        true_label = prop.get_label()
        # print(out[true_label] - out)
        gap = out @ prop.output_constr_mat()  # gap = C^T Y
        config.write_log('True label ' + str(true_label) + '  UB: ' + str(gap))

        if prop.out_constr.is_conjunctive:
            violated = torch.any(gap < 0)
        else:
            violated = torch.all(gap < 0)

        if violated:  # Counter-example if gap < 0
            return True
    else:
        # TODO: The general check if the adversarial is real for any output constraint is not implemented
        return False