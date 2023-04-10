import torch
import torchvision
from torchvision import transforms

from nnverify import util
from nnverify.specs.properties.acasxu import get_acas_spec
from nnverify.specs.property import Property, InputSpecType, OutSpecType
from nnverify.specs.out_spec import Constraint
from nnverify.specs.relu_spec import Reluspec
from nnverify.util import prepare_data
from nnverify.common import Status
from nnverify.common.dataset import Dataset

'''
Specification holds upper bound and lower bound on ranges for each dimension.
In future, it can be extended to handle other specs such as those for rotation 
or even the relu stability could be part of specification.
'''


class Spec:
    def __init__(self, input_spec, relu_spec=None, parent=None, status=Status.UNKNOWN):
        self.input_spec = input_spec
        self.relu_spec = relu_spec
        self.children = []
        self.status = status
        self.lb = 0
        self.chosen_split = None
        self.parent = parent

    def update_status(self, status, lb):
        self.status = status
        if lb is None:
            self.lb = 0
        else:
            self.lb = lb

    def reset_status(self):
        self.status = Status.UNKNOWN
        self.lb = 0


class SpecList(list):
    def batch(self, batch_size=10):
        i = 0
        while i < len(self):
            merge_specs = self[i:i+batch_size]

            i += batch_size



def create_relu_spec(unstable_relus):
    relu_mask = {}

    for layer in range(len(unstable_relus)):
        for id in unstable_relus[layer]:
            relu_mask[(layer, id)] = 0

    return Reluspec(relu_mask)


def score_relu_grad(spec, prop, net=None):
    """
    Gives a score to each relu based on its gradient. Higher score indicates higher preference while splitting.
    """
    relu_spec = spec.relu_spec
    relu_mask = relu_spec.relu_mask

    # Collect all relus that are not already split
    relu_spec.relu_score = {}

    # TODO: support CIFAR10
    ilb = prop.input_lb
    inp = ilb.reshape(1, 1, 28, 28)

    # Add all relu layers for which we need gradients
    layers = {}
    for relu in relu_mask.keys():
        layers[relu[0]] = True

    grad_map = {}

    # use ilb and net to get the grad for each neuron
    for layer in layers.keys():
        x = net[:layer * 2 + 2](inp).detach()
        x.requires_grad = True

        y = net[layer * 2 + 2:](x)
        y.mean().backward()

        grad_map[layer] = x.grad[0]

    for relu in relu_mask.keys():
        relu_spec.relu_score[relu] = abs(grad_map[relu[0]][relu[1]])

    return relu_spec.relu_score


def score_relu_esip(zono_transformer):
    """
    The relu score here is similar to the direct score defined in DeepSplit paper
    https://www.ijcai.org/proceedings/2021/0351.pdf
    """
    center = zono_transformer.centers[-1]
    cof = zono_transformer.cofs[-1]
    cof_abs = torch.sum(torch.abs(cof), dim=0)
    lb = center - cof_abs

    adv_index = torch.argmin(lb)
    relu_score = {}

    for noise_index, relu_index in zono_transformer.map_for_noise_indices.items():
        # Score relu based on effect on one label
        relu_score[relu_index] = torch.abs(cof[noise_index, adv_index])

        # Score relu based on effect on all label
        # relu_score[relu_index] = torch.sum(torch.abs(cof[noise_index, :]))

    return relu_score


def score_relu_esip2(zono_transformer):
    """
    The relu score here is similar to the direct+indirect score defined in DeepSplit paper
    https://www.ijcai.org/proceedings/2021/0351.pdf
    """
    center = zono_transformer.centers[-1]
    cof = zono_transformer.cofs[-1]
    cof_abs = torch.sum(torch.abs(cof), dim=0)
    lb = center - cof_abs

    adv_index = torch.argmin(lb)
    dir_score = {}
    indir_score = {}
    relu_score = {}

    # Compute direct score
    for noise_index, relu_index in zono_transformer.map_for_noise_indices.items():
        # Score relu based on effect on one label
        dir_score[relu_index] = torch.abs(cof[noise_index, adv_index])
        indir_score[relu_index] = 0

        # Score relu based on effect on all label
        # dir_score[relu_index] = torch.sum(torch.abs(cof[noise_index, :]))

    # Compute indirect score
    for noise_index1, relu_index1 in zono_transformer.map_for_noise_indices.items():
        for noise_index2, relu_index2 in zono_transformer.map_for_noise_indices.items():
            if relu_index1[0] >= relu_index2[0]:
                # Indirect effect for next layers
                continue

            layer2 = relu_index2[0]
            id2 = relu_index2[1]
            if len(zono_transformer.relu_layer_cofs[layer2]) > noise_index1:
                s = torch.abs(zono_transformer.relu_layer_cofs[layer2][noise_index1, id2])
                t = torch.sum(torch.abs(zono_transformer.relu_layer_cofs[layer2][:, id2]))
                indir_score[relu_index1] += (s / t) * dir_score[relu_index2]

    # Compute total ReLU score
    for noise_index, relu_index in zono_transformer.map_for_noise_indices.items():
        relu_score[relu_index] = dir_score[relu_index] + indir_score[relu_index]

    return relu_score


def get_specs(dataset, spec_type=InputSpecType.LINF, eps=0.01, count=None):
    if dataset == Dataset.MNIST or dataset == Dataset.CIFAR10:
        if spec_type == InputSpecType.LINF:
            if count is None:
                count = 100
            testloader = prepare_data(dataset, batch_size=count)
            inputs, labels = next(iter(testloader))
            props = get_linf_spec(inputs, labels, eps, dataset)
        elif spec_type == InputSpecType.PATCH:
            if count is None:
                count = 10
            testloader = prepare_data(dataset, batch_size=count)
            inputs, labels = next(iter(testloader))
            props = get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2)
            width = inputs.shape[2] - 2 + 1
            length = inputs.shape[3] - 2 + 1
            pos_patch_count = width * length
            specs_per_patch = pos_patch_count
            # labels = labels.unsqueeze(1).repeat(1, pos_patch_count).flatten()
        return props, inputs
    elif dataset == Dataset.ACAS:
        return get_acas_props(count), None
    elif dataset == Dataset.OVAL_CIFAR:
        return get_oval_cifar_props(count)
    else:
        raise ValueError("Unsupported specification dataset!")


def get_oval_cifar_props(count):
    pdprops = 'base_easy.pkl'  # pdprops = 'base_med.pkl' or pdprops = 'base_hard.pkl'
    path = 'data/cifar_exp/'
    import pandas as pd
    gt_results = pd.read_pickle(path + pdprops)
    # batch ids were used for parallel processing in the original paper.
    batch_ids = gt_results.index[0:count]
    props = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(), normalize]))
    for new_idx, idx in enumerate(batch_ids):
        imag_idx = gt_results.loc[idx]['Idx']
        adv_label = gt_results.loc[idx]['prop']
        eps_temp = gt_results.loc[idx]['Eps']

        ilb, iub, true_label = util.ger_property_from_id(imag_idx, eps_temp, cifar_test)
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=true_label, adv_label=adv_label)
        props.append(Property(ilb, iub, InputSpecType.LINF, out_constr, Dataset.CIFAR10))
    return props, None


def get_acas_props(count):
    props = []
    if count is None:
        count = 10
    for i in range(1, count + 1):
        props.append(get_acas_spec(i))
    return props


def get_linf_spec(inputs, labels, eps, dataset):
    properties = []

    for i in range(len(inputs)):
        image = inputs[i]

        ilb = torch.clip(image - eps, min=0., max=1.)
        iub = torch.clip(image + eps, min=0., max=1.)

        mean, std = util.get_mean_std(dataset)

        ilb = (ilb - mean) / std
        iub = (iub - mean) / std
        image = (image - mean) / std

        ilb = ilb.reshape(-1)
        iub = iub.reshape(-1)

        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        properties.append(Property(ilb, iub, InputSpecType.LINF, out_constr, dataset, input=image))

    return properties


def get_patch_specs(inputs, labels, eps, dataset, p_width=2, p_length=2):
    width = inputs.shape[2] - p_width + 1
    length = inputs.shape[3] - p_length + 1
    pos_patch_count = width * length
    final_bound_count = pos_patch_count

    patch_idx = torch.arange(pos_patch_count, dtype=torch.long)[None, :]

    x_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    y_cord = torch.zeros((1, pos_patch_count), dtype=torch.long, requires_grad=False)
    idx = 0
    for w in range(width):
        for l in range(length):
            x_cord[0, idx] = w
            y_cord[0, idx] = l
            idx = idx + 1

    # expand the list to include coordinates from the complete patch
    patch_idx = [patch_idx.flatten()]
    x_cord = [x_cord.flatten()]
    y_cord = [y_cord.flatten()]
    for w in range(p_width):
        for l in range(p_length):
            patch_idx.append(patch_idx[0])
            x_cord.append(x_cord[0] + w)
            y_cord.append(y_cord[0] + l)

    patch_idx = torch.cat(patch_idx, dim=0)
    x_cord = torch.cat(x_cord, dim=0)
    y_cord = torch.cat(y_cord, dim=0)

    # create masks for each data point
    mask = torch.zeros([1, pos_patch_count, inputs.shape[2], inputs.shape[3]],
                       dtype=torch.uint8)
    mask[:, patch_idx, x_cord, y_cord] = 1
    mask = mask[:, :, None, :, :]
    mask = mask.cpu()

    iubs = torch.clip(inputs + 1, min=0., max=1.)
    ilbs = torch.clip(inputs - 1, min=0., max=1.)

    iubs = torch.where(mask, iubs[:, None, :, :, :], inputs[:, None, :, :, :])
    ilbs = torch.where(mask, ilbs[:, None, :, :, :], inputs[:, None, :, :, :])

    mean, stds = util.get_mean_std(dataset)

    iubs = (iubs - mean) / stds
    ilbs = (ilbs - mean) / stds

    # (data, patches, spec)
    iubs = iubs.view(iubs.shape[0], iubs.shape[1], -1)
    ilbs = ilbs.view(ilbs.shape[0], ilbs.shape[1], -1)

    props = []

    for i in range(ilbs.shape[0]):
        out_constr = Constraint(OutSpecType.LOCAL_ROBUST, label=labels[i])
        props.append(
            Property(ilbs[i], iubs[i], InputSpecType.PATCH, out_constr, dataset, input=(inputs[i] - mean) / stds))
    return props

