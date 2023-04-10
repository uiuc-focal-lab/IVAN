import csv
import os
import resource

import nnverify.common as common
from time import gmtime, strftime
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import nnverify.parse as parse
import nnverify.training.models as models
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.networks import FullyConnected, Conv

rt.set_default_logger_severity(3)


def get_torch_net(net_file, dataset, device='cpu'):
    net_name = net_file.split('/')[-1].split('.')[-2]

    if 'cpt' in net_file:
        return get_torch_test_net(net_name, net_file)

    if dataset == Dataset.MNIST:
        model = models.Models[net_name](in_ch=1, in_dim=28)
    elif dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
        model = models.Models[net_name](in_ch=3, in_dim=32)
    else:
        raise ValueError("Unsupported dataset")

    if 'kw' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    elif 'eran' in net_file:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'][0])
    else:
        model.load_state_dict(torch.load(net_file, map_location=torch.device(device))['state_dict'])

    return model


def get_torch_test_net(net_name, path, device='cpu', input_size=28):
    if net_name == 'fc1':
        net = FullyConnected(device, input_size, [50, 10]).to(device)
    elif net_name == 'fc2':
        net = FullyConnected(device, input_size, [100, 50, 10]).to(device)
    elif net_name == 'fc3':
        net = FullyConnected(device, input_size, [100, 100, 10]).to(device)
    elif net_name == 'fc4':
        net = FullyConnected(device, input_size, [100, 100, 50, 10]).to(device)
    elif net_name == 'fc5':
        net = FullyConnected(device, input_size, [100, 100, 100, 10]).to(device)
    elif net_name == 'fc6':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 10]).to(device)
    elif net_name == 'fc7':
        net = FullyConnected(device, input_size, [100, 100, 100, 100, 100, 10]).to(device)
    elif net_name == 'conv1':
        net = Conv(device, input_size, [(16, 3, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv2':
        net = Conv(device, input_size, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(device)
    elif net_name == 'conv3':
        net = Conv(device, input_size, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(device)
    else:
        assert False

    net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return net.layers


def parse_spec(spec):
    with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])

    return true_label, pixel_values, eps


def sample(net_name, ilb, iub):
    print('Sample some output points:')
    sess = rt.InferenceSession(net_name)
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: ilb.numpy().reshape(1, -1)})
    print('onnx output:', pred_onnx)
    pred_onnx = sess.run(None, {input_name: iub.numpy().reshape(1, -1)})
    print('onnx output2:', pred_onnx)
    pred_onnx = sess.run(None, {input_name: ((iub + ilb) / 2).numpy().reshape(1, -1)})
    print('onnx output3:', pred_onnx)


def reshape_input(x, dataset):
    """
    @return: x reshaped to (batch_size, channels, *input_size)
    """
    if dataset == Dataset.MNIST:
        x = x.reshape(-1, 1, 28, 28)
    elif dataset == Dataset.CIFAR10:
        x = x.reshape(-1, 3, 32, 32)
    elif dataset == Dataset.ACAS:
        # Making it 2-D to avoid ONNX:MatMul (instead of ONNX:GEMM) translation used in AutoLirpa
        x = x.reshape(-1, 5)
    else:
        raise ValueError("Unknown dataset!")
    return x


def compute_output_tensor(inp, net):
    if net.net_format == 'torch':
        out = net.torch_net(inp)
        adv_label = torch.argmax(out)
        out = out.flatten()
    elif net.net_format == 'onnx':
        sess = rt.InferenceSession(net.net_name)
        inp = inp.reshape(net.input_shape)
        out = sess.run(None, {net.input_name: inp.numpy()})
        out = torch.tensor(out).flatten()
        adv_label = torch.argmax(out).item()
    else:
        raise ValueError("We only support torch and onnx!")

    return adv_label, out


def prepare_data(dataset, train=False, batch_size=100, normalize=False):
    transform_list = [torchvision.transforms.ToTensor()]

    if normalize:
        mean, std = get_mean_std(dataset)
        transform_list.append(torchvision.transforms.Normalize(mean=mean, std=std))

    tr = torchvision.transforms.Compose(transform_list)

    if dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
        test_set = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=tr)
    elif dataset == Dataset.MNIST:
        test_set = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=tr)
    else:
        raise ValueError("Unsupported Dataset")

    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return testloader


def get_mean_std(dataset):
    if dataset == Dataset.MNIST:
        means = [0]
        stds = [1]
    elif dataset == Dataset.CIFAR10 or dataset == Dataset.OVAL_CIFAR:
        # For the model that is loaded from cert def this normalization was
        # used
        stds = [0.2023, 0.1994, 0.2010]
        means = [0.4914, 0.4822, 0.4465]
        # means = [0.5, 0.5, 0.5]
        # stds = [1, 1, 1]
    elif dataset == Dataset.ACAS:
        means = [19791.091, 0.0, 0.0, 650.0, 600.0]
        stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
    else:
        raise ValueError("Unsupported Dataset!")
    return torch.tensor(means).reshape(-1, 1, 1), torch.tensor(stds).reshape(-1, 1, 1)


def ger_property_from_id(imag_idx, eps_temp, cifar_test):
    x, y = cifar_test[imag_idx]
    x = x.unsqueeze(0)

    ilb = (x - eps_temp).flatten()
    iub = (x + eps_temp).flatten()

    return ilb, iub, torch.tensor(y)


def get_net_format(net_name):
    net_format = None
    if 'pt' in net_name:
        net_format = 'torch'
    if 'onnx' in net_name:
        net_format = 'onnx'
    return net_format


def is_lirpa_domain(domain):
    lirpa_domains = [Domain.LIRPA_IBP, Domain.LIRPA_CROWN, Domain.LIRPA_CROWN_IBP, Domain.LIRPA_CROWN_OPT]
    if domain in lirpa_domains:
        return True
    return False


def get_net(net_name, dataset):
    net_format = get_net_format(net_name)
    if net_format == 'torch':
        # Load the model
        net_torch = get_torch_net(net_name, dataset)
        net = parse.parse_torch_layers(net_torch)

    elif net_format == 'onnx':
        net_onnx = onnx.load(net_name)
        net = parse.parse_onnx_layers(net_onnx)
    else:
        raise ValueError("Unsupported net format!")

    net.net_name = net_name
    return net


def log_memory_usage():
    mu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mu /= (1024*1024)
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'memory_usage.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['Memory Usage at', strftime("%Y-%m-%d %H:%M:%S", gmtime())])
        writer.writerow([str(mu) + 'MBs'])


def onnx2torch(onnx_model):
    import onnx2pytorch

    # find the input shape from onnx_model generally
    # https://github.com/onnx/onnx/issues/2657
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

    if len(net_feed_input) != 1:
        # in some rare case, we use the following way to find input shape but this is not always true (collins-rul-cnn)
        net_feed_input = [onnx_model.graph.input[0]]

    onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=False, debug=True)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    return pytorch_model, onnx_shape