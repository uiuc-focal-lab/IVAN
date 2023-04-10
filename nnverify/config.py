from nnverify.specs.input_spec import InputSpecType

NET_HOME = "nnverify/nets/"
DEVICE = 'cpu'


class Args:
    def __init__(self, net, domain, count=None, eps=0.01, dataset='mnist', spec_type=InputSpecType.LINF, split=None,
                 pt_method=None, timeout=None, parallel=False, batch_size=10, initial_split=False, attack=None):
        self.net = NET_HOME + net
        self.domain = domain
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.spec_type = spec_type
        self.split = split
        self.pt_method = pt_method
        self.timeout = timeout
        self.parallel = parallel
        self.initial_split = initial_split
        self.attack = attack
        self.batch_size = batch_size


log_file = "log.txt"
log_enabled = False


def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "IVAN"
baseline = "Baseline"

# Networks
MNIST_FFN_01 = "mnist_0.1.onnx"  # 9508 neuron
MNIST_FFN_03 = "mnist_0.3.onnx"  # 12794 neurons
MNIST_FFN_L2 = "mnist-net_256x2.onnx"  # 512 neurons
MNIST_FFN_L4 = "mnist-net_256x4.onnx"   # 1024 neurons
MNIST_FFN_torch1 = "cpt/fc1.pt"

# CIFAR Networks
CIFAR_CONV_2_255 = "cifar10_2_255.onnx"  # 49402 neurons
CIFAR_CONV_8_255 = "cifar10_8_255.onnx"  # 16634 neurons
CIFAR_CONV_SMALL = "convSmall_pgd_cifar.onnx"   # 3,604 neurons
CIFAR_CONV_BIG = "convBig_diffai_cifar.onnx"  # 62,464 neurons

# OVAL21 CIFAR
CIFAR_OVAL_BASE = "oval21/cifar_base_kw.onnx"   # 3172 neurons
CIFAR_OVAL_WIDE = "oval21/cifar_wide_kw.onnx"   # 6244 neurons
CIFAR_OVAL_DEEP = "oval21/cifar_deep_kw.onnx"   # 6756 neurons
CIFAR_OVAL_BASE_T = "oval21/cifar_base_kw.pth"
CIFAR_OVAL_WIDE_T = "oval21/cifar_wide_kw.pth"
CIFAR_OVAL_DEEP_T = "oval21/cifar_deep_kw.pth"


def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name
