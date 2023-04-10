import nnverify.proof_transfer.proof_transfer as pt
import nnverify.proof_transfer.approximate as ap

from unittest import TestCase
from nnverify import config
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.proof_transfer.param_tune import write_result
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from nnverify.common.dataset import Dataset

COUNT = 100
SMALL_TIMEOUT = 100
BIG_TIMEOUT = 400


class TestIVAN(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0.003, 0.003),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 0.003),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 0.007),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 0.007),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)


class TestReuse(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE,
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE,
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE,
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE,
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_DEEP, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=ProofTransferMethod.REUSE, timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)


class TestReorder(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0.003, 0.003),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test2(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0.003, 0.003),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 0.007),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test4(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 0.007),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test5(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test6(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)

    def test7(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test8(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test9(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)

    def test10(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE, domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=4/255,
                               split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=REORDERING(0, 1e-4), timeout=BIG_TIMEOUT)
        pt.proof_transfer(args)


class TestRandom(TestCase):
    def test1(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Random(0.5),
                               dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0.003, 0.003),
                               timeout=SMALL_TIMEOUT)
        pt.proof_transfer(args)


class TestSensitivity(TestCase):
    def test1(self):
        thrs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        alphas = [0, 0.25, 0.5, 0.75, 1]

        for thr in thrs:
            for alpha in alphas:
                args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                                       dataset=Dataset.MNIST, eps=0.02,
                                       split=Split.RELU_ESIP_SCORE, count=COUNT, pt_method=IVAN(alpha, thr),
                                       timeout=SMALL_TIMEOUT)
                sp = pt.proof_transfer(args)
                write_result(alpha, sp, thr)



