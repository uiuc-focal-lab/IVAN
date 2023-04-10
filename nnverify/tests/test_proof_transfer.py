import nnverify.proof_transfer.proof_transfer as pt
import nnverify.proof_transfer.approximate as ap

from unittest import TestCase
from nnverify import config
from nnverify.specs.property import InputSpecType
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from nnverify.common.dataset import Dataset
from nnverify.training.training_args import TrainArgs


class TestReordering(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=REORDERING(0, 0.01),
                               timeout=10)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(0, 0.01),
                               timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_random_1e3(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Random(1e-3), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=1, pt_method=REORDERING(0, 0.01),
                               timeout=200)
        pt.proof_transfer(args)


class TestReusing(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE, timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=ProofTransferMethod.REUSE,
                               timeout=200)
        pt.proof_transfer(args)


class TestCompleteAll(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL, timeout=200)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_prune30(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Prune(30), dataset=Dataset.MNIST, eps=0.03,
                               split=Split.RELU_ESIP_SCORE, count=100, pt_method=ProofTransferMethod.ALL,
                               timeout=200)
        pt.proof_transfer(args)


class TestCompletePruneCIFAR(TestCase):
    def test_conv_small_lp_esip_cifar_int16(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.005), timeout=400)
        pt.proof_transfer(args)

    def test_conv_big_lp_esip_cifar_int16(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_BIG, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.005), timeout=400)
        pt.proof_transfer(args)

    def test_conv_small_lp_esip_cifar_int8(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.005), timeout=400)
        pt.proof_transfer(args)

    def test_conv_big_lp_esip_cifar_int8(self):
        args = pt.TransferArgs(net=config.CIFAR_CONV_BIG, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.CIFAR10, eps=2/255,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.005), timeout=400)
        pt.proof_transfer(args)


class TestCompletePatch(TestCase):
    def test_conv_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST,
                               attack=InputSpecType.PATCH, split=Split.INPUT, pt_method=ProofTransferMethod.ALL,
                               timeout=30)
        pt.proof_transfer(args)

    def test_conv_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST,
                               attack=InputSpecType.PATCH, split=Split.INPUT, pt_method=ProofTransferMethod.ALL,
                               timeout=30)
        pt.proof_transfer(args)


class TestCompleteAcas(TestCase):
    def test_deepz_acas_onnx_int8(self):
        args = pt.TransferArgs(domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8),
                               dataset=Dataset.ACAS, split=Split.INPUT_SB, pt_method=IVAN(0, 0.01), timeout=20)
        pt.proof_transfer_acas(args)

    def test_deepz_acas_onnx_int16(self):
        args = pt.TransferArgs(domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT16),
                               dataset=Dataset.ACAS, split=Split.INPUT_SB, pt_method=IVAN(0, 0.01), timeout=20)
        pt.proof_transfer_acas(args)


class TestPrune(TestCase):
    def test_conv01_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test_conv01_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.1,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.003),
                               timeout=100)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int8(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT8), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(0, 0.003),
                               timeout=30)
        pt.proof_transfer(args)

    def test_mlp_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.MNIST, eps=0.02,
                               split=Split.RELU_ESIP_SCORE, count=10, pt_method=IVAN(0, 0.003),
                               timeout=10)
        pt.proof_transfer(args)

    def test_oval_lp_esip_onnx_int16(self):
        args = pt.TransferArgs(net="oval21/cifar_wide_kw.onnx", domain=Domain.LP,
                               approx=ap.Quantize(ap.QuantizationType.INT16), dataset=Dataset.CIFAR10, eps=8/255,
                               count=20, pt_method=IVAN(0, 0.003), split=Split.RELU_ESIP_SCORE,
                               timeout=20)
        pt.proof_transfer(args)


class TestFinetune(TestCase):
    def test_oval_lp_esip_ft1(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-7)),
                               dataset=Dataset.OVAL_CIFAR, count=10, pt_method=IVAN(0, 0.003),
                               split=Split.RELU_ESIP_SCORE, timeout=1000)
        pt.proof_transfer(args)

    def test_oval_lp_esip_ft2(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-6)),
                               dataset=Dataset.OVAL_CIFAR, count=10, pt_method=IVAN(0, 0.003),
                               split=Split.RELU_ESIP_SCORE, timeout=1000)
        pt.proof_transfer(args)

    def test_oval_lp_esip_ft3(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-5)),
                               dataset=Dataset.OVAL_CIFAR, count=10, pt_method=IVAN(0, 0.003),
                               split=Split.RELU_ESIP_SCORE, timeout=1000)
        pt.proof_transfer(args)

    def test_oval_lp_esip_ft4(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-4)),
                               dataset=Dataset.OVAL_CIFAR, count=10, pt_method=IVAN(0, 0.003),
                               split=Split.RELU_ESIP_SCORE, timeout=1000)
        pt.proof_transfer(args)

    def test_oval_lp_esip_robust_ft1(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE_T, domain=Domain.LP,
                               approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-7, trainer=Domain.LIRPA_CROWN_IBP, epsilon=4/255)),
                               dataset=Dataset.CIFAR10, count=30, pt_method=IVAN(0, 0.003), eps=4 / 255,
                               split=Split.RELU_ESIP_SCORE, timeout=400)
        pt.proof_transfer(args)

    def test_oval_lp_esip_robust_ft2(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE_T, domain=Domain.LP,
                               approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-6, trainer=Domain.LIRPA_CROWN_IBP, epsilon=4/255)),
                               dataset=Dataset.CIFAR10, count=30, pt_method=IVAN(0, 0.003), eps=4 / 255,
                               split=Split.RELU_ESIP_SCORE, timeout=400)
        pt.proof_transfer(args)

    def test_oval_lp_esip_robust_ft3(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE_T, domain=Domain.LP,
                               approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-5, trainer=Domain.LIRPA_CROWN_IBP, epsilon=4/255)),
                               dataset=Dataset.CIFAR10, count=30, pt_method=IVAN(0, 0.003), eps=4 / 255,
                               split=Split.RELU_ESIP_SCORE, timeout=400)
        pt.proof_transfer(args)

    def test_oval_lp_esip_robust_ft4(self):
        args = pt.TransferArgs(net=config.CIFAR_OVAL_WIDE_T, domain=Domain.LP,
                               approx=ap.Finetune(TrainArgs(epochs=1, lr=1e-4, trainer=Domain.LIRPA_CROWN_IBP, epsilon=4/255)),
                               dataset=Dataset.CIFAR10, count=30, pt_method=IVAN(0, 0.003), eps=4 / 255,
                               split=Split.RELU_ESIP_SCORE, timeout=400)
        pt.proof_transfer(args)
