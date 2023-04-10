from unittest import TestCase
from nnverify import config
from nnverify.analyzer import Analyzer
from nnverify.attack.attack import AutoAttack
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.common.dataset import Dataset
from nnverify.specs.input_spec import InputSpecType


class TestAttack(TestCase):
    # 47% attacked
    def test_mlp_auto_attack(self):
        args = config.Args(net=config.MNIST_FFN_torch1, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.1,
                           attack=AutoAttack())
        Analyzer(args).run_analyzer()

    def test_mlp_auto_attack_onnx(self):
        args = config.Args(net=config.MNIST_FFN_L2, domain=Domain.DEEPZ, dataset=Dataset.MNIST, eps=0.1,
                           attack=AutoAttack())
        Analyzer(args).run_analyzer()

    def test_acas_auto_attack(self):
        args = config.Args(net=config.ACASXU(1, 1), domain=Domain.DEEPZ, dataset=Dataset.ACAS,
                           attack=AutoAttack(), spec_type=InputSpecType.GLOBAL)
        Analyzer(args).run_analyzer()