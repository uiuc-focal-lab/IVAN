import nnverify.config as config
import ssl

from unittest import TestCase
from nnverify.bnb import Split
from nnverify.common.dataset import Dataset
from nnverify.common import Domain
from nnverify.analyzer import Analyzer

# This is a hack to avoid SSL related errors while getting CIFAR10 data
ssl._create_default_https_context = ssl._create_unverified_context


class TestLirpa(TestCase):
    # Verifies 29%
    def test_ffn_ibp_no_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LIRPA_IBP, dataset=Dataset.CIFAR10, eps=2 / 255,
                           timeout=30)
        res = Analyzer(args).run_analyzer()

    # Verifies 55%
    def test_ffn_crown_no_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LIRPA_CROWN, dataset=Dataset.CIFAR10,
                           eps=2 / 255, timeout=30)
        res = Analyzer(args).run_analyzer()

    def test_ffn_crown_ibp_no_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LIRPA_CROWN_IBP, dataset=Dataset.CIFAR10,
                           eps=2 / 255, timeout=30)
        res = Analyzer(args).run_analyzer()

    def test_ffn_crown_opt_no_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LIRPA_CROWN_OPT, dataset=Dataset.CIFAR10,
                           eps=2 / 255, timeout=30)
        res = Analyzer(args).run_analyzer()


class TestLirpaOval(TestCase):
    # Should not verify anything without splitting. Just used for sanity checking the flow
    def test_ffn_crown_no_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LIRPA_CROWN, dataset=Dataset.OVAL_CIFAR,
                           count=100)
        res = Analyzer(args).run_analyzer()

    def test_ffn_lp_crown_split(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, dataset=Dataset.OVAL_CIFAR, count=100,
                           split=Split.RELU_ESIP_SCORE)
        res = Analyzer(args).run_analyzer()

    # Does not use Lirpa, should be worse than the torch version above
    def test_ffn_lp_crown_split_onnx(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE, domain=Domain.LP, dataset=Dataset.OVAL_CIFAR, count=100,
                           split=Split.RELU_ESIP_SCORE)
        res = Analyzer(args).run_analyzer()

    def test_ffn_lp_crown_split_kFSB(self):
        args = config.Args(net=config.CIFAR_OVAL_BASE_T, domain=Domain.LP, dataset=Dataset.OVAL_CIFAR, count=100,
                           split=Split.RELU_KFSB)
        res = Analyzer(args).run_analyzer()
