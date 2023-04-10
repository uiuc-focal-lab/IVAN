from nnverify.common import Domain
from nnverify.common.network import LayerType
from nnverify.domains.box import BoxTransformer
from nnverify.domains.deeppoly import DeeppolyTransformer
from nnverify.domains.deepz import ZonoTransformer
from nnverify.domains.lirpa import LirpaTransformer
from nnverify.domains.lp import LPTransformer
from nnverify.parse import forward_layers
from nnverify.util import is_lirpa_domain


def get_domain_transformer(args, net, prop, complete=True):
    if args.domain == Domain.DEEPPOLY:
        transformer = DeeppolyTransformer(prop, complete=complete)
    elif args.domain == Domain.DEEPZ:
        transformer = ZonoTransformer(prop, complete=complete)
    elif args.domain == Domain.BOX:
        transformer = BoxTransformer(prop, complete=complete)
    elif args.domain == Domain.LP:
        transformer = LPTransformer(prop, net, complete=complete)
    elif is_lirpa_domain(args.domain):
        transformer = LirpaTransformer(prop, args.domain, args.dataset, complete=complete)
    else:
        raise ValueError("Unexpected domain!")

    return build_transformer(transformer, net, prop)


def build_transformer(transformer, net, prop, relu_mask=None):
    if type(transformer) in [LPTransformer, LirpaTransformer]:
        if net[-1].type != LayerType.Linear:
            raise ValueError("We assume that the last layer of the network is a Linear layer!")
        transformer.build(net, prop, relu_mask=relu_mask)
        return transformer

    # For all abstraction based domains
    transformer = forward_layers(net, relu_mask, transformer)

    return transformer
