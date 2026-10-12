from .backbone.alm import ALM
from .backbone.dlm import BlockCausalDLM
from .bottleneck.cb import ConceptBottleneckMixin
from .bottleneck.residual import ConceptResidualMixin
from .bottleneck.residual_naive import ConceptNaiveResidualMixin
from .bottleneck.steerling import SteerlingMixin

BACKBONES = {"alm": ALM, "dlm": BlockCausalDLM}

BOTTLENECKS = {
    "cb": ConceptBottleneckMixin,
    "naive_residual": ConceptNaiveResidualMixin,
    "residual": ConceptResidualMixin,
    "steerling": SteerlingMixin,
}


def build_model(backbone: str, bottleneck: str = "none", **kwargs):
    backbone_cls = BACKBONES[backbone]
    if bottleneck == "none":
        return backbone_cls(**kwargs)

    # Put Bottleneck first so Backbone's __init__ finishes before Bottleneck's runs,
    # preventing Backbone from overwriting Bottleneck's layer setups
    bottleneck_cls = BOTTLENECKS[bottleneck]
    cls = type(f"{bottleneck}_{backbone}", (bottleneck_cls, backbone_cls), {})
    return cls(**kwargs)
