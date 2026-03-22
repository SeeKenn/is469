from .llm_only import LLMOnlyPipeline
from .baseline import BaselinePipeline
from .advanced_a import AdvancedAPipeline
from .advanced_b import AdvancedBPipeline
from .advanced_c import AdvancedCPipeline
from .advanced_d import AdvancedDPipeline
from .advanced_e import AdvancedEPipeline

ALL_PIPELINES = {
    "v0_llm_only": LLMOnlyPipeline,
    "v1_baseline": BaselinePipeline,
    "v2_advanced_a": AdvancedAPipeline,
    "v3_advanced_b": AdvancedBPipeline,
    "v4_advanced_c": AdvancedCPipeline,
    "v5_advanced_d": AdvancedDPipeline,
    "v6_advanced_e": AdvancedEPipeline,
}

__all__ = [
    "LLMOnlyPipeline",
    "BaselinePipeline",
    "AdvancedAPipeline",
    "AdvancedBPipeline",
    "AdvancedCPipeline",
    "AdvancedDPipeline",
    "AdvancedEPipeline",
    "ALL_PIPELINES",
]
