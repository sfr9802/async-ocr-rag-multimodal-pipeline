"""MULTIMODAL capability package — v1.

v1 definition: "multimodal" here means an image / PDF input gets run
through OCR + a visual-description provider, and the two signals are
fused into a retrieval query + grounding context that feed the
existing text RAG retriever and generator. This is explicitly NOT
true multimodal retrieval — see `docs/architecture.md`
"Multimodal v1 limitations" for the deferred items.

Public surface used by the rest of the worker:

  - MultimodalCapability, MultimodalCapabilityConfig
      (app.capabilities.multimodal.capability)
  - VisionDescriptionProvider, VisionDescriptionResult, VisionError
      (app.capabilities.multimodal.vision_provider)
  - HeuristicVisionProvider
      (app.capabilities.multimodal.heuristic_vision)
  - build_fusion, FusionResult
      (app.capabilities.multimodal.fusion)
"""

from app.capabilities.multimodal.capability import (
    MultimodalCapability,
    MultimodalCapabilityConfig,
)
from app.capabilities.multimodal.fusion import FusionResult, build_fusion
from app.capabilities.multimodal.heuristic_vision import HeuristicVisionProvider
from app.capabilities.multimodal.vision_provider import (
    VisionDescriptionProvider,
    VisionDescriptionResult,
    VisionError,
)

__all__ = [
    "MultimodalCapability",
    "MultimodalCapabilityConfig",
    "VisionDescriptionProvider",
    "VisionDescriptionResult",
    "VisionError",
    "HeuristicVisionProvider",
    "FusionResult",
    "build_fusion",
]
