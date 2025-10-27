from .interface import BenchModel
from .bench_models import RCFBench, ImageStatsBench
from .timm import TimmPatchBenchModel

__all__: list[str] = [
	"BenchModel",
	"RCFBench",
	"ImageStatsBench",
    "TimmPatchBenchModel"
]