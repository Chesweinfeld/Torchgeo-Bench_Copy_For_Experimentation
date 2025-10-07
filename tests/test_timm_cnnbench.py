import pytest
import torch
from src.bench_models import TimmCNNBench

@pytest.mark.parametrize("model_name", [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "convnext_tiny",
])
def test_timm_cnnbench_forward(model_name):
    # Test with 3-channel input, batch size 2
    model = TimmCNNBench(num_channels=3, model_name=model_name, pretrained=True)
    images = torch.randn(2, 3, 64, 64)
    feats = model.forward_features(images)
    assert feats.shape[0] == 2
    assert feats.shape[1] > 0  # Should be feature dim
    assert not torch.isnan(feats).any()
