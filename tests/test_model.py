import torch
import pytest
from leaf_net.model import EcoConvNet, WeightedPooling

def test_ecoconvnet_forward():
    model = EcoConvNet(img_size=(32, 32), patch_size=(2, 2), in_channels=3, num_classes=10)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 10), f"Expected shape (2, 10), got {output.shape}"

def test_ecoconvnet_custom_resolution():
    model = EcoConvNet(img_size=(128, 128), patch_size=(4, 4), in_channels=1, num_classes=5)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 5), f"Expected shape (1, 5), got {output.shape}"

def test_weighted_pooling():
    input_dim = 64
    pooling = WeightedPooling(input_dim=input_dim)
    
    batch_size = 4
    seq_len = 16
    h = torch.randn(batch_size, seq_len, input_dim)
    output = pooling(h)
    
    assert output.shape == (batch_size, input_dim), f"Expected shape ({batch_size}, {input_dim}), got {output.shape}"
