<div align="center">
  <h1>🍃 Leaf-Net (EcoConvNet)</h1>
  <p><strong>Beyond Self-Attention: Designing Lightweight Transformer-Like Models with 1D-Convolutions for Green AI</strong></p>

  <p>
    <a href="https://github.com/kingknight07/Leaf-net/actions"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/kingknight07/Leaf-net/python-app.yml?style=flat-square&logo=github"></a>
    <a href="https://pypi.org/project/leaf-net/"><img alt="PyPI" src="https://img.shields.io/pypi/v/leaf-net.svg?style=flat-square&logo=pypi&logoColor=white"></a>
    <a href="https://github.com/kingknight07/Leaf-net/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square"></a>
    <a href="https://huggingface.co/models?search=leaf-net"><img alt="Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E?style=flat-square"></a>
    <a href="https://www.python.org/"><img alt="Python 3.8+" src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white"></a>
  </p>
</div>

<hr />

## 📖 Overview

**Leaf-Net (EcoConvNet)** is a novel, extremely lightweight, and state-of-the-art Transformer-like architecture. It fundamentally reimagines sequence modeling for energy-efficient computer vision—commonly referred to as **Green AI**. 

By completely replacing the computationally prohibitive quadratic Multi-Head Self-Attention (MHSA) module with a highly efficient sequence processor utilizing temporal 1D-Convolutions, Leaf-Net retains the modeling power of Transformers but with significantly reduced computational overhead and environmental footprint. Furthermore, it incorporates a revolutionary **Positional-Biased Attention Pooling** mechanism to encapsulate and govern global spatial information efficiently.

This repository hosts the official PyTorch implementation, constructed to be highly modular, exceptionally easy to use, and immediately adaptable to any custom dataset or domain.

## ✨ Key Features

- **Eco-Friendly & Lightweight**: Designed from the ground up for Green AI, dramatically reducing parameter counts and FLOPS compared to standard Vision Transformers (ViTs).
- **1D-Convolutional Sequence Processor**: Substitutes expensive MHSA blocks with agile 1D-Conv operations, achieving comparable or superior representations.
- **Positional-Biased Attention Pooling**: A novel pooling technique designed to distill global spatial configurations accurately without heavy computational burden.
- **Plug-and-Play**: Seamless integration into existing PyTorch pipelines and workflows.
- **Hugging Face Hub Integration**: Out-of-the-box support for loading, saving, and sharing weights via `huggingface_hub`.

## 📦 Installation

Leaf-Net requires Python `3.8+` and PyTorch `1.9+`. You can easily install the package from PyPI:

```bash
pip install leaf-net
```

*For developers, you can clone the repository and install it in editable mode:*
```bash
git clone https://github.com/kingknight07/Leaf-net.git
cd Leaf-net
pip install -e .
```

## 🚀 Quick Start

### 1. Initialize the Model

You can dynamically instantiate Leaf-Net (EcoConvNet) for any image resolution, patch size, and number of classes.

```python
import torch
from leaf_net import EcoConvNet

# Example: 3x128x128 input images, 5 target classes
model = EcoConvNet(
    img_size=(128, 128),
    patch_size=(4, 4),
    in_channels=3,
    num_classes=5
)

# Inference with dummy data
dummy_input = torch.randn(1, 3, 128, 128)
output = model(dummy_input)

print(f"Output shape: {output.shape}") 
# Expected Output: torch.Size([1, 5])
```

### 2. Training on Custom Datasets

Leaf-Net seamlessly integrates with standard PyTorch training loops. Below is a rapid setup guide for training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from leaf_net import EcoConvNet, train_model

# 1. Prepare Data Loaders (e.g., using torchvision.datasets)
# train_loader = ... 

# 2. Setup Device, Model, Optimizer, and Loss Criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EcoConvNet(img_size=(224, 224), num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 3. Launch Training
# history = train_model(model, device, train_loader, optimizer, criterion, epochs=10)
```

### 3. Pre-Trained Models & Hugging Face Hub

Leaf-Net is tightly integrated with the Hugging Face ecosystem. While robust domain-specific models (e.g., medical imaging, satellite data) are actively training, you will soon be able to load state-of-the-art checkpoints seamlessly:

```python
from leaf_net import EcoConvNet

# (Coming Soon) Load a model pre-trained on ImageNet or CIFAR
# model = EcoConvNet.from_pretrained("kingknight07/leaf-net-imagenet")
```

Push your fine-tuned green models to the Hub to accelerate community research:
```python
# Save and push to your Hugging Face account
# model.push_to_hub("your-username/leaf-net-custom")
```

## 👥 Authors & Contributors

This project is authored and maintained by:

- **[Ayush Shukla](https://github.com/kingknight07)** (@kingknight07)
- **Vijay Dwivedi**
- **Sulabh Sachan**
- **Iwona Grobelna**
- **Praveen Pratap Singh**

We welcome contributions! Please feel free to open an issue or submit a pull request if you want to help improve Leaf-Net.

## 📄 Citation

If you use Leaf-Net (EcoConvNet) in your research, we would greatly appreciate it if you cite our book chapter:

```bibtex
@incollection{shukla2025ecoconvnet,
  title={Beyond Self-Attention: Designing Lightweight Transformer-Like Models with 1D-Convolutions for Green AI},
  author={Shukla, Ayush and Dwivedi, Vijay and Sachan, Sulabh and Grobelna, Iwona and Singh, Praveen Pratap},
  year={2025},
  note={Book Chapter Proposal}
}
```

## ⚖️ License

This project is open-sourced and released under the [Apache License 2.0](LICENSE). See the `LICENSE` file for more details.

---
<div align="center">
  <p>Built with ❤️ for Green AI.</p>
</div>
