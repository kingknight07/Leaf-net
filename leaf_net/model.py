import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class WeightedPooling(nn.Module):
    """
    Positional-Biased Attention Pooling mechanism.
    Integrates content-based feature importance with a learnable spatial bias.
    """
    def __init__(self, input_dim: int, alpha: float = 1.0):
        super().__init__()
        self.attention_scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), 
            nn.Tanh(), 
            nn.Linear(input_dim // 2, 1)
        )
        self.alpha = alpha

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        device = h.device
        # Content-based scores
        content_scores = self.attention_scorer(h).squeeze(-1)
        
        # Positional bias
        pos_indices = torch.arange(T, device=device, dtype=torch.float32)
        normalized_pos = pos_indices / (T - 1) if T > 1 else torch.zeros_like(pos_indices)
        positional_bias = self.alpha * normalized_pos
        
        # Combined scores
        combined_scores = content_scores + positional_bias
        weights = torch.softmax(combined_scores, dim=1)
        
        # Weighted pooling
        pooled_vector = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return pooled_vector

class EcoConvNet(
    nn.Module, 
    PyTorchModelHubMixin, 
    library_name="leaf-net", 
    repo_url="https://github.com/your-username/EcoConvNet"
):
    """
    EcoConvNet: A lightweight, Transformer-like architecture built on 1D-Convolutions 
    for energy-efficient computer vision (Green AI).
    
    Inherits from PyTorchModelHubMixin to easily push/pull pre-trained models to/from Hugging Face Hub.
    """
    def __init__(
        self, 
        img_size=(32, 32), 
        patch_size=(1, 1), 
        in_channels=3, 
        cnn_out_channels=32,
        embed_dim=64, 
        num_classes=10, 
        dropout=0.3, 
        pos_bias_alpha=1.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cnn_out_channels = cnn_out_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.pos_bias_alpha = pos_bias_alpha

        # 1. Lightweight CNN Backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, cnn_out_channels, 3, 1, 1), 
            nn.BatchNorm2d(cnn_out_channels), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
        )
        
        # Auto-calculate feature map size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            feature_map = self.cnn_backbone(dummy_input)
            _, C, H_f, W_f = feature_map.shape
        
        ph, pw = self.patch_size
        self.num_patches = (H_f // ph) * (W_f // pw)
        patch_dim = C * ph * pw
        
        # 2. Patch Projection & Positional Encoding
        self.patch_projection = nn.Linear(patch_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # 3. 1D-Convolutional Sequence Processor (replaces Self-Attention)
        self.sequence_processor = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), 
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), 
            nn.BatchNorm1d(embed_dim), 
            nn.ReLU(),
        )
        
        # 4. Positional-Biased Attention Pooling
        self.pooling = WeightedPooling(input_dim=embed_dim, alpha=pos_bias_alpha)
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # 1. CNN Feature Extraction
        features = self.cnn_backbone(x)
        
        # 2. Patchify
        _, C, H, W = features.shape
        ph, pw = self.patch_size
        
        # Rearrange to patches: B, C, H, W -> B, (H//ph)*(W//pw), C*ph*pw
        patches = features.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(B, C, -1, ph, pw)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, patches.shape[1], -1)
        
        # 3. Project & Add Positional Encoding
        projected_patches = self.patch_projection(patches)
        embeddings = projected_patches + self.positional_encoding
        
        # 4. Sequence Processing (Conv1D expects B, C, L)
        embeddings_t = embeddings.transpose(1, 2)
        processed_sequence = self.sequence_processor(embeddings_t)
        processed_sequence = processed_sequence.transpose(1, 2)
        
        # 5. Pooling & Classification
        pooled_representation = self.pooling(processed_sequence)
        logits = self.classifier(pooled_representation)
        return logits
