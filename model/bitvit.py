import torch
import torch.nn as nn
import torch.nn.functional as F
from zeta.nn.attention import MultiheadAttention
from bitnet.bitlinear import BitLinear
from bitnet.bitffn import BitFeedForward

# [ViT ENCODER] VisionTransformer Encoder
class VisionEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.attn = MultiheadAttention(embed_size, num_heads)
        self.ffn = BitFeedForward(dim=embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.dropout(x)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = self.dropout(x)
        return x


# [ViT MODEL] BitNetViT
class BitNetViT(nn.Module):
    r"""BitNet based Vision Transformer Model

        Args:
            image_size      (int): Size of input image
            channel_size    (int): Size of the channel
            patch_size      (int): Max patch size, determines number of split images/patches and token size
            embed_size      (int): Embedding size of input
            num_heads       (int): Number of heads in Multi-Headed Attention
            classes         (int): Number of classes for classification of data
            num_layers       (int): Number of Transformer layers
            hidden_size     (int): Hidden size for Feed Forward layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, image_size, channel_size, patch_size, embed_size, num_heads,
                 classes, num_layers, hidden_size, dropout=0.1):
        super(BitNetViT, self).__init__()

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        # self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.embeddings = BitLinear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(embed_size, num_heads, hidden_size, dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            # nn.Linear(self.embed_size, self.classes)
            BitLinear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, -1, :]

        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)
        return x