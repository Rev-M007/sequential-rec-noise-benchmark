"""
fmlp_rec.py
===========

FMLP-Rec ("Filter-enhanced MLP is All You Need for Sequential Recommendation",
Zhou et al., WWW 2022) as a custom RecBole model.

FMLP-Rec replaces self-attention with learnable filters in the frequency
domain. Each block:

    1. FFT along the sequence axis       X_f = rfft(X)
    2. learnable complex filter          X_f' = W ⊙ X_f      (one weight per frequency × channel)
    3. inverse FFT                       X'   = irfft(X_f')
    4. dropout + residual + LayerNorm
    5. point-wise feed-forward network, dropout + residual + LayerNorm

Because the filter weights are learned, the model can attenuate frequency
components that behave like noise, which is why it is a natural baseline for
noise-robustness experiments.

Differences from the authors' code
----------------------------------
* Loss: cross-entropy over all items (``loss_type: CE``) by default, matching
  every other model in this benchmark. The original used binary cross-entropy
  with one sampled negative; ``loss_type: BPR`` is available as an alternative.
* Padding: RecBole pads sequences on the right, the original pads on the left.
  ``forward`` converts to left padding so the most recent item always sits at
  the last position, exactly as in the original model.

Hyperparameters live in ``props/model/FMLPRec.yaml`` (RecBole-style), loaded by
``train_model.py`` through ``Config(config_file_list=...)``.

Usage
-----
    python train_model.py --model FMLPRec --dataset ml-1m
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

# Tiny fallbacks only if a key is missing from YAML / Config (should not happen
# when props/model/FMLPRec.yaml is loaded).
_FALLBACKS = {
    "n_layers": 2,
    "hidden_size": 64,
    "inner_size": 256,
    "hidden_dropout_prob": 0.5,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,
    "loss_type": "CE",
}

ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class FilterLayer(nn.Module):
    """Learnable global filter in the frequency domain.

    For a length-L sequence, ``rfft`` yields L // 2 + 1 frequency components.
    Each component and hidden channel gets its own learnable complex weight,
    stored as (real, imaginary) pairs like the original implementation.
    """

    def __init__(self, max_seq_length: int, hidden_size: int, dropout: float, eps: float):
        super().__init__()
        n_freq = max_seq_length // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(1, n_freq, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        seq_len = x.size(1)
        # FFT does not support half precision on all devices, so filter in float32.
        x_freq = torch.fft.rfft(x.float(), dim=1, norm="ortho")        # [B, L//2+1, H] complex
        x_freq = x_freq * torch.view_as_complex(self.complex_weight)    # element-wise filter
        filtered = torch.fft.irfft(x_freq, n=seq_len, dim=1, norm="ortho").to(x.dtype)
        return self.layer_norm(self.dropout(filtered) + x)              # residual + LayerNorm


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with residual connection and LayerNorm."""

    def __init__(self, hidden_size: int, inner_size: int, dropout: float, act: str, eps: float):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.act = ACTIVATIONS[act]
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dense_2(self.act(self.dense_1(x)))
        return self.layer_norm(self.dropout(hidden) + x)


class FMLPBlock(nn.Module):
    """One filter-enhanced block: frequency filter followed by the feed-forward network."""

    def __init__(self, max_seq_length, hidden_size, inner_size, dropout, act, eps):
        super().__init__()
        self.filter = FilterLayer(max_seq_length, hidden_size, dropout, eps)
        self.feed_forward = FeedForward(hidden_size, inner_size, dropout, act, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.filter(x))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FMLPRec(SequentialRecommender):
    """FMLP-Rec for RecBole (SequentialRecommender is RecBole's AbstractRecommender
    subclass for sequence models; it supplies the field names, n_items and
    max_seq_length that RecBole's sequential data loaders and trainer rely on)."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def param(key):
            value = config[key]
            return _FALLBACKS[key] if value is None else value

        self.n_layers = param("n_layers")
        self.hidden_size = param("hidden_size")
        self.inner_size = param("inner_size")
        self.hidden_dropout_prob = param("hidden_dropout_prob")
        self.hidden_act = param("hidden_act")
        self.layer_norm_eps = param("layer_norm_eps")
        self.initializer_range = param("initializer_range")
        self.loss_type = param("loss_type")

        # ---- Embedding layer ---------------------------------------------- #
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # ---- Filter-enhanced encoder -------------------------------------- #
        self.blocks = nn.ModuleList([
            FMLPBlock(self.max_seq_length, self.hidden_size, self.inner_size,
                      self.hidden_dropout_prob, self.hidden_act, self.layer_norm_eps)
            for _ in range(self.n_layers)
        ])

        # ---- Loss ---------------------------------------------------------- #
        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' is 'CE' or 'BPR'.")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Truncation-free normal init, as in the original FMLP-Rec and RecBole's SASRec."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ---- Helpers ------------------------------------------------------------ #
    @staticmethod
    def _right_to_left_padding(item_seq: torch.Tensor) -> torch.Tensor:
        """Shift each right-padded row so its items end at the last position.

        [i1, i2, i3, 0, 0] -> [0, 0, i1, i2, i3]
        RecBole sequences never contain padding (id 0) between real items, so the
        number of non-zero ids is the sequence length.
        """
        max_len = item_seq.size(1)
        shift = max_len - (item_seq != 0).sum(dim=1, keepdim=True)            # [B, 1]
        positions = torch.arange(max_len, device=item_seq.device).unsqueeze(0)  # [1, L]
        source = (positions - shift) % max_len                                 # [B, L]
        return item_seq.gather(1, source)

    # ---- RecBole interface ---------------------------------------------------- #
    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of item sequences into one user representation each.

        Args:
            item_seq: [B, L] right-padded item ids, as produced by RecBole.
        Returns:
            [B, H] representation of each sequence (output at the most recent item).
        """
        item_seq = self._right_to_left_padding(item_seq)

        position_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
        hidden = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        hidden = self.dropout(self.layer_norm(hidden))

        for block in self.blocks:
            hidden = block(hidden)

        return hidden[:, -1, :]

    def calculate_loss(self, interaction) -> torch.Tensor:
        seq_output = self.forward(interaction[self.ITEM_SEQ])
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_score = (seq_output * self.item_embedding(pos_items)).sum(dim=-1)
            neg_score = (seq_output * self.item_embedding(neg_items)).sum(dim=-1)
            return self.loss_fct(pos_score, neg_score)

        logits = seq_output @ self.item_embedding.weight.T                 # [B, n_items]
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction) -> torch.Tensor:
        """Score one candidate item per sequence (used by sampled evaluation)."""
        seq_output = self.forward(interaction[self.ITEM_SEQ])
        test_item_emb = self.item_embedding(interaction[self.ITEM_ID])
        return (seq_output * test_item_emb).sum(dim=-1)                    # [B]

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """Score every item for each sequence (used by full-ranking evaluation)."""
        seq_output = self.forward(interaction[self.ITEM_SEQ])
        return seq_output @ self.item_embedding.weight.T                   # [B, n_items]
