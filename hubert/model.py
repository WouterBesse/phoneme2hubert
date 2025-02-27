import copy
from typing import Optional, Tuple
import random

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hubert(nn.Module):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True, downsamples: int = 4):
        """
        Initializes the Hubert model.

        Args:
            num_label_embeddings (int): The number of label embeddings.
            mask (bool): Whether to apply masking during training.

        """
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu", batch_first=True
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2, bias=False) for _ in range(downsamples)
        ])

        self.maskupscale = nn.ModuleList([
            nn.ConvTranspose1d(2, 2, kernel_size=2, stride=2, bias=False) for _ in range(downsamples)
        ])

        self.downconvs = nn.ModuleList([
            nn.Conv1d(2, 2, kernel_size=2, stride=2, bias=False) for _ in range(downsamples)
        ])

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies masking to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the masked tensor and the mask.

        """
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask

    def encode(
        self, x: torch.Tensor, layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor using the Hubert model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, feature_dim).
            layer (Optional[int]): The layer index to output from the encoder. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded tensor and the mask tensor.
        """
        
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logits for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The logits tensor.
        """
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        for downconv in self.downconvs:
            x = downconv(x)
        x, mask = self.encode(x)
        x = self.proj(x)

        x = x.transpose(1, 2)
        for upconv in self.upconvs:
            x = upconv(x)
        x = x.transpose(1, 2)

        # for maskup in self.maskupscale:
        #     mask = maskup(mask)

        logits = self.logits(x)

        return logits, mask


class HubertSoft(Hubert):
    """HuBERT-Soft content encoder from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`."""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract soft speech units.

        Args:
            wav (Tensor): an audio waveform of shape (1, 1, T), where T is the number of samples.

        Returns:
            Tensor: soft speech units of shape (1, N, D), where N is the number of frames and D is the unit dimensions.
        """
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)


class HubertDiscrete(Hubert):
    """HuBERT-Discrete content encoder from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`."""

    def __init__(self, kmeans: KMeans):
        super().__init__(504)
        self.kmeans = kmeans

    @torch.inference_mode()
    def units(self, wav: torch.Tensor) -> torch.LongTensor:
        """Extract discrete speech units.

        Args:
            wav (Tensor): an audio waveform of shape (1, 1, T), where T is the number of samples.

        Returns:
            LongTensor: soft speech units of shape (N,), where N is the number of frames.
        """
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav, layer=7)
        x = self.kmeans.predict(x.squeeze().cpu().numpy())
        return torch.tensor(x, dtype=torch.long, device=wav.device)


class FeatureExtractor(nn.Module):
    def __init__(self, hidden_units: int = 512, kernel_sizes: list = [3, 3, 3, 3, 2, 2]):
        super().__init__()
        self.conv0 = nn.Conv1d(2, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_units, hidden_units, kernel_size, bias=False) for kernel_size in kernel_sizes
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x)))
        for conv in self.convs:
            x = F.gelu(conv(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        output_layer: Optional[int] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


def _compute_mask(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask
