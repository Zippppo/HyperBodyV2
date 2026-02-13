from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .label_embedding import LorentzLabelEmbedding
from .projection_head import LorentzProjectionHead


class HyperBodyNet(nn.Module):
    """Wrap a nnU-Net backbone with hyperbolic projection/label embedding branches."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 70,
        embed_dim: Optional[int] = None,
        curv: float = 1.0,
        class_depths: Optional[Dict[int, int]] = None,
        min_radius: float = 0.1,
        max_radius: float = 2.0,
        direction_mode: str = "random",
        text_embedding_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hyperbolic_mode = True

        in_channels = self.backbone.decoder.stages[-1].output_channels
        if embed_dim is None:
            embed_dim = in_channels

        self.hyp_head = LorentzProjectionHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            curv=curv,
        )
        self.label_emb = LorentzLabelEmbedding(
            num_classes=num_classes,
            embed_dim=embed_dim,
            curv=curv,
            class_depths=class_depths,
            min_radius=min_radius,
            max_radius=max_radius,
            direction_mode=direction_mode,
            text_embedding_path=text_embedding_path,
        )

        self._decoder_features: Optional[Tensor] = None
        self._hook = self.backbone.decoder.stages[-1].register_forward_hook(self._capture_decoder_features)

    def _capture_decoder_features(self, _module, _inputs, output: Tensor) -> None:
        self._decoder_features = output

    @property
    def decoder(self):
        return self.backbone.decoder

    @property
    def encoder(self):
        return self.backbone.encoder

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        seg_output = self.backbone(x)

        if not self.hyperbolic_mode:
            self._decoder_features = None
            return seg_output

        if self._decoder_features is None:
            raise RuntimeError("Decoder features were not captured by the forward hook.")

        voxel_emb = self.hyp_head(self._decoder_features)
        label_emb = self.label_emb()
        self._decoder_features = None
        return seg_output, voxel_emb, label_emb

    def compute_conv_feature_map_size(self, input_size):
        return self.backbone.compute_conv_feature_map_size(input_size)
