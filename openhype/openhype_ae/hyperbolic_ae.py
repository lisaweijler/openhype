from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import List
import torch
from torch import nn

from openhype.utils import lorentz as L


@dataclass
class HyperEmbedderConfig:

    act_fn: str = "gelu"
    enc_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    dec_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    input_dim: int = 512  # 768  # clip embedding dim
    curv_init: float = 1.0
    contrastive_weight: float = 1.0
    temperature: float = 0.2
    normalize_input_feats: bool = False
    normalize_output_feats: bool = False
    reconstruction_weight: float = 1.0

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


class HyperEmbedder(nn.Module):
    """
    Encode CLIP mask features and their hierarchies in
    hyperbolic space:

    1. Use a few layer MLP to reduce dimensionality and creat new embedding vectors.

    2. Lift embeddings from encoders onto the Lorentz hyperboloid using
       exponential map operator.

    3. Compute hierarchical loss (conrastive loss between parents and children in hyperbolic space).
    4. Decode hyperbolic embeddings back to Euclidean space using
       logarithmic map operator and a few layer MLP decoder.
    5. Compute reconstruction loss between decoded embeddings and original CLIP features.

    """

    def __init__(self, config: HyperEmbedderConfig):

        super().__init__()
        self.config = config

        if config.act_fn == "gelu":
            act_fn = nn.GELU()

        # create encoder MLP
        encoder_layers = []
        for i in range(len(config.enc_hidden_dims)):
            if i == 0:
                encoder_layers.append(
                    nn.Linear(config.input_dim, config.enc_hidden_dims[i])
                )
            else:

                encoder_layers.append(act_fn)
                encoder_layers.append(
                    nn.Linear(config.enc_hidden_dims[i - 1], config.enc_hidden_dims[i])
                )
        self.encoder = nn.Sequential(*encoder_layers)

        # create decoder MLP
        decoder_layers = []
        for i in range(len(config.dec_hidden_dims) - 1):
            decoder_layers.append(
                nn.Linear(config.dec_hidden_dims[i], config.dec_hidden_dims[i + 1])
            )

            decoder_layers.append(act_fn)
        decoder_layers.append(nn.Linear(config.dec_hidden_dims[-1], config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = torch.tensor(config.curv_init).log()

        self.contrastive_weight = config.contrastive_weight
        self.reconstruction_weight = config.reconstruction_weight

        self.temperature = config.temperature

    def encode_features(self, clip_features: torch.Tensor, project: bool):
        """
        Args:
            clip_features: Batch of clip features of masks of images, (B,D).
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        if self.config.normalize_input_feats:
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        embedded_feats = self.encoder(clip_features)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            with torch.autocast(embedded_feats.device.type, dtype=torch.float64):
                embedded_feats = L.exp_map0(embedded_feats, self.curv.exp())

        return embedded_feats

    def decode_features(self, embedded_features: torch.Tensor, project: bool):
        """
        Args:
            embedded_features: Batch of hyperbolic features of masks of images, (B,D_latent).
            project: Project features from the Hyperboloid back to the tangent space at origin.
        Returns:
            Batch of decoded features of shape `(B, D)`.
        """

        # These features are space components of embeddings on the hyperboloid. (Hyperbolic features)
        # Apply projection to go back to tangent space at origin (Euclidean).
        if project:
            with torch.autocast(embedded_features.device.type, dtype=torch.float64):
                eucl_embedded_feats = L.log_map0(embedded_features, self.curv.exp())

        decoded_features = self.decoder(eucl_embedded_feats)

        if self.config.normalize_output_feats:
            decoded_features = decoded_features / decoded_features.norm(
                dim=-1, keepdim=True
            )

        return decoded_features

    def forward(
        self,
        clip_features: torch.Tensor,
        global_parent_id: torch.Tensor,
        has_parent: torch.Tensor,
        keep_for_negs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            clip_features: Batch of clip features of masks of images, (B,D). B= n_masks_per_image * n_images_in_batch
            global_parent_id: for each mask that has a parent, what is the index of the parent in the batch, (n_masks_with_parent_in_batch,)
            has_parent: boolean mask which masks have a parent in the batch
            keep_for_negs: boolean mask which masks to keep for negatives (not parents or children)
        Returns:
            A dict with the following entries:
                - loss: The computed loss.
                - logging: A dict of logging variables.
        """

        _curv = self.curv.exp()

        embedded_feats = self.encode_features(clip_features, project=True)
        reconstructed_feats = self.decode_features(embedded_feats, project=True)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(embedded_feats.device.type, dtype=torch.float64):

            # set same mask to -1, we dont want to include this in the calculation
            targets = (
                torch.eye(embedded_feats.shape[0], device=embedded_feats.device) * -1
            )
            targets = targets[has_parent, :]

            # set those to -1 that we want to exclude from negatives
            targets[~keep_for_negs[has_parent, :]] = (
                -1
            )  # n_masks x n_masks, 1 if pos 0 for negs -1 if excluded from negs (e.g. if it is child or parent or in different image)
            targets[torch.arange(has_parent.sum()), global_parent_id] = (
                1  # set positives
            )

            # ---------------- angluar logits
            # want to minimize alpha -> maximize -alpha
            all_logits_alpha_parentview = -L.pairwise_oxy_angle(
                embedded_feats, embedded_feats[has_parent], _curv, eps=1e-8
            ).T
            # want to maximize beta
            all_logits_beta_parentview = (
                torch.pi + all_logits_alpha_parentview
            )  # b= pi-alpha, here plus since alpha is already multiplied with -1

            # ---------------- geodesic distance logits
            # we have different amount of negatives for each mask
            all_logits_dist = -L.pairwise_dist(
                embedded_feats[has_parent], embedded_feats, _curv, eps=1e-8
            )

            # ---------------- geodesic distance contrastive loss
            all_logits = torch.exp(all_logits_dist / self.temperature).clone()
            all_logits[targets == -1] = 0
            loss_denom = all_logits.sum(dim=-1)
            loss_nom = all_logits[
                targets == 1
            ]  # should be exactly one per row, the positive pair
            loss_logits = torch.div(loss_nom, loss_denom)
            contrastive_loss_dist = -torch.log(loss_logits).mean()

            # ---------------- angular contrastive loss - parent view
            all_logits = torch.exp(
                all_logits_alpha_parentview / self.temperature
            ).clone()
            all_logits[targets == -1] = 0
            loss_denom = all_logits.sum(dim=-1)
            loss_nom = all_logits[
                targets == 1
            ]  # should be exactly one per row, the positive pair
            loss_logits = torch.div(loss_nom, loss_denom)
            contrastive_loss_alpha_parentview = -torch.log(loss_logits).mean()

            all_logits = torch.exp(
                all_logits_beta_parentview / self.temperature
            ).clone()
            all_logits[targets == -1] = 0
            loss_denom = all_logits.sum(dim=-1)
            loss_nom = all_logits[
                targets == 1
            ]  # should be exactly one per row, the positive pair
            loss_logits = torch.div(loss_nom, loss_denom)
            contrastive_loss_beta_parentview = -torch.log(loss_logits).mean()

            # ----------------- combine contrastive losses
            contrastive_loss = (
                contrastive_loss_dist
                + (contrastive_loss_beta_parentview + contrastive_loss_alpha_parentview)
                / 2.0
            )

            # ----------------- mse reconstruction loss
            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstructed_feats, clip_features
            )

            # ----------------- final loss
            loss = (
                self.reconstruction_weight * reconstruction_loss
                + self.contrastive_weight * contrastive_loss
            )

        return {
            "loss": loss,
            "logging": {
                "reconstruction_loss": reconstruction_loss,
                "contrastive_loss_dist": contrastive_loss_dist,
                "contrastive_loss_alpha_parentview": contrastive_loss_alpha_parentview,
                "contrastive_loss_beta_parentview": contrastive_loss_beta_parentview,
                "contrastive_loss": contrastive_loss,
                "curv": _curv,
            },
        }
