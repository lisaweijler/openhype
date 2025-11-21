from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import torch
from open_clip import get_tokenizer, create_model_and_transforms

torch.autograd.set_detect_anomaly(True)


from openhype.utils import lorentz as L
from openhype.openhype_ae.hyperbolic_ae import (
    HyperEmbedder,
    HyperEmbedderConfig,
)

MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


@dataclass
class TextEmbConfig:
    model_source: str = ""  # open_clip or clip
    model_name: str = ""  # e.g. ViT-L-14
    pretrained: str = ""  # e.g. laion2b_s32b_b82k


@dataclass
class BaseEvaluaterConfig:
    ae_ckpt_path: str = ""
    output_dir: str = ""
    gt_path: str = ""
    transforms_path: Union[str, Path] = ""
    nerf_config_path: Union[str, Path] = ""
    interpolation_steps: int = 50
    extrapolated_leafs: bool = False
    aggregation: str = "softmax_weighted"  # "max", "mean", "sum",
    text_embedder_config: TextEmbConfig = TextEmbConfig()
    hyperembedder_config: HyperEmbedderConfig = HyperEmbedderConfig()
    verbose_visualization: bool = True
    negatives: List[str] = field(
        default_factory=lambda: ["object", "things", "stuff", "texture"]
    )

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


class BaseEvaluater:
    def __init__(self, config: BaseEvaluaterConfig, device: str = "cuda:0"):
        self._ae_model = None  # hyperbolic AE model
        self._vl_model = None  # vision-language model e.g. CLIP
        self._gt_data = None
        self._vl_model_tokenizer = None
        self._output_dir_experiment = None
        self.config = config
        self.device = torch.device(device)
        self.negatives = config.negatives
        self.negative_text_embedings = None  # created in call funciton

    @property
    def output_dir_experiment(self):
        if self._output_dir_experiment is None:

            vlf_type = f"{self.config.text_embedder_config.model_name}_{self.config.text_embedder_config.pretrained}"

            # create output dir
            self._output_dir_experiment = Path(self.config.output_dir) / vlf_type

            self._output_dir_experiment = (
                self._output_dir_experiment / f"steps_{self.config.interpolation_steps}"
            )

            self._output_dir_experiment.mkdir(exist_ok=True, parents=True)
        return self._output_dir_experiment

    @output_dir_experiment.setter
    def output_dir_experiment(self, new_output_dir_experiment):
        self._output_dir_experiment = new_output_dir_experiment

    @property
    def gt_data(self):
        if self._gt_data is None:
            self._gt_data = self._get_gt_data()
        return self._gt_data

    @property
    def ae_model(self):
        if self._ae_model is None:
            print(f"Loading hyperbolic AE model from {self.config.ae_ckpt_path}")
            checkpoint = torch.load(self.config.ae_ckpt_path, map_location="cpu")
            self._ae_model = HyperEmbedder(self.config.hyperembedder_config)
            self._ae_model.load_state_dict(checkpoint["state_dict"])
            self._ae_model.to(self.device)
            self._ae_model.eval()
        return self._ae_model

    @property
    def vl_model(self):
        if self._vl_model is None:
            print(f"Loading Vision-Language Model.")
            if self.config.text_embedder_config.model_source == "open_clip":
                # embed tokens
                pretrained = (
                    self.config.text_embedder_config.pretrained
                )  # "laion2b_s32b_b82k"  # f
                model_name = self.config.text_embedder_config.model_name  # "ViT-L-14"
                self._vl_model, _, preprocess = create_model_and_transforms(
                    model_name=model_name, pretrained=pretrained, device=self.device
                )

            else:
                raise ValueError(
                    f"Model source {self.config.text_embedder_config.model_source} not supported."
                )
        return self._vl_model

    @property
    def vl_model_tokenizer(self):
        if self._vl_model_tokenizer is None:
            if self.config.text_embedder_config.model_source == "open_clip":
                model_name = self.config.text_embedder_config.model_name  # "ViT-L-14"
                self._vl_model_tokenizer = get_tokenizer(model_name=model_name)

            else:
                raise ValueError(
                    f"Model source {self.config.text_embedder_config.model_source} not supported."
                )
        return self._vl_model_tokenizer

    @abstractmethod
    def _get_gt_data(self, *args, **kwargs):
        """Abstract method that returns the ground truth for the eval dataset.

        Args:
            TODO

        Returns:
           TODO
        """

    @abstractmethod
    def get_text_data(self, idx: int):
        """Get the text embedding and text prompts for eval frame of index idx."""

    @abstractmethod
    def get_rendered_eval_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method that returns the rbg and openhype renderings of the eval dataset.

        Args:
            update get those two from the config:
            transforms_path: path to the transforms.json file of the nerf dataset.
            nerf_config_path: path to the config.yml file of the trained nerf.

        Returns:
            torch tensors of shape n_eval x H x W x 3 and n_eval x H x W x 512 for rgb and openhype features respectively.
        """

    @abstractmethod
    def get_results_single_frame(self, pixel_text_similarity_map):
        """Abstract method, get results for single frame e.g. save in dict, acc.. iou etc.
        Then the return is then processed by get_final results
        """

    @abstractmethod
    def get_final_results(self, single_frame_results_dict):
        """Abstract method, get results for the whole experiment.
        Takes the single frame results dict as input.
        """

    @torch.no_grad()
    def get_relevancy(
        self,
        mask_pos_text_sim: torch.Tensor,
        mask_neg_text_sim: torch.Tensor,
        positive_id: int,
    ) -> torch.Tensor:
        """Get relevancy scores for masks based on positive and negative text similarities.
            Taken and adpated from LangSplat repo.
        Args:
            mask_pos_text_sim: torch.Tensor of shape n_masks x n_positive_prompts
            mask_neg_text_sim: torch.Tensor of shape n_masks x n_negative_prompts
            positive_id: int, index of the positive prompt to use for relevancy computation
        Returns:
            torch.Tensor of shape n_masks, relevancy scores
        """

        output = torch.cat([mask_pos_text_sim, mask_neg_text_sim], dim=-1)
        n_positives = mask_pos_text_sim.shape[-1]
        n_negatives = mask_neg_text_sim.shape[-1]
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., n_positives:]
        repeated_pos = positive_vals.repeat(1, n_negatives)

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(
            10 * sims, dim=-1
        )  # n_pixel x 4 (= n_neg phrases) x 2 (because of stack)
        best_id = softmax[..., 0].argmin(dim=1)  # among the negative samples
        final_sim = torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], n_negatives, 2),
        )[  # best_id[....].expnad(..) -> npixels x 4 x 2, brings best_id tensor whichhas length n_mask in same shape as sim tensor
            :, 0, 0
        ]
        return final_sim  # n_masks

    def embed_features_to_hyperbolic_latent(
        self, features: torch.Tensor, is_in_latent: bool = False
    ):
        if not is_in_latent:
            with torch.no_grad():
                embedded_features = self.ae_model.encode_features(
                    features.to(self.device), project=True
                )
            return embedded_features.detach()

        else:
            # output of nerf is already in latent space,
            # only need to put to hyperbolic space
            return L.exp_map0(features, curv=self.ae_model.config.curv_init)

    def decode_features_from_hyperbolic_latent(self, hyper_features):
        hyper_features = hyper_features.to(self.device)

        with torch.no_grad():
            decoded_hyper_features = self.ae_model.decode_features(
                hyper_features, project=True
            )

        return decoded_hyper_features.detach()

    def get_text_embeddings(self, text_prompts: List[str]):

        text_tokenized = self.vl_model_tokenizer(text_prompts).to(self.device)
        if self.config.text_embedder_config.model_source == "open_clip":

            text_embeddings = (
                self.vl_model.encode_text(text_tokenized, normalize=False)
                .detach()
                .cpu()
            )  # nprompts x embedding_dim

        else:
            raise ValueError(
                f"Model source {self.config.text_embedder_config.model_source} not supported."
            )

        return text_embeddings

    def get_mask_text_cos_sim(
        self, mask_embeddings_img, text_embeddings_img
    ):  # actually this is not the masks but the rendered features per pixel

        text_embeddings_img_norm = text_embeddings_img / torch.clamp(
            text_embeddings_img.norm(dim=-1, keepdim=True), min=1e-8
        )
        mask_norm = torch.clamp(
            mask_embeddings_img.norm(dim=-1, keepdim=True), min=1e-8
        )
        mask_embeddings_img_norm = mask_embeddings_img / mask_norm

        mask_text_sim = mask_embeddings_img_norm.cpu() @ text_embeddings_img_norm.T

        return mask_text_sim

    def get_pixel_text_interpolated_sim(
        self,
        vl_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ):
        # expects vl_features to be of shape ((h w) d)s
        # embed in AE
        hyperb_feats = self.embed_features_to_hyperbolic_latent(
            vl_features, is_in_latent=True
        )  # (H * W) x latent dim

        interpolated_hyperb_feats = L.get_interpolated_hyperbolic_features(
            hyperb_feats,
            steps=self.config.interpolation_steps,
            curv=self.config.hyperembedder_config.curv_init,
            max_dist=11.1,  # asinh(2**15) ≈ 11.09, capping of our lorentz latent space
        )  # (H * W * interpolation steps) x latent dim

        interpolated_decoded_hyperb_feats = self.decode_features_from_hyperbolic_latent(
            interpolated_hyperb_feats
        ).to(
            interpolated_hyperb_feats.device
        )  # (H * W * interpolation steps) x V-L feature dim

        # get similarities
        pixel_text_interpolated_sim = self.get_mask_text_cos_sim(
            interpolated_decoded_hyperb_feats, text_embeddings
        )  ## (H * W * interpolation steps) x n_prompts

        return pixel_text_interpolated_sim

    def get_similarity_map(
        self,
        pixel_text_similarity_field: torch.Tensor,  # h w steps n_prompts
        aggregation_method: str,  # mean, max, sum, mean_std, max_std
        verbose_visualization: bool = False,
        text_prompts: List[str] = None,  # only used when verbose_visualization is True
        eval_frame_name: str = None,  # only used when verbose_visualization is True
    ):
        """Function that creates the final similarity map, going from sim field with interpoalted features to a 1D map, one value per pixel per prompt.

        Args:
            pixel_text_similarity_field: similarities field of text prompt and rendered v-l features H x W x n_interpolation_steps x n_prompts,
                                         it contains nan values if the feature for the smallest scale of a pixel is at a bigger scale than the smallest feature scale in the map.

        Returns:
            H x W x n_prompts
        """

        extrapolated_mask = torch.isnan(pixel_text_similarity_field)

        if aggregation_method == "mean":
            pixel_sim = pixel_text_similarity_field
            pixel_sim[extrapolated_mask] = 0.0
            divider = (~extrapolated_mask).sum(dim=-2)
            pixel_sim_mean = pixel_sim.sum(dim=-2)
            pixel_sim_mean[torch.where(divider != 0)] = torch.div(
                pixel_sim_mean[torch.where(divider != 0)],
                divider[torch.where(divider != 0)],
            )

            pixel_sim_aggregated = pixel_sim_mean

        elif aggregation_method == "max":
            pixel_sim = pixel_text_similarity_field

            pixel_sim[extrapolated_mask] = -10  # something smaller than -1
            pixel_sim_max = pixel_sim.max(dim=-2)[0]
            pixel_sim_aggregated = pixel_sim_max

        elif aggregation_method == "sum":

            pixel_sim = pixel_text_similarity_field

            pixel_sim[extrapolated_mask] = 0.0
            pixel_sim_aggregated = pixel_sim.sum(dim=-2)

        elif aggregation_method == "softmax_weighted":

            pixel_sim = pixel_text_similarity_field
            pixel_sim[extrapolated_mask] = float("-inf")
            softmax_weights = torch.nn.functional.softmax(pixel_sim, dim=2)

            pixel_sim[extrapolated_mask] = 0.0
            weighted_sum = (softmax_weights * pixel_sim).sum(
                dim=2
            )  # shape: [H, W, n_prompts]
            pixel_sim_aggregated = weighted_sum

        else:
            raise ValueError(f"Aggregation mode {aggregation_method} not supported.")

        # H x W x n_prompts -> 1 x n_prompts x H x W
        similarity_map = pixel_sim_aggregated.permute(2, 0, 1).unsqueeze(0)

        # do vis if true visualize all options of aggregations and the interpolation similarities -develop mode
        if verbose_visualization:
            assert (
                text_prompts is not None and eval_frame_name is not None
            ), "'verbose_visualization' is set to True but no text_prompts and eval_frame_name is given for creating similarity heatmap plots."
            pixel_sim_vis = torch.cat(
                [
                    pixel_sim_aggregated.unsqueeze(2),
                    pixel_text_similarity_field,
                ],
                dim=2,
            )
            # H x W x interp steps x num_prompts -> interp_step x num_prompts x H x W
            pixel_sim_vis = pixel_sim_vis.permute(2, 3, 0, 1)
            self.create_plots(pixel_sim_vis, text_prompts, eval_frame_name)

        return similarity_map

    def create_plots(self, valid_map, prompts, image_name: Path = None):

        n_head, n_prompt, h, w = valid_map.shape  # n_head:

        for k in range(n_prompt):
            for i in range(n_head):

                fig = plt.figure(figsize=(30, 10))
                ax1 = fig.add_subplot(111)
                if i == 0:

                    vmax_val = 0.8
                    vmin_val = 0.2
                else:
                    vmax_val = 0.6
                    vmin_val = 0.2
                i1 = ax1.imshow(
                    valid_map[i][k].cpu().numpy(),
                    vmin=vmin_val,
                    vmax=vmax_val,
                    cmap="plasma",
                )

                ax1.title.set_text("Similarity")
                fig.colorbar(i1, ax=ax1, shrink=0.3)
                plt.tight_layout()
                if i == 0:
                    img_save_path = (
                        self.output_dir_experiment
                        / image_name
                        / "vis"
                        / f"{prompts[k]}_aggregated"
                    )

                else:
                    img_save_path = (
                        self.output_dir_experiment
                        / image_name
                        / "vis"
                        / f"{prompts[k]}_level_{i}"
                    )
                img_save_path.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(str(img_save_path) + ".png")
                plt.clf()  # Clear figure
                plt.close()  # Close a figure window

    @abstractmethod
    def __call__(self, *args, **kwds):
        """Abstract method that runs the evaluation pipeline."""
