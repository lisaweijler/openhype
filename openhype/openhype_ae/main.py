from datetime import datetime
from dacite import from_dict
from pathlib import Path

from openhype.openhype_ae.embed_feats import embed_feats
from openhype.openhype_ae.train import HyperEmbTrainConfig, train
from openhype.utils.io import convert_dictconf_to_dict
from openhype.task_registry import register_task
from openhype.variable_registry import GlobalVars


@register_task("openhype_ae")
def main(configs, img_dir, global_vars: GlobalVars, output_dpath):
    # could initialise wandb logger here, left as is for now
    configs = convert_dictconf_to_dict(configs)
    configs = from_dict(data_class=HyperEmbTrainConfig, data=configs)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        Path(output_dpath)
        / Path(
            global_vars.mask_feature_dir
        ).stem  # Vit-B-16_laion... Clip model specification
        / configs.experiment_name  # / timestamp
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpts"
    Path(ckpt_dir).mkdir(exist_ok=True)
    global_vars.ae_ckpt_path = ckpt_dir / "model_best.pth"

    train(
        configs,
        img_dir,
        output_dir,
        ckpt_dir,
        global_vars.mask_feature_dir,
        global_vars.mask_hierarchy_dir,
        global_vars.mask_feature_suffix,
        global_vars.mask_hierarchy_suffix,
    )

    # save mask features embeddign in hyperb. space if specified in config
    if configs.cache_feature_embeds:
        embeds_dir = output_dir / "feat_embeds"
        latent_eucl_feature_suffix = "_f.npy"
        latent_eucl_feat_mask_id_suffix = "_m_id.npy"
        Path(embeds_dir).mkdir(exist_ok=True)
        embed_feats(
            configs.model_config,
            img_dir,
            embeds_dir,
            ckpt_dir,
            global_vars.mask_feature_dir,
            global_vars.mask_hierarchy_dir,
            global_vars.mask_feature_suffix,
            global_vars.mask_hierarchy_suffix,
            latent_eucl_feature_suffix,
            latent_eucl_feat_mask_id_suffix,
        )
        global_vars.latent_eucl_feature_dir = Path(embeds_dir)
        global_vars.latent_eucl_feature_suffix = latent_eucl_feature_suffix
        global_vars.latent_eucl_feat_mask_id_suffix = latent_eucl_feat_mask_id_suffix
