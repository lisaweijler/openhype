import json
import logging
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def load_config(config_path):
    logger.info(f"load config file from {config_path}")
    config = OmegaConf.load(config_path)
    return config


def convert_dictconf_to_dict(config):
    return OmegaConf.to_container(config)


def save_config(config, save_path):
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    with open(save_path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_dict_as_json(dict_to_save, save_path):
    with open(save_path, "w") as f:
        data = json.dump(dict_to_save, f)
    return data
