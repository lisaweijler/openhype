from dacite import from_dict

from openhype.openhype_preprocess.preprocessor import (
    HyperPreProcessorConfig,
    HyperPreProcessor,
)
from openhype.utils.io import convert_dictconf_to_dict
from openhype.task_registry import register_task


@register_task("openhype_preprocess")
def main(configs, img_dir, global_vars, output_dpath):
    configs = convert_dictconf_to_dict(configs)
    configs = from_dict(data_class=HyperPreProcessorConfig, data=configs)
    preprocessor = HyperPreProcessor(configs, img_dir, global_vars, output_dpath)

    preprocessor.preprocess()
