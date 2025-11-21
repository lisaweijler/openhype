# 🔥OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields

[Lisa Weijler](https://lisaweijler.github.io/), [Sebastian Koch](https://kochsebastian.com/), [Fabio Poiesi](https://fabiopoiesi.github.io/), [Timo Ropinski](https://scholar.google.com/citations?user=FuY-lbcAAAAJ), [Pedro Hermosilla](https://phermosilla.github.io/)

**NeurIPS 2025** | [[Paper](https://openreview.net/pdf?id=zQmXDUbZ5D)] [[Project Page](https://lisaweijler.github.io/openhype-projectpage/)] [[Poster](https://lisaweijler.github.io/openhype-projectpage/static/images/openhype_poster.pdf)]
---
OpenHype represents scene hierarchies using hyperbolic geometry, enabling continuous hierarchical open-vocabulary 3D scene representations.
![Teaser Figure](media/teaser.png)
## 📦 Installation

We recommend using a conda env using [miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) as well as [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for a faster conda solver.

#### 1. configure libmamba (optional):
```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

#### 2. create conda env:
```
conda create --name openhype python=3.10.12
conda activate openhype
python -m pip install --upgrade pip
```

#### 3. install dependencies and openhype:
```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
git clone git@github.com:lisaweijler/openhype.git
cd openhype 
python -m pip install -e .
```
If the tiny-cuda installation fails as it cannot find torch or another package try with the `--no-build-isolation` flag.

### Semantic-SAM:
Since we use [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) for the extraction of masks, we recommend creating a seperate environment to avoid dependency issues. You can follow the instructions given in the original repository or the installation steps below, as long as you have a working environment with semantic-sam installed. 

#### 1. create conda env:
First, make sure to navigate out of the openhype folder and deactivate openhype env. 
```
cd ..
conda deactivate openhype
```
Then run:
```
conda create --name semanticsam python=3.10.12
conda activate semanticsam
python -m pip install --upgrade pip
```

#### 2. install dependencies and semantic-sam:
Download a checkpoint from the Semantic-SAM Model Zoo, in this paper we used the one with the SwinL backbone: [download model](https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth). Make sure to update the path in the train config template to your model path. 
```
conda install -c conda-forge cudatoolkit-dev=11.7
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
git clone git@github.com:UX-Decoder/Semantic-SAM.git
cd Semantic-SAM
python -m pip install -e .
pip install tyro==0.9.35
```
Again if the detectron2-xyz installation fails due to not finding packages try with the `--no-build-isolation` flag.

## 📊 Data
Please see the [benchmark docs](benchmark/README.md) for data downloading and processing.

## 🗂️ File Structure
The OpenHype pipeline can be seperated in 3 steps:
1. Preprocessing: 
    - extracting masks, 
    - constructing mask hierarchies
    - and extracting vision-language (VL) features for each mask crop.
2. Hyperbolic AE training:
    - embedding the VL features in a lower-dim. hyperbolic latent space,
    - so that the embedding reflects the mask hierarchies extracted in step 1.
3. NeRF training:
    - training a NeRF with an additional VL field that is supervised with hyperbolic latent embeddings.

The code is setup so that every step is executed automatically, where the folder structure is automatically created and intermediate products such as extracted VL features are automatically saved. If the Preprocessing has already be done for a scene this step is skipped automatically. 

The specifications for each step are given in the config file, as an example you can look at the given [config template](configs/config_template_train.yaml). 

Executing the pipeline with the [main_pipeline.py](main_pipeline.py) script conducts the openhype training for one scene and will produce the following folder structure:

```
output_dpath/
    └──scene_id/ 
        ├── 1_openhype_preprocess/
        |   ├── clip_crop_features_semantic_sam/
        |   |   └── ViT-B-16_laion2b_s34b_b88k/ # specifying the VL model used
        |   ├── mask_hierarchies_semantic_sam/
        |   └── masks_semantic_sam/
        ├── 2_openhype_ae/
        │   └── ViT-B-16_laion2b_s34b_b88k/ # specifying the VL moodel used 
        |       └── experiment-name/ # eg. scannetpp
        |           ├── ckpts/ # saved ckpts including *model_best.pth*
        |           ├── feat_embeds/ # mask feature embeddings in latent space
        |           └── config.json
        └── 3_openhype_nerf/ # substructure given due to nerstudio
            └── experiment-name/ # eg. scannetpp
                └──openhype/
                    └──run0/
                        ├── nerfstudio_models/
                        ├── wandb/
                        ├── dataparser_transforms.json
                        └── config.yml
```

## ⚡ Running OpenHype
We provide template configs and bash files for single scene as well as multiple scene training and evaluation. 
Please note that we used weights & biases for the AE training, yet is not necessar to do so. If you want to use it please set your specifications in the `WandBConfig` [here](openhype/utils/wandb_logger.py). Otherwise remove all WandB logging form the AE training. 

### Single scene
To train OpenHype for a single scene you can use the following commands. Please specify your paths in the train config first as indicated in the [template file](configs/single_scene/config_template_train.yaml).

#### Training

```
python main_pipeline.py --output_dpath openhype_output/scannetpp/scene_ID --config_path configs/config_train_sceneID.yaml
```

#### Testing

```
python eval.py --config_path configs/config_test_sceneID.yaml --evaluator scannetpp_evaluator
```

### Multiple scenes - batch experiment
If you want to get results for multiple scenes and multiple runs at once you can use as an example the bash file [exp5_scenes.sh](exp_5_scenes.sh). This bash files produces results over 5 runs for the 5 scenes used in our ablations. 
The final results are saved in the folder `output_directory/eval_aggregated_output/experiment_name` as a dictionary in `*.json` format: `all_runs_aggregated_final_values.json`.
The structure of the output dictionary is given below. For example if you want to know the average IoU for parts prompts over all scenes over all runs, you would have to look in `all_scenes->iou->parts->value`.

```
{
  "ViT-B-16_laion2b_s34b_b88k_steps_20": { # exp. spec., VL model and interp. steps used
    "all_scenes": {
        "acc":{ 
            "all":{...}
            "objects":{...}
            "parts":{...}
        },
        "iou":{
            "all":{...}
            "objects":{...}
            "parts":{...}
        }
    }
    "scene_id": {...}
    ... # results for single scenes    
  }
}
```

#### Run training, testing and final results aggregation 

```
chmod +x exp_5_scenes.sh
./exp_5_scenes.sh
```

Optionally if several GPUs are available one can use the [gpu_schedular.py](gpu_schedular.py) script, which distributes the commands across GPUs in chunks (evaluation only starts when all training processes are finished). So the models for different scenes are trained in parallell on different GPUs. You can specify the GPU IDs at the top of the script as well as the min amount of memory in Mb, so a GPU is considered free. 


## 🙏 Acknowledgement
Parts of the code base are inspired and build on top of [OpenNerf](https://github.com/opennerf/opennerf), [LangSplat](https://github.com/minghanqin/LangSplat), [MERU](https://github.com/facebookresearch/meru), [GARField](https://github.com/chungmin99/garfield), [RelationField](https://github.com/boschresearch/relationfield).

## 📚 BibTeX
If you find our code or paper useful, please cite:
```@article{weijler2025openhype,
  title = {OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields},
  author = {Weijler, L. and Koch, S. and Poiesi, F. and Ropinski, T. and Hermosilla, P.},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2025},
}
```
