# OpenHype Search3D ScanNet++ benchmark data: open-vocabulary object and part annotations on a subset of the ScanNet++ dataset adapted for radiance fields

We provide the OpenHype benchmark consisting of object and part annotations on 20 scenes of the ScanNet++ dataset for open-vocabulary segmentations and localization from radiance fields. These annotations are created using the ScanNet++ subset of the [Search3D benchmark](https://github.com/aycatakmaz/search3d/tree/main/search3d/benchmark). In particular, we projected the 3D annotations of Search3D to test frames of the novel view synthesis benchmark of Scannet++. We include 30 different object and 33 object-part annotations across 20 ScanNet++ scenes along with open-vocabulary prompts. The part prompts were created by combining the Search3D labels of the object with the label of the object part, resulting in prompts with the following form: "[PART] of a [OBJ]". 


## Download  
We use the undistorted DSLR images of ScanNet++. Please download the DSLR assests (and scans assests to get the mesh_aligned_0.05.ply for completeness) for the 20 scenes specified [here](scene_ids.txt) via the official [ScanNet++ webpage](https://scannetpp.mlsg.cit.tum.de/scannetpp/). Then download and unzip the annotations form [here](https://drive.google.com/file/d/14XGCM20-w1-2fae1saDo05nfToeY8sAs/view?usp=sharing). 

## Processing
Once the scenes and annotations are downloaded you can use the [scannetpp_preprocess.py](scannetpp_preprocess.py) script to subsample and transform the images into nerfstudio format. 





## Licensing

This dataset is distributed under the original ScanNet++ dataset license. Users must adhere to all terms and conditions outlined in the [ScanNet++ License Information](https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf). Please ensure compliance with these rules when using the data. Access to the ScanNet++ scans are obtained by following the official instructions provided by the original dataset.


## Citation

If you use this benchmark in your research, please consider citing our paper:

```bibtex
@article{weijler2025openhype,
  title = {OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields},
  author = {Weijler, L. and Koch, S. and Poiesi, F. and Ropinski, T. and Hermosilla, P.},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2025},
}
```

the Search3D benchmark paper:

```bibtex
@article{takmaz2025search3d,
  title={{Search3D: Hierarchical Open-Vocabulary 3D Segmentation}},
  author={Takmaz, Ayca and Delitzas, Alexandros and Sumner, Robert W. and Engelmann, Francis and Wald, Johanna and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```

and the original ScanNet++ dataset:

```bibtex
@inproceedings{yeshwanth2023scannet++,
  title={Scannet++: A high-fidelity dataset of 3d indoor scenes},
  author={Yeshwanth, Chandan and Liu, Yueh-Cheng and Nie{\ss}ner, Matthias and Dai, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```