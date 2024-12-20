# HC-LLM: Historical-Constrained Large Language Models for Radiology Report Generation (AAAI 2025)

## Introduction
Radiology report generation (RRG) models are designed to describe individual examinations, but often neglect the integration of historical information from either visual or textual modality. However, incorporating longitudinal data is crucial for the clinical application of medical reports, particularly for patient follow-ups and reviews. Existing methods often struggle with long sequence dependencies when integrating historical information, whereas large language models (LLMs) possess strong in-context learning capabilities, making them promising for analyzing longitudinal medical data. In light of this, we introduce a novel Historical-Constrained Large Language Models (HC-LLM) framework for RRG, which empowers LLMs with longitudinal report generation capabilities by constraining the consistency and differences between longitudinal images and their corresponding reports. 

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
cd HC-LLM
pip install -r requirements.txt
```


**2. Prepare the training dataset**

Longitudinal-MIMIC: you can download this dataset from [here](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

MS-CXR-T: download the dataset from [here](https://physionet.org/content/ms-cxr-t/1.0.0/) and process this dataset according to the MS-CXR-T-processor.py to get the data we used.

After downloading the data, place it in the ./data folder.

### Training

```bash
bash scripts/1-1.shallow_run.sh
```

### Testing (For MIMIC-CXR)

```bash
bash scripts/1-2.shallow_test.sh
```

## Acknowledgement

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) Some codes of this repo are based on R2GenGPT.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).

## Citation
If you find this code is useful, please cite us.

```bash
@article{liu2024hc,
  title={HC-LLM: Historical-Constrained Large Language Models for Radiology Report Generation},
  author={Liu, Tengfei and Wang, Jiapu and Hu, Yongli and Li, Mingjie and Yi, Junfei and Chang, Xiaojun and Gao, Junbin and Yin, Baocai},
  journal={arXiv preprint arXiv:2412.11070},
  year={2024}
}
```
