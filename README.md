# Show Me What and Where has Changed? Question Answering and Grounding for Remote Sensing Change Detection

**[🏠 [Project page]](https://like413.github.io/CDQAG/)** &emsp; 
**[📄 [arXiv]](https://arxiv.org/abs/2410.23828)**  &emsp; 
**[💾 [Dataset Download]](https://drive.google.com/drive/folders/1EiOJNr8bde7apUQwqoN6cjXWKxIz7QdI?usp=sharing)**

This repository is the official implementation:
> [Show Me What and Where has Changed? Question Answering and Grounding for Remote Sensing Change Detection](https://arxiv.org/abs/2410.23828)  
> Ke Li, Fuyu Dong, Di Wang, Shaofeng Li, Quan Wang, Xinbo Gao, Tat-Seng Chua

## 📢 Update
- **(2025/6/26)** The benchmark method [VisTA](https://github.com/like413/VisTA/tree/main/VisTA) is released.
- **(2025/2/27)** Contact me for the testing set of the QAG-360K (like0413@stu.xidian.edu.cn).
- **(2024/11/8)** The first CDQAG dataset [QAG-360K](https://drive.google.com/drive/folders/1EiOJNr8bde7apUQwqoN6cjXWKxIz7QdI?usp=sharing) is released.

## Abstract
Remote sensing change detection aims to perceive changes occurring on the Earth’s surface from remote sensing data in different periods, and feed these changes back to humans. 
However, most existing methods only focus on detecting change regions, lacking the ability to interact with users to identify changes that the users expect. 
In this paper, we introduce a new task named Change Detection Question Answering and Grounding (CDQAG), which extends the traditional change detection task by providing interpretable textual answers and intuitive visual evidence. 
To this end, we construct the first CDQAG benchmark dataset, termed QAG-360K, comprising over 360K triplets of questions, textual answers, and corresponding high-quality visual masks.
It encompasses 10 essential land-cover categories and 8 comprehensive question types, which provides a large-scale and diverse dataset for remote sensing applications. 
Based on this, we present VisTA, a simple yet effective baseline method that unifies the tasks of question answering and grounding by delivering both visual and textual answers.
Our method achieves state-of-the-art results on both the classic CDVQA and the proposed CDQAG datasets. 
Extensive qualitative and quantitative experimental results provide useful insights for the development of better CDQAG models, and we hope that our work can inspire further research in this important yet underexplored direction.
<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/1.png?raw=true" width="100%" height="100%"/>
</div><br/>

## 🔥 Benchmark dataset QAG-360K
<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/2.png?raw=true" width="100%" height="100%"/>
</div><br/>

## 🌟 Simple Baseline Model VisTA
<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/3.png?raw=true" width="100%" height="100%"/>
</div><br/>

<!---
## Installation:

The code is tested under CUDA 11.8, Pytorch 1.11.0 and Detectron2 0.6.

1. Install [Detectron2](https://github.com/facebookresearch/detectron2) following the [manual](https://detectron2.readthedocs.io/en/latest/)
2. Run `sh make.sh` under `gres_model/modeling/pixel_decoder/ops`
3. Install other required packages: `pip -r requirements.txt`
4. Prepare the dataset following `datasets/DATASET.md`

## Training

```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_weights]
```

## Inference

```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```
-->

## 🌈 Results
QAG-360K
<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/table1.png?raw=true" width="100%" height="100%"/>
</div><br/>

CDVQA
<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/table2.png?raw=true" width="100%" height="100%"/>
</div><br/>


## 🙏 Acknowledgement
The dataset is based on [HiUCD](https://github.com/Daisy-7/Hi-UCD-S), [SECOND](https://captain-whu.github.io/SCD/), [LEVIR-CD](https://chenhao.in/LEVIR/), and [CDVQA](https://github.com/YZHJessica/CDVQA).
The code is based on [CRIS](https://github.com/DerrickWang005/CRIS.pytorch). We thank the authors for their open-sourced datasets and codes and encourage users to cite their works when applicable.

## 🚀 Citation
If you use our data or code in your research or find it is helpful, please cite this project.

```bibtex
@article{li2024show,
  title={Show Me What and Where has Changed? Question Answering and Grounding for Remote Sensing Change Detection},
  author={Li, Ke and Dong, Fuyu and Wang, Di and Li, Shaofeng and Wang, Quan and Gao, Xinbo and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2410.23828},
  year={2024}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
