# DI2SDiff

This repo is the official implementation of  "Diverse Intra- and Inter-Domain Activity Style Fusion for Cross-Person Generalization in Activity Recognition" accepted by [KDD 2024 research track.] ([Paper](https://arxiv.org/abs/2406.04609))



# Datasets

We employ the identical preprocessing method as outlined in the work by [Qin et al DDLearn](https://github.com/microsoft/robustlearn/tree/main/ddlearn) on three publicly available HAR datasets, namely [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities), [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), and [USC-HAD](https://sipi.usc.edu/had/). By running its data preprocess code involving deal, divide_domain, and raw_aug_loader, we can obtain the dataset with the specified target and remain rate. The data file should be put in the `data` folder, such as `data/uschad/uschad_crosssubject_rawaug_rate0.2_t0_seed1_scalernorm.pkl`. Here, we only use the preprocessed original data for training.



# Prerequisites

1. See requirements.txt

```
 pip install -r requirements.txt
```

2. Prepare dataset

Please download the dataset and run the [preprocessing code](https://github.com/microsoft/robustlearn/tree/main/ddlearn). Once you have processed the data, move the resulting file to the data folder. For example, the processed file for the USC-HAD dataset with 20% of the data used for training, the target set to 1, and a seed value of 1 should be saved as `data/uschad/uschad_crosssubject_rawaug_rate0.2_t0_seed1_scalernorm.pkl`.



# How to Run
We provide the code for training the style conditioner, diffusion model, and feature network, which should be trained sequentially during each epoch. The commands for training all tasks with 20% of the USC-HAD, PAMAP, and DASADS datasets are provided below.

#### USC-HAD 20% training data

```
sh run_uschad.sh
```

#### PAMAP 20% training data

```
sh run_pamap.sh
```

#### DSADS 20% training data

```
sh run_dsads.sh
```



# Instructions

We provide a brief introduction to each code folder.

1. **Style_conditioner**: This folder is used to train the style conditioner, which is built based on [TS-TCC](https://github.com/emadeldeen24/TS-TCC/).  To adapt to our training dataset, we added new configs.
2. **Diffusion_model**: This folder is used to train our diffusion model, which is adapted from [denoising-diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch). to achieve the guidance goal, we made adjustments to the model structure and diffusion method, such as adding style embedding and multiple conditional guidance.
3. **Featurenet**: This folder is used to generate synthetic data and train the feature network for classification. 

We ran these experiments on a GeForce RTX 3090 Ti. The generation time may take some time, but we believe that existing fast diffusion models can help alleviate this problem.



# Acknowledgements

We would like to thank the pioneer  work including: [DDLearn](https://github.com/microsoft/robustlearn/tree/main/ddlearn)， [TS-TCC](https://github.com/emadeldeen24/TS-TCC/) and [denoising-diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch).



# Citation

@article{zhang2024diverse,
  title={Diverse Intra-and Inter-Domain Activity Style Fusion for Cross-Person Generalization in Activity Recognition},
  author={Zhang, Junru and Feng, Lang and Liu, Zhidan and Wu, Yuhan and He, Yang and Dong, Yabo and Xu, Duanqing},
  journal={arXiv preprint arXiv:2406.04609},
  year={2024}
}

# Contact

junruzhang@zju.edu.cn
