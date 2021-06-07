# Multi-level Knowledge Distillation via Knowledge Alignment and Correlation

This repo:

**(1) covers the implementation of the following MLKD  paper:**

[Multi-level Knowledge Distillation via Knowledge Alignment and Correlation](https://arxiv.org/abs/2012.00573)

**(2) benchmarks 12 state-of-the-art knowledge distillation methods in PyTorch, including:**

(KD) - Distilling the Knowledge in a Neural Network  
(FitNet) - Fitnets: hints for thin deep nets  
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer  
(SP) - Similarity-Preserving Knowledge Distillation  
(CC) - Correlation Congruence for Knowledge Distillation  
(VID) - Variational Information Distillation for Knowledge Transfer  
(RKD) - Relational Knowledge Distillation  
(PKT) - Probabilistic Knowledge Transfer for deep representation learning  
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons  
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer  
(FSP) - A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning  
(NST) - Like what you like: knowledge distill via neuron selectivity transfer 

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
 
3. Running **MLKD** is something like:
    ```
    CUDA_VISIBLE_DEVICES=0 sh scripts/run_cifar_mlkd.sh
    CUDA_VISIBLE_DEVICES=9,10,11,12,13,14,15 sh scripts/run_imagenet_mlkd.sh
    ```
    
## Citation

If you find this repo useful for your research, please consider citing the paper

```

@misc{ding2021multilevel,
      title={Multi-level Knowledge Distillation via Knowledge Alignment and Correlation}, 
      author={Fei Ding and Yin Yang and Hongxin Hu and Venkat Krovi and Feng Luo},
      year={2021},
      eprint={2012.00573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
For any questions, please contact Fei Ding.

## Reference

Most code is borrowed from CRD. Please also find the pretrained teacher models in the following repos:

- [CRD](https://github.com/HobbitLong/RepDistiller)
- [SSKD](https://github.com/xuguodong03/SSKD)

