# adaGNN-IQA
The official repo of the paper.

`perceptual no-reference image quality assessment via adaptive deformable convolution and graph neural network.`  

## Abatract

The lack of connection between distortion information and image content has become a major bottleneck in the development of no-reference image quality assessment (NR-IQA). To address this issue, we propose an NR-IQA method that extracts features by deformable convolution and represents distortions with graph structures. In this paper, we propose an adaptive deformable convolution module that realizes the simulation of the edge enhancement mechanism (side suppression effect) of the human visual system (HVS) by learning offsets to adapt the convolution kernel to the features of various objects in the image. Compared to other deformable convolution based NR-IQA methods, the model incorporates a graph neural network to represent non-Euclidean information in the image,improving the model’s ability to predict under-representation distortions. We thoroughly evaluated the model on five public image quality databases: LIVE, LIVEC, KonIQ-10k, CSIQ, Kadid-10k. Experimental results demonstrate that the proposed model outperforms other state-of-the-art NR-IQA methods across most benchmark datasets. 

## Framework Overview

The following figure shows the framework of this model :

![fig333333](https://github.com/Zzzhy5/P-nriqa/assets/148023964/61238b66-e5a5-4d85-84d8-0b36f81129e1)

## Result
Comparisons with other good methods on all distortion types in the Kadid-10k dataset, boldface indicates best data.

![kadid每种失真对比](https://github.com/Zzzhy5/P-nriqa/assets/148023964/8c1041c1-31fe-47a5-9852-0fec0093a3b2)

## Preparation
1. Create a env environment using miniconda：

`conda create --name Piqa python ==3.7.7`

2. Install pytorch

`conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch`

3. Create environment and install as following: 

`pip install -r requirements.txt`

## Usages

1. Run this code when you want to run it on the kadid-10k dataset:

   `python pretrain.deform.py --dataset kadid-P`

2. Run this command when you want to test this code on a dataset:

   `python finetuned.py --dataset DATASET  --gpus 1 --gpu_ids 0 --bs 32`

The following parameters can be adjusted according to your needs:

 * --dataset: The dataset you want to test.
 * --ckpt: You can save the trained model under this path.
 * --gpus: The number of gpu's you want to use for training.
 * --bs: Batch size.
