# EfficientGaze: Resource-efficient Gaze Estimation via Frequency-domain Multi-task Contrastive Learning

This repository contains the introductions and the codes for the submission to TOSN: EfficientGaze: Resource-efficient Gaze Estimation via Frequency-domain Multi-task Contrastive Learning by [Lingyu Du](https://github.com/LingyuDu), [Xucong Zhang](https://www.ccmitss.com/zhang), and [Guohao Lan](https://guohao.netlify.app/). If you have any questions, please send an email to Lingyu.Du AT tudelft.nl.

## Description

Gaze estimation is of great importance to many scientific fields and daily applications, ranging from fundamental research in cognitive psychology to attention-aware systems. Despite recent advancements in deep learning have led to highly accurate gaze estimation systems, these solutions often come with high computational costs and depend on large-scale labeled gaze data for supervised learning, posing significant practical challenges. To move beyond these limitations, we present EfficientGaze, a resource-efficient framework for gaze representation learning. We introduce the frequency-domain gaze estimation, which exploits the feature extraction capability and the spectral compaction property of discrete cosine transform to substantially reduce the computational cost of gaze estimation systems for both calibration and inference. Moreover, to overcome the data labeling hurdle, we design a contrastive learning-based framework for unsupervised gaze representation learning. Specifically, we introduce gaze-specific data augmentation to preserve the gaze-semantic features and devise a novel multi-task contrastive learning framework to learn gaze representations that are generic across subjects in an unsupervised manner. Our evaluation on two gaze estimation datasets demonstrates that EfficientGaze achieves comparable gaze estimation accuracy to existing supervised learning-based approaches, while enabling up to 6.80 times and 1.67 times speedup in system calibration and gaze estimation, respectively.

## Getting Started

### Dependencies

* Tensorflow-gpu 2.9.0, jpeg2dct
* ex. Ubuntu 20.04

### Dataset preparation
In our implementation, to efficiently obtain DCT coefficients of the original RGB images, we preprocess the dataset by saving all the images in the format of .jpg. We then use jpeg2dct to directly read DCT coefficients from a jpg image in the training and testing stages. Moreover, we apply facial landmark detection to locate the positions of eyes and save the coordinates of periocular bounding boxes for each image as numpy arrays.

### Codes
* The file Contrastive_gaze_representation_learning.ipynb contains the main function for contrastive gaze representation learning.
* The file common_functions.py includes how to calculate the average angular error given a batch of images and gaze annotations.
* The file trans_in_rgb.py details the data augmentation we adopted in this project.

## System Overview
The overview design of EfficientGaze is shown in the following figure, which includes three stages: the self-supervised pre-training stage, the supervised calibration stage, and the deployment stage. In the self-supervised pre-training stage, EfficientGaze takes unlabeled facial images as inputs to pre-train a gaze embedding network for unsupervised gaze representation learning. Next, the pre-trained gaze embedding network is transferred to the supervised calibration stage, and serves as a feature extractor for the downstream gaze estimation task. Specifically, taking a small number of labeled facial images from the targeted subject as input, we first leverage the frequency-domain image processing module to obtain the DCT coefficients of the original RGB images. Then, we fine-tune the pre-trained gaze embedding network and the gaze estimator for subject-specific gaze estimation. Note that subject-specific calibration is essential to ensure good gaze-estimation accuracy, even for high-end eye-tracking systems. Finally, in the deployment stage, the fine-tuned gaze embedding network and the gaze estimator are used for run-time gaze estimation. 

<img src="https://github.com/FreeGaze/EfficientGaze/blob/main/figures/overview.png" alt="My Image" width="800"/>


## Frequency-domain Gaze Estimation

To reduce the latency for gaze estimation system in both calibration and inference stages, we devise the frequency-domain gaze estimation. It leverages the feature extraction capability of the discrete cosine transform (DCT) and takes the frequency-domain DCT coefficients of the original RGB image as inputs for gaze estimation. Moreover, motivated by the fact that the critical content-defining information of the image is concentrated in the low end of the frequency spectrum, whereas signals in the high-frequency end are mostly trivial and are associated with noise, we further exploit the spectral compaction property of DCT to aggressively compact the essential perceptual information in the RGB image into a few DCT coefficients. The pipeline of frequency-domain image processing is shown as below:

<img src="https://github.com/FreeGaze/EfficientGaze/blob/main/figures/dctProcessing.png" alt="My Image" width="800"/>


## Frequency-domain Multi-task Contrastive Learning

To overcome the data labeling hurdle of existing supervised gaze estimation systems, we propose a contrastive learning (CL)--based framework that leverages unlabeled facial images for gaze representation learning. The conventional CL are ill-suited for gaze estimation, as they focus on learning general representations that are more related to the appearance and the identity of the subjects. To resolve this challenge, we introduce a set of optimizations to enable contrastive gaze representation learning. Specifically, we introduce gaze-specific data augmentation to generate positive and negative image pairs. These augmented images are then transformed from the RGB color space to the DCT frequency domain using our devised frequency-domain image processing, denoted as $Freq\(\cdot\)$. Subsequently, these augmented images in the frequency domain are fed into the gaze embedding network $f\(\cdot\)$, which is shared across different subjects. This network learns gaze representations that are invariant to subject variations. 

To encode subject-specific features, we employ multi-task learning. This approach further maps the generic gaze representations in the general representation space $\mathbb{GP}$ to subject-specific embedding spaces $\mathbb{SP}_i$ tailored for each subject $i$, through the proposed subject-conditional projection, referred to as $S \(\cdot\)$. 

Finally, to encourage the shared gaze embedding network to learn discriminative gaze representations, we design the subject-specific gaze-aware contrastive loss within each subject-specific embedding space to train the end-to-end framework. The pipeline of the proposed frequency-domain multi-task contrastive gaze representation learning framework is shown as below:

<img src="https://github.com/FreeGaze/EfficientGaze/blob/main/figures/CL_framework.png" alt="My Image" width="800"/>
