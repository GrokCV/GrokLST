# Cross-sensor remote sensing imagery super-resolution via an edge-guided attention-based network
It is the pytorch implementation of the paper: Cross-sensor remote sensing imagery super-resolution via an edge-guided attention-based network. Here is the first commit, more illustrations of the implementation will be released soon.
# Our environments
python==3.8.11
GDAL==3.0.2
imageio==2.9.0
matplotlib==3.4.2
numpy==1.20.3
scikit_image==0.18.1
scipy==1.6.2
torch==1.8.0
torchvision==0.9.0
# Dataset
* Description: This dataset included the training and test sets, which were both obtained from GF-2. Taking the Wuhan urban agglomeration as the study region, we collected images of various scenes and different seasons in 2020. Specially, we conducted strict screening to ensure that the images were not perceptually contaminated with clouds and obvious artifacts in the process of collecting images. The GF-2 training dataset contained 850 LR/HR image pairs, 800 of which are used for training while another 50 for validation. Each HR image has a size of 1000×1000, and thus, we named the dataset 'GF1K'. The HR images were downsampled to obtain LR images with the size of 500×500×4 (resp., 250×250×4) when the scale factor was 2 (resp., 4), which were the input for the network.
* Download the dataset from the following link: [Baidu cloud disk](https://pan.baidu.com/s/1NeFj2gnAHuq0tKdZW_bqoA) (code:isku)
# Acknowledgement
Thanks for the excellent work and codebase of EDSR! 
