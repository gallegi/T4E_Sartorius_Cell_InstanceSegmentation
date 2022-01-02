# 1. About the challenge
- The goal of Sartorius - Cell Instance Segmentation challenge on Kaggle is to segment individual neuronal cells in microscopic images. 
- Note that we need to distinguish among different cells as separated instances, you can refer to [this link](https://serengetitech.com/tech/deep-learning-instance-segmentation/) if not familiar with instance segmentation task. This is fundamentally different from semantic segmentation when only a binary mask is predicted to mark whether a pixel belongs to a foreground class or background. 
![Challenge image](/materials/challenge_img.png "Image taken from the challenge homepage")


# 2. Solution overview

#TODO: image goes here

More detail can be found in this discussion in Kaggle:


# 3. How to run the code
- We have tried many different models such as PointRend, MaskRCNN Swin Transformer, CellPose, GCNet, etc. However the final solution is the ensembling prediction of 2 MaskRCNN ResNeSt200 model, thus, to keep everything simple, we only provide the training and inference code of MaskRCNN ResNeSt200.
