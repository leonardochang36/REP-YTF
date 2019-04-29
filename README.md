# REP-YTF

## Evaluation toolkit for the REP-YTF protocols_

This MATLAB package provides an evaluation toolkit for the REP-YTF protocols introduced in our CVPR 2018 paper Toward More Realistic Face Recognition Evaluation Protocols for the YouTube Faces Database. REP-YTF includes face verification and open/closed-set face identification at different FARs. Also, video-to-video and video-to-image scenarios with different openness values are considered on the face identification evaluations. The YouTube Faces (YTF) database [1] is the main benchmark database in this package. 

## You can find our paper in:

[Toward More Realistic Face Recognition Evaluation Protocols for the YouTube Faces Database link](https://github.com/leonardochang36/repo/REP-YTF/paper/Toward_More_Realistic_CVPR_2018_paper.pdf)

## The package contains the following components.

(1) REP-YTF configuration files contained in the config subfolder. The .mat configuration file provides basic image and video lists and indexes of the training, and test sets for the different benchmark evalutions.

(2) Evaluation utilities and demo codes included in the code subfolder.

(3) Basic features contained in the data subfolder. In this case we only included VGG-Face descriptors due to the limit on the size of files. 

## Usage

For a quick start, run the demo_VGG.m example that shows the evaluation process under REP-YTF protocols. For the benchmark of your own algorithm, replace the basic features, and integrate your own algorithms in the demo codes. Please make sure to report the average performance and the standard deviation.

Besides, ROC curves for verification, ROC curves for open-set identification at rank 1 and CMC curves for closed-set identification at different rank levels can also be illustrated. 

## If you use our protocol, please cite:

@inproceedings{Martinez-Diaz2018, 
author = {Martinez-Diaz, Y. and Mendez-Vazquez, H. and Lopez-Avila, L. and Chang, L. and {Enrique Sucar}, L. and Tistarelli, M.},
booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops},
doi = {10.1109/CVPRW.2018.00082},
isbn = {9781538661000},
issn = {21607516},
title = {{Toward more realistic face recognition evaluation protocols for the youtube faces database}},
volume = {2018-June},
year = {2018}
}

Version: 1.0
Date: 2018-04-13

References:
[1] L. Wolf, T. Hassner, and I. Maoz. Face recognition in unconstrained videos with matched background similarity. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
