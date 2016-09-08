###################################################################
#				   												  #	
#	HoG-SVM Evaluation Tool										  #
#                  												  #	 
###################################################################

1. Introduction

We use Piotr Dollar's framework to evaluate the hardware equivalent of our FPL 2015 work on HoG-SVM

If using any of the above, please cite the following works in any resulting publication:

 @article{Dollar2012PAMI,
   author = {Piotr Doll\'ar and Christian Wojek and Bernt Schiele and Pietro Perona},
   title = {Pedestrian Detection: An Evaluation of the State of the Art},
   journal = {PAMI},
   volume = {34},
   year = {2012},
 }
 
 @INPROCEEDINGS{fpl2015, 
author={Advani, Siddharth and Tanabe, Yasuki and Irick, Kevin and Sampson, Jack and Narayanan, Vijaykrishnan}, 
booktitle={Field Programmable Logic and Applications (FPL), 2015 25th International Conference on}, 
title={A scalable architecture for multi-class visual object detection}, 
year={2015}, 
month={Sept},
pages={1-8}, 
}

###################################################################

2. Installation

To run the evaluation, download and install  

(1) Piotr's Matlab Toolbox -> http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
(2) Piotr's Matlab evaluation/labeling code (version 3.2.1) -> http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip

###################################################################

3. Evaluation

(1) Modify dbEval.m to include your detection algorithms 
(2) Place your detection output V000.txt in the corresponding folder of <root>\code3.2.1\data-INRIA\res\
(3) Delete the previously cached results in <root>\code3.2.1\results\ 
(4) Run dbEval.m in Matlab (Tested on Windows 7 - Matlab R2014a)

################################################################### 