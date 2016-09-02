# HOGSim
Source code based on the paper titled 'A scalable architecture for multi-class visual object detection' that was published in FPL 2015.
Paper available at http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7293961

If you use this code for evaluation and/or benchmarking, we appreciate if you cite the following paper:

@INPROCEEDINGS{fpl2015, 
author={Advani, Siddharth and Tanabe, Yasuki and Irick, Kevin and Sampson, Jack and Narayanan, Vijaykrishnan}, 
booktitle={Field Programmable Logic and Applications (FPL), 2015 25th International Conference on}, 
title={A scalable architecture for multi-class visual object detection}, 
year={2015}, 
month={Sept},
pages={1-8}, 
}

-------------
Contents
-------------

This code package contains the following files:

- HoG_Test_Vehicle.cpp is the code that runs the hardware model (proposed in FPL 2015) on the INRIA Benchmark

- We also provide a switch to run the original OpenCV HOG model for comparison purposes. 

----------------
Getting Started
----------------

- Navigate to Solution 
> cd ./HoG_Test_Vehicle/

- Compile code using Visual Studio (tested with Visual Studio 2013 on Windows 7) 

- Download the INRIA dataset from http://pascal.inrialpes.fr/data/human/

- Navigate to Release folder and run as follows
> HoG_Test_Vehicle.exe -i inputDir -o outputDir -m 1 -gt groundtruthDir

Detections are generated in the directory outputDir. 
V000.txt registers all detections in a format that can then be used by
Piotr Dollar's Pedestrian Detection toolkit (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip) for plotting P-R curves

- To run OpenCV HOG model
> HoG_Test_Vehicle.exe -i inputDir -o outputDir -m 0 -gt groundtruthDir

----------------
Dependencies
----------------

OpenCV 2.4.11 (make sure the OpenCV dlls are in the Release folder)
Dirent 1.20.1
hwHOGDll (see lib folder)

----------------
License
----------------

This code is published under the MIT License.

