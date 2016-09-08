% This script takes the Dalal svm model and visualizes it using vlfeat
% Siddharth Advani
% 07/28/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('R:/ska130/Research/Tools/vlfeat-0.9.20/toolbox/');
vl_setup
load('C:\RnD\Repos\HoG_Demo_Release\HoG_Train_Vehicle\data\model\hmdescriptorvector.dat');
w = opencv2vlfeat(hmdescriptor, [15,7,36]);
imhog = vl_hog('render', w, 'variant', 'DalalTriggs');
imshow(imhog,[]);