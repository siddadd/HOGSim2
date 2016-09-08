% This function takes a single weight vector output by hw hog in opencv and 
% converts to a form recognized by vlfeat
% Siddharth Advani
% 07/29/2015
%
% Example : w = opencv2vlfeat(w_oc, [15,7,36])
% Output w will be a 15x7x36 matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = hwopencv2vlfeat(w_oc, dim)

w_oc_rs = single(reshape(w_oc, dim(3), dim(2), [])) ;
w = permute(w_oc_rs,[3 2 1]);

end