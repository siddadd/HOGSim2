% This function takes a single weight vector output by opencv and converts to
% a form recognized by vlfeat
% Siddharth Advani
% 04/25/2015
%
% Example : w = opencv2vlfeat(w_oc, [15,7,36])
% Output w will be a 15x7x36 matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = opencv2vlfeat(w_oc, dim)

w_oc_rs = single(reshape(w_oc, dim(3), dim(1), [])) ;
w = permute(w_oc_rs,[2 3 1]);

end