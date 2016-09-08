The INRIA Pedestrian full image data is reproduced here for convenience only. Full copyright remains with the authors (Naveet Dalal, et al.), see http://pascal.inrialpes.fr/data/human/. The converted version is so the associated evaluation code (dbEval) can be used to generate the associated ROC plots.

The 614 full positive training images are merged into a single seq file set00/V000.seq. The associated Pascal annotations are merged into a single vbb file set00/V000.vbb. The 1218 full negative test images are merged into a single seq file set00/V001.seq (these have no associated ground truth).

The 288 full positive testing images are merged into a single seq file set01/V000.seq. The associated Pascal annotations are merged into a single vbb file set01/V000.vbb. The 453 full negative testing images are merged into a single seq file set01/V001.seq (these have no associated ground truth).

Note 1: All evaluation results are reported on the 288 full positive testing images (the negative images are not used).

Note 2: The annotations for the training data are "incomplete" in that lots of pedestrians are NOT labeled. A few pedestrian annotations are also missing from the testing data.

Note 3: Since the labeled bounding boxes appear to have a wide variance in the width, we standardize all bbs to have a width of .41 times the height during evaluation. A thorough discussion of this choice appears in our PAMI 2011 paper.