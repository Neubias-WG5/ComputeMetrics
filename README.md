# ComputeMetrics
Test_ComputeMetrics is a sample wrapper script calling the benchmarking module (ComputeMetrics), some calls and sample images are  provided.

# Test_ComputeMetric
# Test script parsing images (ome-tif or tif) from two folders (in and ref)
# and computing metrics on these two images for a specific problemclass (see compute metrics)
#
# Usage: python Test_ComputeMetrics.py in ref problemclass tmp
# in: input folder with workflow output images
# ref: reference folder with ground truth images
# problemclass: configure metrics computation depending on problem class
# tmp: path to a folder with I/O permission (used to store temporary data)
#
# Note: Sample images are provided for the different problem classes
#
# Sample calls: 
# python Test_ComputeMetrics.py imgs/in_objseg_tiflbl imgs/ref_objseg_tiflbl "ObjSeg" tmp
# python Test_ComputeMetrics.py imgs/in_sptcnt imgs/ref_sptcnt "SptCnt" tmp
# python Test_ComputeMetrics.py imgs/in_pixcla imgs/ref_pixcla "PixCla" tmp
