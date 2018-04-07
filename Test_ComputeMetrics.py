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

import sys
import os
from os import walk
from ComputeMetrics import computemetrics

infolder = sys.argv[1]
reffolder = sys.argv[2]
problemclass = sys.argv[3]
tmpfolder = sys.argv[4]

# Assume that matched images appear in same order in both lists
infilenames = [os.path.join(infolder,filename) for _, _, files in walk(infolder) for filename in files]
reffilenames = [os.path.join(reffolder,filename) for _, _, files in walk(reffolder) for filename in files]

for i in range(0,len(infilenames)):
	bchmetrics = computemetrics(infilenames[i],reffilenames[i],problemclass,tmpfolder)
	print(bchmetrics)