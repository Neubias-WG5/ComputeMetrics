# Usage:            ComputeMetrics InFolder RefFolder ProblemClass (MetricsParams)
# InFolder:         workflow output folder (single files 16-bit ome multi-tiff images)
# RefFolder:        reference images folder (single files 16-bit ome multi-tiff images)
# ProblemClass:     problem class (string, see below)
# MetricsParams:    optional, metric specific
#
# Notes:    For every image ImageName.ome.tif in infolder, there must be a corresponding ImageName_lbl.ome.tif in reffolder
#           This is the Windows version of the script, should be slightly adapted for Linux (command line calls / exectuables)
#           In InFolder, images can actually also be plain tif files (workflow output), but still single file per image + .ome.tiff extension
#           Images should be calibrated, if not pix distance + isotropic assumed (e.g. for marker matching distance threshold)
#
# ProblemClass:
# "ObjSeg"      (DICE, AVD)
# "ObjDet"      (TP, FN, FP, Recall, Precision, F1-score, RMSE over TP)
# "PrtTrk"      (PTC)
#
# To be added
# "EventDet"    (FN, FP, TP, accuracy, precision, recall, F-score, True positive RMSE)
# "FilTreeTrc"  (DIADEM)
# "FilLoopTrc"  ?
# "ObjTrk"      (CTC), extra inputs: 2 text files encoding divisions as specified in PTC - same folders as images
# "PixClass"    (Confusion matrix, accuracy, precision, recall), inputs are two images (pixels > 0 are markers)
# "MrkClass"    (Confusion matrix, accuracy, precision, recall), inputs are two csv files (list of labels in same order)

import fnmatch
import os
import sys
from img_to_xml import *

if len(sys.argv)<4: # First argument is the name of the script
    sys.exit("Error: missing arguments.\nUsage: ComputeMetrics infolder outfolder MetricName (MetricArguments)")

infolder = sys.argv[1]
reffolder = sys.argv[2]
metricname = sys.argv[3]
pattern = "*.ome.tif"

# Assume that for every image in infolder there is a reference image named imagename_lbl.ome.tiff
for root, dirs, files in os.walk(infolder):
    for filename in fnmatch.filter(files, pattern):
        path1 = os.path.join(infolder, filename)
        path2 = os.path.join(reffolder, filename[:-8] + "_lbl.ome.tif")
        path3 = "./tmp/" +filename[:-8] + ".xml"

        if metricname == "ObjSeg":
            os.system("EvalSegmentation "+path1+" "+path2+" -use DICE -use AVGDIST -xml "+path3)
            # should erase tmp folder content upon results upload to Cytomine
        elif metricname == "ObjDet":
            gt_xml_fname = path1[:-8] + '.xml'
            tracks_to_xml(gt_xml_fname, img_to_tracks(path1), False)
            res_xml_fname = path2[:-8] + '.xml'
            tracks_to_xml(res_xml_fname, img_to_tracks(path2), False)
            os.system('java -jar DetectionPerformance.jar ' + gt_xml_fname + ' ' + res_xml_fname + ' ' + sys.argv[4])
        elif metricname == "PrtTrk":
            gt_xml_fname = path1[:-8] + '.xml'
            tracks_to_xml(gt_xml_fname, img_to_tracks(path1), True)
            res_xml_fname = path2[:-8] + '.xml'
            tracks_to_xml(res_xml_fname, img_to_tracks(path2), True)
            os.system('java -jar TrackingPerformance.jar -r ' + gt_xml_fname + ' -c ' + res_xml_fname + ' -o ' + res_xml_fname + ".score.txt" + ' ' + sys.argv[4])
