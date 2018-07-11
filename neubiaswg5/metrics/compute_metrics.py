# Usage:            ComputeMetrics infile reffile problemclass tmpfolder
# infile:           Worflow output image
# reffile:     	    Reference images (ground truth)
# problemclass:     Problem class (6 character string, see below)
# tmpfolder:        A temporary folder required for some metric computation
# extra_params:     A list of possible extra parameters required by some of the metrics
#
# problemclass:
# "ObjSeg"      Object segmentation (DICE, AVD), work with binary or label 2D/3D masks images (regular multipage tif / OME-tif)
# "SptCnt"      Spot counting (Normalized spot count difference), same as above
# "PixCla"    	Pixel classification (Confusion matrix, F1-score, accuracy, precision, recall), same as above (0 pixels ignored)
# "TreTrc"      Filament tracing (trees), we consider including DIADEM metric but that requires to convert skeletons (workflow outputs) to SWC format
# "LooTrc"      Filament tracing (loopy networks)
# "ObjDet"      Object detection matching (TP, FN, FP, Recall, Precision, F1-score, RMSE over TP), not working yet
# "PrtTrk"      Particle (point) tracking (Particle Tracking Challenge metric), maximum linking distance set to a fixed value
# To be completed by Martin
# "ObjTrk"      Object tracking (Cell Tracking Challenge metric), for object divisions requires an extra text file encoding division locations

import os
import re
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from skimage.morphology import ball
from skimage.morphology import dilation
import numpy as np
from scipy import ndimage
import tifffile as tiff
from .img_to_xml import *
from .img_to_seq import *

def computemetrics( infile, reffile, problemclass, tmpfolder, extra_params=None ):

    # Remove all xml and txt (temporary) files in tmpfolder
    filelist = [ f for f in os.listdir(tmpfolder) if (f.endswith(".xml") or f.endswith(".txt")) ]
    for f in filelist:
        os.remove(os.path.join(tmpfolder, f))

    # Switch problemclass
    if problemclass == "ObjSeg":

        # Call Visceral (compiled) to compute DICE and average Hausdorff distance
        os.system("Visceral "+infile+" "+reffile+" -use DICE,AVGDIST -xml "+tmpfolder+"/metrics.xml"+" > nul 2>&1")
        with open(tmpfolder+"/metrics.xml", "r") as myfile:
            # Parse returned xml file to extract all value fields
            data = myfile.read()
            inds = [m.start() for m in re.finditer("value", data)]
            bchmetrics = [data[ind+7:data.find('"',ind+7)] for ind in inds]

    elif problemclass == "SptCnt":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        y_pred = np.array(Pred_Data).ravel()	# Convert to 1-D array
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        y_true = np.array(True_Data).ravel()	# Convert to 1-D array
        cnt_pred = np.count_nonzero(y_pred)
        cnt_true = np.count_nonzero(y_true)
        bchmetrics = abs(cnt_pred-cnt_true)/cnt_true

    elif problemclass == "PixCla":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        y_pred = np.array(Pred_Data).ravel()	# Convert to 1-D array
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        y_true = np.array(True_Data).ravel()	# Convert to 1-D array

        # Clean the predictions and ground truths (labels: 1,2,3... anything else is discarded)
        y_true_cleaned = y_true[np.where(y_true > 0)]
        y_pred_cleaned = y_pred[np.where(y_true > 0)]

        CM = confusion_matrix(y_true_cleaned, y_pred_cleaned)

        F1score = f1_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        Accscore = accuracy_score(y_true_cleaned, y_pred_cleaned, normalize=True, sample_weight=None)
        Prescore = precision_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        Recscore = recall_score(y_true_cleaned, y_pred_cleaned, labels=None, pos_label=1, average='weighted', sample_weight=None)
        bchmetrics = [CM, F1score, Accscore, Prescore, Recscore]

    elif problemclass == "TreTrc":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        Dst1 = ndimage.distance_transform_edt(Pred_Data==0)
        Dst2 = ndimage.distance_transform_edt(True_Data==0)
        indx = np.nonzero(np.logical_or(Pred_Data,True_Data))
        Dst1_onskl = Dst1[indx]
        Dst2_onskl = Dst2[indx]
        gating_dist = 5
        if extra_params is not None: gating_dist = int(extra_params[0])
        Err_skl_frc = (sum(Dst1_onskl > gating_dist)+sum(Dst2_onskl > gating_dist))/(Dst1_onskl.size+Dst2_onskl.size)
        bchmetrics = [Err_skl_frc]

    elif problemclass == "LooTrc":

        Pred_ImFile = tiff.TiffFile(infile)
        Pred_Data = Pred_ImFile.asarray()
        True_ImFile = tiff.TiffFile(reffile)
        True_Data = True_ImFile.asarray()
        Dst1 = ndimage.distance_transform_edt(Pred_Data==0)
        Dst2 = ndimage.distance_transform_edt(True_Data==0)
        indx = np.nonzero(np.logical_or(Pred_Data,True_Data))
        Dst1_onskl = Dst1[indx]
        Dst2_onskl = Dst2[indx]
        gating_dist = 5
        if extra_params is not None: gating_dist = int(extra_params[0])
        Err_skl_frc = (sum(Dst1_onskl > gating_dist)+sum(Dst2_onskl > gating_dist))/(Dst1_onskl.size+Dst2_onskl.size)
        bchmetrics = [Err_skl_frc]

        #Msk1 = dilation(Pred_Data, ball(5))
        #Msk2 = dilation(True_Data, ball(5))
        #Msk1 = np.array(Msk1).ravel()
        #Msk2 = np.array(Msk2).ravel()
        #dildice = 2*sum(Msk1&Msk2)/(sum(Msk1)+sum(Msk2))
        #bchmetrics = [dildice]

    elif problemclass == "ObjDet":

        ref_xml_fname = tmpfolder+"/reftracks.xml"
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile), False)
        in_xml_fname = tmpfolder+"/intracks.xml"
        tracks_to_xml(in_xml_fname, img_to_tracks(infile), False)
        # the third parameter represents the gating distance
        gating_dist = ''
        if extra_params is not None: gating_dist = extra_params[0]
        os.system('java -jar /usr/bin/DetectionPerformance.jar ' + ref_xml_fname + ' ' + in_xml_fname + ' ' + gating_dist)

        # Parse *.score.txt file created automatically in tmpfolder
        with open(in_xml_fname+".score.txt", "r") as f:
            bchmetrics = [line.split(':')[1].strip() for line in f.readlines()]

    elif problemclass == "PrtTrk":

        ref_xml_fname = tmpfolder+"/reftracks.xml"
        tracks_to_xml(ref_xml_fname, img_to_tracks(reffile), True)
        in_xml_fname = tmpfolder+"/intracks.xml"
        tracks_to_xml(in_xml_fname, img_to_tracks(infile), True)
        res_fname = in_xml_fname + ".score.txt"
        # the fourth parameter represents the gating distance
        gating_dist = ''
        if extra_params is not None: gating_dist = extra_params[0]
        os.system('java -jar /usr/bin/TrackingPerformance.jar -r ' + ref_xml_fname + ' -c ' + in_xml_fname + ' -o ' + res_fname + ' ' + gating_dist)

                # Parse the output file created automatically in tmpfolder
        with open(res_fname, "r") as f:
                        bchmetrics = [line.split(':')[0].strip() for line in f.readlines()]

    elif problemclass == "ObjTrk":

        # Martin: please adapt the code accordingly

        #tmp_folder = path1[:-len(suffix)]
        #ctc_gt_folder = os.path.join(tmp_folder, '01_GT')
        #ctc_gt_seg = os.path.join(ctc_gt_folder, 'SEG')
        #ctc_gt_tra = os.path.join(ctc_gt_folder, 'TRA')
        #ctc_res = os.path.join(tmp_folder, '01_RES')
        #os.mkdir(tmp_folder)
        #os.mkdir(ctc_gt_folder)
        #os.mkdir(ctc_gt_seg)
        #os.mkdir(ctc_gt_tra)
        #os.mkdir(ctc_res)
        #img_to_seq(path1, ctc_gt_seg, 'man_seg')
        #img_to_seq(path1, ctc_gt_tra, 'man_track')
        #img_to_seq(path2, ctc_res, 'mask')
        #os.system('./SEGMeasure ' + tmp_folder + ' 01')
        ## we need to copy the tracking text file to ctc_gt_tra and name it 'man_track.txt'
        ## we need to copy the tracking text file to ctc_res and name it 'res_track.txt'
        ##os.system('./TRAMeasure ' + tmp_folder + ' 01')
        ## we need to delete tmp_folder upon uploading the scores to Cytomine

        #Parse result files
        bchmetrics = ""

    return bchmetrics
