# Usage:            ComputeMetrics infile reffile problemclass tmpfolder
# infile:         	Worflow output image
# reffile:     	    Reference images (ground truth)
# problemclass:     Problem class (6 character string, see below)
# tmpfolder:    	A temporary folder required for some metric computation
#
# problemclass:
# "ObjSeg"      (DICE, AVD)
# "SptCnt"      (Normalized spot counts difference)
# "PixCla"    	(Confusion matrix, F1-score, accuracy, precision, recall), inputs are two images (pixels > 0 are markers)
# To be completed by Martin
# "ObjDet"      (TP, FN, FP, Recall, Precision, F1-score, RMSE over TP)
# "PrtTrk"      (PTC)
# "ObjTrk"      (CTC), extra inputs: 2 text files encoding divisions as specified in CTC - same folders as images
# To be added
# "FilTreeTrc"  (DIADEM)
# "FilLoopTrc"  ?

import os
import re
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
import tifffile as tiff
from img_to_xml import *
from img_to_seq import *

def computemetrics( infile, reffile, problemclass, tmpfolder ):
	
	# Remove all .xml (temporary) files in tmpfolder
	filelist = [ f for f in os.listdir(tmpfolder) if f.endswith(".xml") ]
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
		
	elif problemclass == "ObjDet":
	
		# Martin please review / add valid images: Unknown exception caught with current test images
	
		gt_xml_fname = tmpfolder+"/intracks.xml"
		tracks_to_xml(gt_xml_fname, img_to_tracks(infile), True)
		res_xml_fname = tmpfolder+"/reftracks.xml"
		tracks_to_xml(res_xml_fname, img_to_tracks(reffile), True)
		os.system('java -jar DetectionPerformance.jar ' + gt_xml_fname + ' ' + res_xml_fname + ' ' + tmpfolder+'/score.txt')
		
		#Parse score.txt file
		bchmetrics = ""
	
	elif problemclass == "PrtTrk":
	
		# Martin please add valid test images and test
	
		gt_xml_fname = tmpfolder+"/intracks.xml"
		tracks_to_xml(gt_xml_fname, img_to_tracks(infile), True)
		res_xml_fname = tmpfolder+"/reftracks.xml"
		tracks_to_xml(res_xml_fname, img_to_tracks(reffile), True)
		os.system('java -jar TrackingPerformance.jar -r ' + gt_xml_fname + ' -c ' + res_xml_fname + ' -o ' + ".score.txt") 
		# Could add a gobal paremeter to change maximum linking distance
	
		bchmetrics = ""
	
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
		
	return bchmetrics;
	