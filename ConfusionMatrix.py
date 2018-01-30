# Usage:    ConfusionMatrix GroundTruthFilename PredictionFilename OutputFilename
#
#
#
# Notes: Change lines 21, 22 and 104 to provide input and output filenames as arguments.
#
# ProblemClass:
# "PixClass"    (Confusion matrix, precision, recall, accuracy, f1-score, etc.) inputs are two images (pixels > 0) are markers





from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import tifffile as tiff


# Get the filenames of the input predictions and ground truths
True_File_Name_OmeTiff = "GT.ome.tif" #sys.argv[1]
Pred_File_Name_OmeTiff = "Pred.ome.tif" #sys.argv[2]



# Read the predictions and ground truths as arrays
Pred_ImFile = tiff.TiffFile(Pred_File_Name_OmeTiff)
Pred_Data = Pred_ImFile.asarray()
y_pred = np.array(Pred_Data).ravel()        # Convert to 1-D array
True_ImFile = tiff.TiffFile(True_File_Name_OmeTiff)
True_Data = True_ImFile.asarray()
y_true = np.array(True_Data).ravel()
#y_true = [2, 0, 2, 2, 0, 1, 1, 0, 2]
#y_pred = [0, 0, 2, 2, 0, 2, 1, 1, 0]


# Check if metadata information of ground truth and prediction files are the same
Pred_Metadata = Pred_ImFile.ome_metadata
Pred_DimOrder = Pred_Metadata.get('OME').get('Image').get('Pixels').get('DimensionOrder')
True_Metadata = True_ImFile.ome_metadata
True_DimOrder = True_Metadata.get('OME').get('Image').get('Pixels').get('DimensionOrder')
if Pred_Data.size == True_Data.size:
    print("Ground truths and predictions has the same size.")
else:
    print("Ground truth and prediction sizes mismatch!")
if True_DimOrder == Pred_DimOrder:
    print("Dimension orders are the same.")
else:
    print("Dimension orders do not match!")


# Clean the predictions and ground truths (labels: 1,2,3... anything else is discarded)
y_true_cleaned = y_true[np.where(y_true > 0)]
y_pred_cleaned = y_pred[np.where(y_true > 0)]


# Compute confusion matrix
CM = confusion_matrix(y_true_cleaned, y_pred_cleaned)
print()
print("Confusion Matrix")
print(CM)

FP_PerClass = CM.sum(axis=0) - np.diag(CM)
FN_PerClass = CM.sum(axis=1) - np.diag(CM)
TP_PerClass = np.diag(CM)
TN_PerClass = CM.sum() - (FP_PerClass + FN_PerClass + TP_PerClass)

FP = FP_PerClass#.sum()
FN = FN_PerClass#.sum()
TP = TP_PerClass#.sum()
TN = TN_PerClass#.sum()

# (1.0*) added to cover python2 with floats
# Sensitivity, hit rate, recall, or true positive rate
TPR = 1.0*TP/(TP+FN)
# Specificity or true negative rate
TNR = 1.0*TN/(TN+FP)
# Precision or positive predictive value
PPV = 1.0*TP/(TP+FP)
# Negative predictive value
NPV = 1.0*TN/(TN+FN)
# Fall out or false positive rate
FPR = 1.0*FP/(FP+TN)
# False negative rate
FNR = 1.0*FN/(TP+FN)
# False discovery rate
FDR = 1.0*FP/(TP+FP)

# Overall accuracy
ACC = 1.0*(TP+TN)/(TP+FP+FN+TN)

# F1-score
F1score = 1.0*(2*TPR*PPV)/(TPR+PPV)

# print()
# print(ACC)
# print("Accuracy : %.3f +/- %.3f" % (ACC.mean(), ACC.std()))
# print()
# print(F1score)
# print("F1-score : %.3f +/- %.3f" % (F1score.mean(), F1score.std()))


# Write metric scores to the output file
OutputFilename = "Output.txt"
with open(OutputFilename, 'a') as f_handle:
    f_handle.write("Prediction file :\t%s\n" % Pred_File_Name_OmeTiff)
    f_handle.write("Ground truth file :\t%s\n" % True_File_Name_OmeTiff)
    f_handle.write("Confusion matrix :\n%s\n" % CM)
    f_handle.write("Precision :\t%s\n" % PPV)
    f_handle.write("Recall/Sens :\t%s\n" % TPR)
    f_handle.write("Specificity : \t%s\n" % TNR)
    f_handle.write("Neg Pred Value : %s\n" % NPV)
    f_handle.write("False Pos Rate : %s\n" % FPR)
    f_handle.write("False Neg Rate : %s\n" % FNR)
    f_handle.write("False Dis Rate : %s\n" % FDR)
    f_handle.write("Accuracy :\t%s\n" % ACC)
    f_handle.write("F1-score :\t%s\n" % F1score)
    f_handle.close()
