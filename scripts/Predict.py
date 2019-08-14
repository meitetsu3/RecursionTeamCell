# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:38:27 2019

@author: meite
"""
import pandas as pd
import numpy as np
from skimage.io import imread
from tensorflow.contrib import predictor

# The mean and stds for each of the channels
GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,
                                10.45369445,  5.49959796, 9.81545561]),
                       np.array([7.95876312, 12.17305868, 5.86172946,
                                 7.83451711, 4.701167, 5.43130431]))
test_df = pd.read_csv(r"../data/metadata/test.csv")
submission_df = pd.read_csv(r"../data/metadata/sample_submission.csv")


export_dir = r"../model/Resnet50-bs16-CLR01_06-DC4/saved_model/1565720609"
predict_fn = predictor.from_saved_model(export_dir)

# grabbing both site 1 and site 2 for the 
# 19897 records (both on submission and test.csv)

# double checking the ids in test.csv and submission.csv are ordered the same.
# they are the same.
for i, idcode in enumerate(test_df["id_code"]):
    if (idcode!=submission_df["id_code"][i]):
        print("Does not match:{}".format(idcode))
        break

# Use sample_submission to get images
imagbase = r"../data/raw/test/"
image_shape = [512, 512, 6]
img_s1 = np.zeros(image_shape,dtype=np.float32)
img_s2 = np.zeros(image_shape,dtype=np.float32)
for i, idcode in enumerate(submission_df["id_code"]):
    folder = idcode[0:idcode.find("_")]+r"/Plate"+idcode[idcode.find("_")+1:idcode.find("_")+2]+r"/"
    well = idcode[idcode.find("_")+3:len(idcode)+1]
    for c in range(0,6):
        imgpath_s1 = imagbase + folder+well+r"_s1_w"+str(c+1)+".png"
        imgpath_s2 = imagbase + folder+well+r"_s2_w"+str(c+1)+".png"
        img_s1[:,:,c] = (imread(imgpath_s1)-GLOBAL_PIXEL_STATS[0][c])/GLOBAL_PIXEL_STATS[1][c]
        img_s2[:,:,c] = (imread(imgpath_s2)-GLOBAL_PIXEL_STATS[0][c])/GLOBAL_PIXEL_STATS[1][c]
    pred_s1 = predict_fn({"feature": np.reshape(img_s1,(1,512,512,6))})
    pred_s2 = predict_fn({"feature": np.reshape(img_s2,(1,512,512,6))})
    prob = pred_s1["probabilities"]+pred_s2["probabilities"]
    submission_df.iloc[i,1]=np.argmax(prob)
    print("image {} : {}".format(i,pred_s1["classes"])) # just cheking s1 class

submission_df.to_csv("../submission.csv",index=False)

"""
testing model with training data

train_df = pd.read_csv(r".\data\raw\train\train.csv")
imagbase = r"./data/raw/train/"
for i, idcode in enumerate(train_df["id_code"]):
    if i < 20:
        folder = idcode[0:idcode.find("_")]+r"/Plate"+idcode[idcode.find("_")+1:idcode.find("_")+2]+r"/"
        well = idcode[idcode.find("_")+3:len(idcode)+1]
        for c in range(0,6):
            imgpath_s1 = imagbase + folder+well+r"_s1_w"+str(c+1)+".png"
            imgpath_s2 = imagbase + folder+well+r"_s2_w"+str(c+1)+".png"
            img_s1[:,:,c] = imread(imgpath_s1)
            img_s2[:,:,c] = imread(imgpath_s2)
        pred_s1 = predict_fn({"feature": np.reshape(img_s1,(1,512,512,6))})
        pred_s2 = predict_fn({"feature": np.reshape(img_s2,(1,512,512,6))})
        prob = pred_s1["probabilities"]+pred_s2["probabilities"]
        submission_df.iloc[i,1]=np.argmax(prob)
        print("image {} : {}".format(i,pred_s1["classes"])) # just cheking s1 class
"""