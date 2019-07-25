# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:38:27 2019

@author: meite
"""
import pandas as pd
import numpy as np
from skimage.io import imread

from tensorflow.contrib import predictor

test_df = pd.read_csv(r".\data\raw\test\test.csv")
submission_df = pd.read_csv(r".\data\raw\sample_submission.csv")


export_dir = r".\model\saved_model\1563817997"
predict_fn = predictor.from_saved_model(export_dir)

# grabbing both site 1 and site 2 for the 
# 19897 records (both on submission and test.csv)

test_df = pd.read_csv(r".\data\raw\test\test.csv")
submission_df = pd.read_csv(r".\data\raw\sample_submission.csv")

# double checking the ids in test.csv and submission.csv are ordered the same.
# they are the same.
for i, idcode in enumerate(test_df["id_code"]):
    if (idcode!=submission_df["id_code"][i]):
        print("Does not match:{}".format(idcode))
        break

# Use sample_submission to get images
imagbase = r"./data/raw/test/"
image_shape = [512, 512, 6]
img_s1 = np.zeros(image_shape,dtype=np.int8)
img_s2 = np.zeros(image_shape,dtype=np.int8)
for i, idcode in enumerate(submission_df["id_code"]):
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

submission_df.to_csv("submission.csv")