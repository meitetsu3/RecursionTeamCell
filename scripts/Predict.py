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

CELL_TYPES = {'HEPG2':0,'HUVEC':1,'RPE':2,'U2OS':3}

batchval_df = pd.read_csv(r"./BatchValLookup.csv")

test_df = pd.read_csv(r"../data/metadata/test.csv")
train_df = pd.read_csv(r"../data/metadata/train.csv")
test_cnt_df = pd.read_csv(r"../data/metadata/test_controls.csv")
train_cnt_df = pd.read_csv(r"../data/metadata/train_controls.csv")
sirna_groups_df =pd.read_csv(r"../data/metadata/sirna_groups.csv")
train_df["pred"] = np.nan
test_df["pred"] = np.nan
test_cnt_df["pred"] = np.nan
train_cnt_df["pred"] = np.nan

submission_df = pd.read_csv(r"../data/metadata/sample_submission.csv")

export_dir = r"../model/Resnet50FAS64m05-D554WMPole-bs16-ep45-CLR001-015-WD5-Cell-ValC03HO-FlipRotCrop/saved_model/1570534434"
predict_fn = predictor.from_saved_model(export_dir)

# grabbing both site 1 and site 2 for the 
# 19897 records (both on submission and test.csv)

# double checking the ids in test.csv and submission.csv are ordered the same.1566708068
# they are the same.
for i, idcode in enumerate(test_df["id_code"]):
    if (idcode!=submission_df["id_code"][i]):
        print("Does not match:{}".format(idcode))
        break

# Use sample_submission to get images
#imagbase = r"../data/raw/test/"
imagbase = r"../data/raw/test/"
tgt = submission_df#train_df #submission_df
image_shape = [512, 512, 6]
img_s1 = np.zeros(image_shape,dtype=np.float32)
img_s2 = np.zeros(image_shape,dtype=np.float32)
#probmtx = pd.DataFrame(np.zeros([19897,1139],dtype=np.float32))
#probmtxhd = pd.DataFrame(columns=['exp', 'plate'])
#probmtx = pd.concat([probmtxhd,probmtx])
problist = []
for i, idcode in enumerate(tgt["id_code"]): 
    folder = idcode[0:idcode.find("_")]+r"/Plate"+idcode[idcode.find("_")+1:idcode.find("_")+2]+r"/"
    well = idcode[idcode.find("_")+3:len(idcode)+1]
    for c in range(0,6):
        imgpath_s1 = imagbase + folder+well+r"_s1_w"+str(c+1)+".png"
        imgpath_s2 = imagbase + folder+well+r"_s2_w"+str(c+1)+".png"
        img_s1[:,:,c] = (imread(imgpath_s1,format='png')-GLOBAL_PIXEL_STATS[0][c])/GLOBAL_PIXEL_STATS[1][c]
        img_s2[:,:,c] = (imread(imgpath_s2,format='png')-GLOBAL_PIXEL_STATS[0][c])/GLOBAL_PIXEL_STATS[1][c]
    img_s1_c = img_s1[64:448,64:448,:] # 384 center crop
    img_s2_c = img_s2[64:448,64:448,:]
    img_s1_ul = img_s1[0:384,0:384,:]
    img_s2_ul = img_s2[0:384,0:384,:]
    img_s1_ur = img_s1[0:384,128:512,:] 
    img_s2_ur = img_s2[0:384,128:512,:]
    img_s1_ll = img_s1[128:512,0:384,:] 
    img_s2_ll = img_s2[128:512,0:384,:]
    img_s1_lr = img_s1[128:512,128:512,:] 
    img_s2_lr = img_s2[128:512,128:512,:]
    cell = np.reshape(CELL_TYPES[idcode[0:idcode.find("-")]],(1,))
    plate = np.reshape(int(idcode[idcode.find("_")+1:idcode.find("_")+2]),(1,))
    exp = np.reshape(idcode[0:idcode.find("_")],(1,))
    print("{},exp:{}, cell : {}, plate: {}".format(i,exp,cell,plate))
    pred_s1_c = predict_fn({'image': np.reshape(img_s1_c,(1,384,384,6)),'cell':cell}) # can createa a batch. to be improved.
    pred_s2_c = predict_fn({'image': np.reshape(img_s2_c,(1,384,384,6)),'cell':cell}) 
    pred_s1_ul = predict_fn({'image': np.reshape(img_s1_ul,(1,384,384,6)),'cell':cell}) 
    pred_s2_ul = predict_fn({'image': np.reshape(img_s2_ul,(1,384,384,6)),'cell':cell}) 
    pred_s1_ur = predict_fn({'image': np.reshape(img_s1_ur,(1,384,384,6)),'cell':cell}) 
    pred_s2_ur = predict_fn({'image': np.reshape(img_s2_ur,(1,384,384,6)),'cell':cell}) 
    pred_s1_ll = predict_fn({'image': np.reshape(img_s1_ll,(1,384,384,6)),'cell':cell}) 
    pred_s2_ll = predict_fn({'image': np.reshape(img_s2_ll,(1,384,384,6)),'cell':cell}) 
    pred_s1_lr = predict_fn({'image': np.reshape(img_s1_lr,(1,384,384,6)),'cell':cell}) 
    pred_s2_lr = predict_fn({'image': np.reshape(img_s2_lr,(1,384,384,6)),'cell':cell}) 
    prob = pred_s1_c["probabilities"] \
            +pred_s2_c["probabilities"] \
            +pred_s1_ul["probabilities"] \
            +pred_s2_ul["probabilities"] \
            +pred_s1_ur["probabilities"] \
            +pred_s2_ur["probabilities"] \
            +pred_s1_ll["probabilities"] \
            +pred_s2_ll["probabilities"] \
            +pred_s1_lr["probabilities"] \
            +pred_s2_lr["probabilities"]
    
    row = np.concatenate((exp,plate,prob[0]),axis=0)
    problist.append(row)

df_prob = pd.DataFrame(problist,columns=['exp', 'plate',*list(range(0,1139))])
df_prob[list(range(0,1139))] = df_prob[list(range(0,1139))].astype(float) # takes long time
df_prob.dtypes
prob = df_prob[list(range(0,1108))]
pred_raw = prob.idxmax(axis=1)
pred_raw = pd.DataFrame(pred_raw, columns=["pred_raw"])
pred_raw = pd.concat([df_prob[["exp","plate"]],pred_raw],axis=1)

pred_raw_grp = pred_raw.join(sirna_groups_df.set_index('sirna'),on='pred_raw')
pred_raw_grp_cnt = pred_raw_grp.groupby(['exp','plate','group']).count().reset_index()

pred_raw_grp_cnt_spread=pd.pivot_table(pred_raw_grp_cnt,index=['exp','plate'],columns='group',values='pred_raw',fill_value = 0)
pred_raw_grp_cnt_spread=pred_raw_grp_cnt_spread.reset_index()
pred_raw_grp_cnt_spread.columns.name = None

from scipy.optimize import linear_sum_assignment

groupest=[]
for exp,df in pred_raw_grp_cnt_spread.groupby('exp'):
    row_ind, col_ind = linear_sum_assignment(-df[[1,2,3,4]].to_numpy())
    groupest.append(col_ind)

pred_grp = pd.DataFrame(np.concatenate(groupest), columns=["pred_grp"])
pred_grp = pd.concat([pred_raw_grp_cnt_spread[["exp","plate"]],pred_grp],axis=1)


for i in pred_grp.iterrows():
    tgt_sirna=sirna_groups_df[sirna_groups_df["group"]==i[1]['pred_grp']]["sirna"].to_list()
    tgt_row = (df_prob["exp"]==i[1]['exp']) & (df_prob["plate"]==i[1]['plate'])
    df_prob.loc[tgt_row,tgt_sirna]=0


df_prob.loc[:,list(range(1108,1139))]=0

sirna=[]
for exp,df in df_prob.groupby('exp'):
    print(df)
    row_ind, col_ind = linear_sum_assignment(-df.loc[:,list(range(0,1108))].to_numpy())
    sirna.append(col_ind)

pred_sirna = pd.DataFrame(np.concatenate(sirna), columns=["pred_sirna"])

tgt.iloc[:,1]=pred_sirna["pred_sirna"]

tgt.to_csv("../submit_prediction.csv",index=False)

