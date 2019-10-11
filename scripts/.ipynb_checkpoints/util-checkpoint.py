#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:07:42 2019

@author: user1
"""

import pandas as pd
df_train = pd.read_csv(r"../data/metadata/train.csv")

sirnas = []
exp = "HEPG2-03" # select experiment that has all sirnas
df_exp = df_train.groupby("experiment").get_group(exp)
for plate, df_exp_pl in df_exp.groupby("plate"):
    ss = sorted(df_exp_pl["sirna"].unique())
    sirnas += ss
    print("Plate {} has {} sirnas.".format(plate, df_exp_pl["sirna"].nunique()))
    print("   First 10 in ordered group:", ss[:10])
    

    
import pandas as pd
import numpy as np

pixel_df = pd.read_csv(r"../data/metadata/pixel_stats.csv")

GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,
                                10.45369445,  5.49959796, 9.81545561]),
                       np.array([7.95876312, 12.17305868, 5.86172946,
                                 7.83451711, 4.701167, 5.43130431]))

meanByExCh = pixel_df[['experiment','channel','mean']].groupby(['experiment','channel']).mean()

meanByExCh = meanByExCh.reset_index()
#experiment = stats_df[['experiment','mean']].groupby(['experiment']).mean()
#meanAll = meanByExCh.groupby(['channel']).mean()
#stdByExCh = stats_df[['experiment','channel','std']].groupby(['experiment','channel']).mean()

len(meanByExCh)/6 

stats_df = pd.DataFrame({'cmean':np.tile(GLOBAL_PIXEL_STATS[0],51),'cstd':np.tile(GLOBAL_PIXEL_STATS[1],51)})

LookupTbl= pd.concat([meanByExCh,stats_df],axis=1)

LookupTbl['batch_val'] = (LookupTbl['mean']-LookupTbl['cmean'])/LookupTbl['cstd']
LookupTbl['key'] = LookupTbl['experiment']+LookupTbl['channel'].apply(str)

LookupTbl[['key','batch_val']].to_csv('./BatchValLookup.csv')


batchval_df = pd.read_csv(r"./BatchValLookup.csv")

data=[[1,2,3]]
data_tf = tf.convert_to_tensor(np.asarray(data),np.float32)
