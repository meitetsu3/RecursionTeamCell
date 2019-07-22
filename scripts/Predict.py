# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:38:27 2019

@author: meite
"""
import pandas as pd
from tensorflow.contrib import predictor
export_dir = r"./model/saved_model/1563788066"

test_df = pd.read_csv(r"./data/raw/test/test.csv")

submission_df = pd.read_csv(r"./data/raw/sample_submission.csv")

predict_fn = predictor.from_saved_model(export_dir)


predictions = predict_fn(
    {"F1": 1.0, "F2": 2.0, "F3": 3.0, "L1": 4.0})

print(predictions)

def generate_df(train_df,sample_num=1):
    train_df['path'] = train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate') + '_s'+str(sample_num) + '_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna'])
    return train_df
