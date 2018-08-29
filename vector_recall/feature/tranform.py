#coding=utf-8

import time
import os
import sys 
sys.path.append("../")
import pandas as pd
import numpy as np
import random
import json

import config.config as config

origin_file = config.DATA_ORIGIN
sub_file = config.DATA_ADDIDS
lib_file = config.DATA_LIB
index_file = config.INDEX_FILE
columns = config.COLUMNS
sub_columns = config.MODEL_O
model_col = config.MODEL_T
field_index = config.FIELD_INDEX

# merge new feature
def video_ids(video_list,max_num,df_len):
    out_list = []
    for i in range(df_len):
        ran = random.randint(0,max_num)
        tmp_list = video_list[ran:ran+10]
        out_str = "|".join(tmp_list)
        out_list.append(out_str)
    return out_list

# create origin sample
def get_df(filename,columns,sub_columns,subfile):
    df = pd.read_csv(filename,names=columns,index_col=False)
    df_filter = df[df['label']==1]
    df_new = df_filter[sub_columns]
    #print df_new.head()
    video_list = list(set(df_new["docid"]))
    max_num = len(video_list) - 11
    df_len = df_new.shape[0]
    videoIds = video_ids(video_list,max_num,df_len)
    df_new["video_ids"] = videoIds
    df_new.to_csv(subfile,index=False,sep=",",header=False)

# feature tranform
def tran_process(columns,filename,lib_file):
    df = pd.read_csv(filename,names=columns,index_col=False)
    video_ids = set(df['docid'])
    df_num = df.shape[0]
    for i in range(df_num):
        docids = set(df['video_ids'].ix[i].split("|"))
        video_ids = video_ids | docids
    feature_index = {}
    index = 0
    for video in video_ids:
        feature_index[video] = str(index)
        index += 1
    print "videoNUm:",index
    with open(lib_file,"w") as f:
        for i in range(df_num):
            f_list = []
            label = ""
            for field in columns:
                findex = field_index[field]
                value = df.ix[i,[field]].values[0]
                if field == "docid":
                    label = feature_index[value]
                elif field == "video_ids":
                    for video in value.split("|"):
                        fea_index = feature_index[video]
                        tmp = findex + ":" + str(fea_index) + ":" + "1.0"
                        f_list.append(tmp)
                elif field == "userid":
                    fea_index = feature_index.get(value,"")
                    if fea_index == "":
                        feature_index[value] = str(index)
                        fea_index = str(index)
                        index += 1
                    tmp = findex + ":" + str(fea_index) + ":" + "1.0"
                    f_list.append(tmp)
                else:
                    fea_index = feature_index.get(field,"")
                    if fea_index == "":
                        feature_index[field] = str(index)
                        fea_index = str(index)
                        index += 1
                    tmp = findex + ":" + str(fea_index) + ":" + str(round(value,5))
                    f_list.append(tmp)
            sample = label + "," + " ".join(f_list)
            f.write(sample + "\n")
    print "featureNum:",index
    with open(index_file,"w") as f:
        field_index_out = "field_index\001" + json.dumps(field_index)
        feature_index_out = "feature_index\001" + json.dumps(feature_index)
        f.write(field_index_out + "\n") 
        f.write(feature_index_out + "\n")       

def main():
    get_df(origin_file,columns,sub_columns,sub_file)
    tran_process(model_col,sub_file,lib_file)

if __name__ == "__main__":
    main()
