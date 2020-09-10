import os
import sys

from scGCO import *
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def nomalize_based_factor(data,normalizing_factor):
    data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    return data

sample_info=pd.read_csv("../processed_data/Rep11_MOB_info_scgco.csv")
locs=sample_info[["x","y"]].values
normalizing_factor = (sample_info["total_counts"]/np.median(sample_info["total_counts"])).values
exp_diff=1

for noise in [0.1,0.2,0.3,0.4,0.5,0.6]:
    for irep in range(10):
        
        ff="../processed_data/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+".csv"

        print(ff)
        df=pd.read_csv(ff,index_col=0)
        smooth_factor=10
        unary_scale_factor=100
        label_cost=10
        algorithm='expansion'
        sim_data=df.T.loc[(df != 0).sum(axis=0) >3,:].T

        sim_data_norm = nomalize_based_factor(sim_data,normalizing_factor)
        sim_data_norm=log1p(sim_data_norm)

        exp= sim_data_norm.iloc[:,0]
        cellGraph= create_graph_with_weight(locs, exp)

        locs_new,data_norm_new,newPoints=AddPoints_XY_and_update_data(locs,sim_data_norm,cellGraph,axis=1)

        gmmDict= multiGMM(data_norm_new)

        #gmmDict=grab_gmm(fileName)

        exp=data_norm_new.iloc[:,0].values
        cellGraph=create_graph_with_weight(locs_new, exp)

        result_df= identify_spatial_genes_optimize_gmm(locs_new, data_norm_new, cellGraph,gmmDict,
                                                        smooth_factor) #,factor= True)
        fileName="../scgco_results/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+"_scgco.csv"
        write_result_to_csv(result_df,fileName)

        fdr05 = result_df[result_df.fdr < 0.05]
        print(fdr05.shape)


