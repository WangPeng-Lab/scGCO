import os
import sys
from scGCO import *
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def get_numbers(df,fdr,methods):
    if methods=="scGCO":
        fdr_df=df[df.fdr < fdr].sort_values(by=['fdr'])
        arr= fdr_df.index.str.strip("gene").astype("int").values
    elif methods=="spatialDE":
        fdr_df=df.query('qval <@fdr & g!="log_total_count"')
        arr=fdr_df.g.str.strip("gene").astype("int").values
    elif methods=="SPARK":
        fdr_df=df.query('adjusted_pvalue  <@fdr')
        arr=fdr_df.index.str.strip("gene").astype("int").values
    if len(arr)==0:
        arr=np.array([])
    true_genes=np.arange(1,1001)
    false_genes=np.arange(1001,10001)
    positive_genes= arr.copy()
    negative_genes= np.setdiff1d(np.arange(1,10001),arr)
    TN=len(np.intersect1d(false_genes,negative_genes))
    TP=len(np.intersect1d(true_genes,positive_genes))
    FN=len(np.intersect1d(true_genes,negative_genes))
    FP=len(np.intersect1d(false_genes,positive_genes))
    return [TN,TP,FN,FP]


fdr_cutoff=0.05
fdr01=0.01

exp_diff=1
scgco_list=[]
spatialde_list=[]
spark_list=[]
for i,noise in enumerate([0.1,0.2,0.3,0.4,0.5,0.6]):
    scgco_list.append([])
    spatialde_list.append([])
    spark_list.append([])
    for irep in range(10):
        scgco_ff="../scgco_results/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+ "_scgco.csv"
        print(scgco_ff)

        scgco_results=read_result_to_dataframe(scgco_ff)


        scgco_arr=get_numbers(scgco_results,fdr01,"scGCO")
        print(scgco_arr)


        spatialde_ff="../spatialde_results/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+ "_spe.csv"
        print(spatialde_ff)

        spark_ff="../spark_results/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+ "_spark.csv"
        print(spark_ff)

        spatialde_results=pd.read_csv(spatialde_ff,index_col=0)


        spark_results=pd.read_csv(spark_ff,index_col=0)

        spark_arr=get_numbers(spark_results,fdr_cutoff,"SPARK")

        spatialde_arr=get_numbers(spatialde_results,fdr_cutoff,"spatialDE")


        scgco_list[i].append(scgco_arr)
        spatialde_list[i].append(spatialde_arr)
        spark_list[i].append(spark_arr)

scgco_df=pd.DataFrame(scgco_list)
scgco_df.index=["noise"+str(noise) for noise in [0.1,0.2,0.3,0.4,0.5,0.6] ]
scgco_df.columns=["rep"+str(i) for i in range(10)]


spatialde_df=pd.DataFrame(spatialde_list)
spatialde_df.index=["noise"+str(noise) for noise in [0.1,0.2,0.3,0.4,0.5,0.6] ]
spatialde_df.columns=["rep"+str(i) for i in range(10)]

spark_df=pd.DataFrame(spark_list)
spark_df.index=["noise"+str(noise) for noise in [0.1,0.2,0.3,0.4,0.5,0.6] ]
spark_df.columns=["rep"+str(i) for i in range(10)]


scgco_df.to_csv("../compare/scgco_numbers_noise.csv")
spatialde_df.to_csv("../compare/spatialde_numbers_noise.csv")
spark_df.to_csv("../compare/spark_numbers_noise.csv")