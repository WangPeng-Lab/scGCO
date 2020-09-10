import os
import sys
from scGCO import *
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

sample_info=pd.read_csv("../processed_data/Rep11_MOB_info_scgco.csv")

i=11
ff="../../data_science/MOB-breast-cancer/Rep" + str(i) +"_MOB_count_matrix-1.tsv"

def read_spatial_expression2(file,sep='\s+',num_exp_genes=0.01, num_exp_spots=0.01, min_expression=1):
    
    '''
    Read raw data and returns pandas data frame of spatial gene express
    and numpy ndarray for single cell location coordinates; 
    Meanwhile processing raw data.
    
    :param file: csv file for spatial gene expression; 
    :rtype: coord (spatial coordinates) shape (n, 2); data: shape (n, m); 
    '''
    counts = pd.read_csv(file, sep=sep, index_col = 0)
    print('raw data dim: {}'.format(counts.shape))

    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
    print("Number of expressed genes a spot must have to be kept " \
    "({}% of total expressed genes) {}".format(num_exp_genes, min_genes_spot_exp))
    #counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    mark_points=np.where((counts != 0).sum(axis=1) < min_genes_spot_exp)[0]
    print("Marked {} spots".format(len(mark_points)))

    temp = [val.split('x') for val in counts.index.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])

    similar_points=np.argsort(cdist(coord[mark_points,:],coord),axis=1)[:,1]
    for i,j in zip(mark_points,similar_points):
        counts.iloc[i,:]=counts.iloc[j,:]
   
    # Spots are columns and genes are rows
    counts = counts.transpose()
  
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots) 
    print("Removing genes that are expressed in less than {} " \
    "spots with a count of at least {}".format(min_features_gene, min_expression))
    counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
    print("Dropped {} genes".format(num_genes - len(counts.index)))
    
    data=counts.transpose()

    return coord,data

locs,data=read_spatial_expression2(ff)


def normalize(data):
    '''
    normalize count as in cellranger
    
    :param file: data: A dataframe of shape (m, n);
    :rtype: data shape (m, n);
    '''
    normalizing_factor = np.sum(data, axis = 1)/np.median(np.sum(data, axis = 1))
    data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    
    return data,normalizing_factor.values
data_norm,normalizing_factor=normalize(data)

tissue_mat=np.load("../processed_data/tissue_mat.npy")
tissue_mat=tissue_mat.astype("int")

def get_mu0(pattern,exp_diff,noise=0.2):
    high_exp=np.random.choice(range(2,5))
    low_exp= high_exp-exp_diff
    uu=[low_exp,high_exp]
    mu0=np.array([uu[i] for i in pattern])+np.random.normal(0,noise,len(pattern))
    mu0=np.clip(mu0,0,None)
    return mu0 
def get_mu1(num_cells,noise=0.2):
    noise=np.random.choice(np.arange(0,noise,step=0.1))+0.1
    mean_exp=np.random.choice(range(3))
    mu1=np.tile(mean_exp,num_cells)+np.random.normal(0,noise,num_cells)
    mu1=np.clip(mu1,0,None)
    return mu1

def get_counts(mu,normalizing_factor):
    return (np.expm1(mu)*normalizing_factor+0.5).astype("int")

    
for noise in [0.1,0.2,0.3,0.4,0.5,0.6]:
    for irep in range(10):
        np.random.seed(irep)
        exp_diff=1
        num_genes=10000
        data_subset=data.iloc[:,:10000].copy()

        for i in range(150):
            c=tissue_mat[0]
            mu0=get_mu0(c,exp_diff,noise)
            data_subset.iloc[:,i]=get_counts(mu0,normalizing_factor)
        for i in range(150,850):
            c=tissue_mat[1]
            mu0=get_mu0(c,exp_diff,noise)
            data_subset.iloc[:,i]=get_counts(mu0,normalizing_factor)
        for i in range(850,1000):
            c=tissue_mat[2]
            mu0=get_mu0(c,exp_diff,noise)
            data_subset.iloc[:,i]=get_counts(mu0,normalizing_factor)
        #gaussian noise
        for i in range(1000,10000):
            mu1=get_mu1(data_subset.shape[0],noise=noise)
            data_subset.iloc[:,i]=get_counts(mu1,normalizing_factor)
        data_subset.columns=["gene"+str(i+1) for i in range(10000)]
        df=data_subset.copy() 
        ff="../processed_data/sim_MOB_expdiff"+str(exp_diff)+"_noise"+str(noise)+"_counts"+str(irep)+".csv"
        df.to_csv(ff)

