import multiprocessing
import logging
import math
import random
import parmap
import itertools
from itertools import repeat
from scipy.spatial import distance
import operator
from tqdm import tqdm
from functools import reduce
from sklearn import mixture
import statsmodels.stats.multitest as multi
import networkx as nx
import multiprocessing as mp
import numpy as np
from scipy.stats import poisson
import pandas as pd
import pygco as pygco # cut_from_graph # pip install git+git://github.com/amueller/gco_python
import scipy.stats.mstats as ms
from itertools import repeat
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import skewnorm
from scipy.sparse import issparse
import matplotlib as mpl
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapely.geometry
import shapely.ops
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, KDTree, ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition 
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN,KMeans
from scipy.stats import ttest_ind
from sklearn.utils import shuffle
import hdbscan
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import stats
import sys
import os
import time
from scipy.stats import skewnorm
from scipy import stats
from ast import literal_eval
from .Preprocessing import *
from .Graph_cut import *
from .Visualization import *


def check_target_status_jaccard(p, node, cand_seed_mat, cand_seed_size, hamming_cutoff = 0.6):   ### cutoff 0.5-0.7
#geneID = 'Ccnl1'
#newLabels=result_df.loc[geneID,][4:].values.astype(int)
#p=result_df.loc[geneID,'p_value']
#node=result_df.loc[geneID,'nodes']
#gmm = gmmDict[geneID]
#hamming_cutoff = 0.6
#if True:
    target_mat_list = list()
    for p_index in np.where(np.array(p) < 0.01)[0]:
        temp_vec = np.zeros(cand_seed_mat.shape[1])
        temp_vec[node[p_index]] = 1  
        if len(node[p_index]) <= len(temp_vec)/2:
            target_mat_list.append(temp_vec) 
 
    if len(target_mat_list) > 0:
        target_mat = np.array(target_mat_list)
        overlap_hdist = cdist(cand_seed_mat, target_mat, jaccard_dist)
#        print(np.min(overlap_hdist, axis=1))
        hit_pos = np.where(np.min(overlap_hdist, axis=1) < hamming_cutoff)[0]
        unique_hit, count_hit = np.unique(np.array(cand_seed_size)[hit_pos], return_counts = True)
        if len(target_mat_list) >= 3:
            if sum(count_hit >= len(target_mat_list)-1) >  0:
#            print('good')
                return True
            else:
                return False
        else:
            if sum(count_hit >= len(target_mat_list)-1) >  0:
                return True
            else:
                return False      
    else:
        return False

def compute_diff_vs_common_const10(u, v):
    return len(np.where((u+v) == 1)[0])/(2*len(np.where((u+v) == 2)[0]) + 10)

def jaccard_dist(u, v):
    return  1 - len(np.where((u+v) == 2)[0])/(len(np.where((u+v) > 0)[0]))


def generate_target_shape_mat(data_norm, result_df, fdr_opt):
    
    '''
    Generating target mat from these genes that needed comparing pattern.
    
    ''' 
    
    target_gene_list = list()
    target_mat_list = list()
    missed_gene = list() # save genes that are not in data_norm; and just extract from origianl data
    for geneID in fdr_opt.index.values:
        if geneID in result_df.index:
            p = result_df.loc[geneID].p_value
            node = result_df.loc[geneID].nodes
            target_gene_list.append(geneID)
            target_list = list()
            for p_index in np.arange(len(p)): 
#                if len(node[p_index]) < fdr_opt.shape[1]/2:
                temp_vec = np.zeros(data_norm.shape[0])
                temp_vec[node[p_index]] = 1  
                target_list.append(temp_vec)    
            target_mat = np.asarray(target_list)      
            target_mat_list.append(target_mat)
        else:
            missed_gene.append(geneID)
    target_df = pd.DataFrame([target_gene_list, target_mat_list]).T        
    target_df.columns = ['geneID', 'mat']
    target_df.index =  target_df.geneID     
        
    return target_df, missed_gene


def compute_p_reverLabels(locs,newLabels,gmm,exp,cellGraph):
    '''
    compute p values for each subgraph, and reverse labels of bad subgraph.
    return new reversal labels.
    '''
    
    com_factor = 1
    p_values = list()
    node_lists = list()
    reverLabel=newLabels.copy()
    a= exp[exp > 0]
    gmm_pred = gmm.predict(a.reshape(-1,1))
    zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
    label_pred = gmm.predict(exp.reshape(-1,1))
  
    if len(np.where(exp == 0)[0]) > 0:
        np.place(label_pred, exp==0, zero_label)
    gmm_pred =label_pred
    
    unique, counts = np.unique(gmm_pred,return_counts=True)
    con_components = count_component(locs,cellGraph, newLabels)
    
    for j in np.arange(len(con_components)):  ## all subgraphs are normal,score all
        if len(con_components[j]) <= 9: # len(exp)/2+10:  ## too big pattern is not reverse labels
            node_list = con_components[j]
            com_size = len(node_list)
            temp_exp = exp[np.array(list(node_list))]
            gmm_pred_com = gmm.predict(temp_exp.reshape(-1,1))
            if sum(temp_exp == 0) > 0:
#                 a=temp_exp[temp_exp>0]
#                 zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
                np.place(gmm_pred_com, temp_exp==0, zero_label)           

    # check 0s
            unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
            major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]

            label_count = counts[np.where(unique == major_label)[0]]
            count_in_com =  counts_com.max()
            cover = exp.shape[0]/com_size

            pmf= poisson.pmf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
            psf= poisson.sf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
            prob_sf= (pmf+psf)*cover
            prob = (1 - poisson.cdf(count_in_com, com_size*(label_count/exp.shape[0]))[0])*cover
    #         if prob>=1:
    #             prob=1
          #  p_values.append(prob)
            #print(prob_sf,prob,major_label,com_size,cover)
          #  node_lists.append(np.array(list(node_list)))
#             for i in node_list:
#                 plt.text(locs[i,0],locs[i,1],str(i))
#             plot_voronoi_boundary(geneID,locs,exp,newLabels,prob)

            ## 1. find connected pattern of noise pattern
            if prob_sf >0.1:  
                test_node=node_list
                node_otherPattern=[]
                for i in test_node:
                    if i in cellGraph[:,0:2]:
                        node = cellGraph[np.where(i==cellGraph[:,0:2])[0],0:2]
                        node_array= node.reshape(1,-1)[0]
                        other_node= [int(val) for val in list(set(node_array)-set((test_node)))] 
                        node_otherPattern.extend(other_node)
                   #     print(i, ' ',node_array,other_node)

                node_otherPattern=np.unique(node_otherPattern)
              #  print(test_node,node_otherPattern) 
                ## 2. judege nodes of connencted other pattern in one same pattern
                diff_con=[]
                for con in range(len(con_components)):
                    if con !=j:
                        diff=set(node_otherPattern)-set(con_components[con])     
                        if len(diff)==0:
                            diff_con.append(con)
                if len(diff_con)==1:
                        #reverLabel=newLabels.copy()
                        rever =newLabels[list(con_components[diff_con[0]])][0]
                        reverLabel[list(test_node)]=rever
                    #    plot_voronoi_boundary(geneID,locs,exp,reverLabel,prob)
    return reverLabel

### 20190930 distance

def calc_distance_df( locs, data_norm,cellGraph, gmmDict, fdr_df, tissue_mat_new,sort_tissue=False):
    
    hamming=list()
    jaccard = list()
    hausdorff = list()
    
    recal_genes=[]
    genelist=[]
    for geneID in fdr_df.index:
        exp =  data_norm.loc[:,geneID].values
        gmm=gmmDict[geneID]
        target_seed_mat_list = list()
   
        temp_factor = int(fdr_df.loc[geneID].smooth_factor)  
        newLabels= fdr_df.loc[geneID][4:].values.astype(int)
        node= fdr_df.loc[geneID,'nodes']
        p= fdr_df.loc[geneID,'p_value']
        
        num_isolate = count_isolate(locs,cellGraph, newLabels)
        num_noise=int((num_isolate[0]+num_isolate[1])*1+(num_isolate[2]+num_isolate[3])*2+(num_isolate[4]+num_isolate[5])*3)
       # print(num_noise,len(node[0]),(locs.shape[0]-num_noise),p)

               
        if len(p)==1 and len(node[0])==locs.shape[0]:
            recal_genes.append(geneID)  ## no cuts genes, can't calculate hausdorff
                  
#         elif len(p)==1 and min(p)>0.1 and len(node[0])==(locs.shape[0]-num_noise):  ## all noise,like    
#             genelist.append(geneID)
#             total_tissue_mat=np.sum(tissue_mat_new,axis=0)
#             total_target_mat=newLabels
            
#             hamming.append(compute_diff_vs_common_const10(total_tissue_mat , total_target_mat))
#             jaccard.append(jaccard_dist(total_tissue_mat , total_target_mat))   
#             target_locs = locs[total_target_mat == 1]
#             tissue_locs = locs[total_tissue_mat == 1]
#             hausdorff.append(compute_hausdorff(target_locs,tissue_locs))
        
        else: 
            genelist.append(geneID)
            gmm_pred = gmm.predict(exp.reshape(-1,1))
            a= exp[exp > 0]
            zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
            if sum(exp == 0) > 0:
                np.place(gmm_pred, exp==0, zero_label)    

    #        print(major_label, unique_com, counts_com)
    #        plot_voronoi_boundary(geneID, locs, exp,  newLabels, min(p))

    # get major label in min(p) segment
            target_seed_mat_list=list()
            node_list = node[np.argmin(p)]
            temp_vec = np.zeros(len(exp))
            temp_vec[node_list] = 1  
            target_seed_mat_list.append(temp_vec)
            # for nn in node[np.argmin(p)]:
            #     plt.text(locs[nn,0],locs[nn,1],str(nn)) 
            # plot_voronoi_boundary(geneID, locs,exp,newLabels,min(p))

            com_size = len(node_list)
            temp_exp = exp[np.array(list(node_list))]
            gmm_pred_com = gmm.predict(temp_exp.reshape(-1,1))
            if sum(temp_exp == 0) > 0:
                # a= temp_exp[temp_exp > 0]
                # zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
                np.place(gmm_pred_com, temp_exp==0, zero_label)  
            unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
            major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]

            max_size = 0
            size=[len(node[nn]) for nn in range(len(node))]
            p_sort=[p[i] for i in np.argsort(size)]
            node_sort=[node[i] for i in np.argsort(size)]
            #p_sort
            for p_index in np.arange(len(p)): #np.where(np.array(p) < 0.01)[0]:
                if p_sort[p_index]>min(p):  #p[p_index] <= p_cutoff and 
                    temp_vec = np.zeros(len(exp))
                    node_list = node_sort[p_index]
                    temp_exp = exp[np.array(list(node_list))]
                    gmm_pred_com = gmm.predict(temp_exp.reshape(-1,1))
                    if sum(temp_exp == 0) > 0:
                        # a= temp_exp[temp_exp > 0]
                        # zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]  
                        np.place(gmm_pred_com, temp_exp==0, zero_label) 
                    unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
                    temp_major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
               #     print(temp_major_label)
                    if temp_major_label == major_label :  ## get other pattern with major label
                          #['Psap'] in Rep8
                        if len(node_list)+sum(np.sum(target_seed_mat_list,axis=0))< (2/3)*len(exp)+10:
                            temp_vec[node_list] = 1  
                            target_seed_mat_list.append(temp_vec)                      
                            if len(node_list) > max_size:
                                max_size = len(node_list)
                              #  print(max_size)
                        else:
                            if sum(target_seed_mat_list[0])>max_size:
                                max_size = sum(target_seed_mat_list[0])
                    else:
                        if sum(target_seed_mat_list[0])>max_size:
                            max_size = sum(target_seed_mat_list[0])  ## sum(np.sum(target_seed_mat_list,axis=0))
                else:
                    if sum(target_seed_mat_list[0])>max_size:
                        max_size = sum(target_seed_mat_list[0])

                temp_list = list()
                if max_size > 5:
                    for nn in target_seed_mat_list:
                        if sum(nn==1) > 5 :
                            temp_list.append(nn)
                else:
                    for nn in target_seed_mat_list:
                        if sum(nn==1) == max_size:
                            temp_list.append(nn)         

            target_mat = np.array(temp_list)  
            total_target_mat = np.sum(target_mat, axis=0)  


        ## effect target mat: merge target mat as same label with min(p)   
            if sum(total_target_mat) > len(exp)/2+10:
                total_target_mat = abs(1 - total_target_mat)  ##  1-total_target_mat, for 'Hk3'


        # now find all tissue mat that with 50% of the tissue mat seg are in target
            temp_tissue_seg_list = list()
            best_overlap = list()
            best_overlap_val = 0
            match_index=[]
            for ts_index in np.arange(len(tissue_mat_new)):
            #             overlap = max(sum((tissue_mat_new[kk,:] + total_target_mat) == 2)/sum(tissue_mat_new[kk,:]), 
            #                      sum((tissue_mat_new[kk,:] + total_target_mat) == 2)/sum(total_target_mat))        
                overlap= compute_inclusion_min(tissue_mat_new[ts_index],total_target_mat)
              #  print(ts_index, overlap)
                if overlap >0.45:
                    match_index.append(ts_index)
                    temp_tissue_seg_list.append(tissue_mat_new[ts_index])


                if overlap >= best_overlap_val : #and best_overlap_val>=0.1 :
                  #  match_index.append(ts_index)
                    best_overlap_val = overlap
                    best_overlap=tissue_mat_new[ts_index].reshape(1,-1)

            if len(temp_tissue_seg_list) >0:    
                if sort_tissue==True:    
                    if len(temp_tissue_seg_list)==1 and match_index[0] > 3:
                        if match_index[0]==4 or match_index[0]==6:
                            temp_tissue_seg_list.append(tissue_mat_new[5])
                        if match_index[0]==5:
                            temp_tissue_seg_list.append(tissue_mat_new[4])
                            temp_tissue_seg_list.append(tissue_mat_new[6])
                        match_tissue_mat=np.array(temp_tissue_seg_list)
                    else:
                        match_tissue_mat = np.array(temp_tissue_seg_list)
                else:
                    match_tissue_mat = np.array(temp_tissue_seg_list)
            else:
                if best_overlap_val>0:
                    match_tissue_mat = np.array(best_overlap)
            ## 2. some points are missed by tissue mat, using min(hausdorff) to match tissue mat.['Postn'] in Rep8
                else:             ## all(overlap==0)
                    dist_temp=[]                                    
                    temp_vec=np.zeros(locs.shape[0])
                    temp_vec[:]=1
                    temp_vec[node[0]]=0  ## mostly nodes
                    u=locs[np.where(temp_vec==1)]
                    for ts_index in range(len(tissue_mat_new)):
            #                             ja_dist=jaccard_dist(temp_vec , tissue_mat_new[ts_index]) ## single noise not in any tissue mat.
            #                             ja_dist_temp.append(ja_dist)

                        v=locs[np.where(tissue_mat_new[ts_index]==1)]
                        dist=compute_hausdorff(u,v)
                        dist_temp.append(dist)
                        temp_index=np.argmin(dist_temp)
                        match_tissue_mat=tissue_mat_new[temp_index].reshape(1,-1)


            total_tissue_mat = np.sum(match_tissue_mat, axis=0) ## get tissue_mat that matched target 
            
            hamming.append(compute_diff_vs_common_const10(total_tissue_mat , total_target_mat))
            jaccard.append(jaccard_dist(total_tissue_mat , total_target_mat))   

            target_locs = locs[total_target_mat == 1]
            tissue_locs = locs[total_tissue_mat == 1]
            hausdorff.append(compute_hausdorff(target_locs,tissue_locs))
    #         print(geneID, len(target_seed_mat_list), sum(total_target_mat), 
    #               len(temp_tissue_seg_list), sum(total_tissue_mat), jaccard_dist(total_tissue_mat , total_target_mat),
    #               max(directed_hausdorff(tissue_locs, target_locs)[0], 
    #                              directed_hausdorff(target_locs, tissue_locs)[0]))
    #         plot_voronoi_boundary(geneID, locs, exp,  total_target_mat, min(p))
            #plot_voronoi_boundary(geneID, locs, exp,  total_tissue_mat, min(p))   
    #         print('------------------')
    data_array = np.array((hamming, jaccard, hausdorff), dtype=float).T
    c_labels = ['Hamming', 'Jaccard', 'Hausdorff']

    dist_df = pd.DataFrame(data_array, index=genelist, 
                      columns=c_labels)
# print out the target sum mat, and matched tissue sum mat to see whether they make sense
    return dist_df,recal_genes



def compute_inclusion_min(u, v):
#    if len(np.where((u+v) == 2)[0]) == 0:
#        return len(np.where(abs(u-v) == 1)[0])
#    else:
    return len(np.where((u+v) == 2)[0])/min(sum(v),sum(u))

def compute_hausdorff(u,v):
    '''
    Compute norm hausdorff geometry distance between two pattern.
    u: target_mat
    v: tissue_mat
    
    '''
    
    dist_u_v=distance.directed_hausdorff(u,v)[0]
    dist_v_u=distance.directed_hausdorff(v,u)[0]
    
    # if norm_factor:
    #     dist= max(dist_u_v/compute_norm_factor(v),dist_v_u/compute_norm_factor(u))
    # else:
     
        
    return max(dist_u_v,dist_v_u)


### calculate distance

def Estimate_sf_And_dist(locs,data, cellGraph, gmmDict, test_genes, tissue_mat,plot=False,
                           unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    
    '''
    Relationship smooth factor with distance for estimating smooth factor.
    the function just like recal_dist_to_tissue() with tissue_mat,
    but getting best sf by using min(hamming_dist).
    Return Best_smooth_factor, related size of gmm labels' components and three distance.
    
    '''
  
    result_df_new=pd.DataFrame()
    new_hamming=[]
    new_jaccard=[]
    new_hausdorff=[]
    con_list=[]

    for geneID in test_genes: #zeor_boundGenes: #de_counts:
        #if geneID not in result_df.index:
        count =  data.loc[:,geneID].values
        gmm=gmmDict[geneID]
        Labels = gmm.predict(count.reshape(-1,1))
        zero_label=gmm.predict(np.min(count[count>0]).reshape(-1,1))
        if sum(count==0):
            np.place(Labels,count==0, zero_label)
        con= count_component(locs, cellGraph, Labels)
        con_list.append(len(con))

        hamming_dist_list=[]
        jaccard_dist_list=[]
        hausdorff_dist_list=[]
        test_df_list=pd.DataFrame()
        
        #figsize(10,8)
        for sf in np.arange(0,110,10):  # 'Cldn9' need start from sf=0 
            temp_factor=sf
            newLabels = cut_graph_general(cellGraph, count ,gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)

            #print(geneID)
            reverLabels=  compute_p_reverLabels(locs,newLabels,gmm,count,cellGraph)
            p,node, com = compute_p_CSR(locs, reverLabels, gmm, count, cellGraph) 
            
           # figsize(8,6)
            #print(sf,p)
            

            labels_array = np.array(reverLabels).reshape(1, -1)
            data_array = np.array((geneID, p, min(p),temp_factor, node), dtype=object).reshape(1,-1)
            t_array = np.hstack((data_array, labels_array))
            c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes']
            for i in np.arange(labels_array.shape[1]) + 1:
                temp_label = 'label_cell_' + str(i)
                c_labels.append(temp_label)
            test_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                              columns=c_labels)

            dist_df,false_genes=calc_distance_df( locs,data,cellGraph, gmmDict,test_df, tissue_mat)
            if plot==True:
                plot_voronoi_boundary(geneID,locs,count,reverLabels,min(p))
                print(sf,p,dist_df)

            hamming_dist_list.append(dist_df.iloc[:,0].values)
            jaccard_dist_list.append(dist_df.iloc[:,1].values)
            hausdorff_dist_list.append(dist_df.iloc[:,2].values)
            test_df_list=pd.concat([test_df_list,test_df])
            
        if len(jaccard_dist_list)>0:
            min_inde =np.argmin(hamming_dist_list)  ## multiple min(p) select min(hamming_dist) amony
            
            new_hamming.append(hamming_dist_list[min_inde][0])
            new_jaccard.append(jaccard_dist_list[min_inde][0])
            new_hausdorff.append(hausdorff_dist_list[min_inde][0])
            best_test_df=test_df_list.iloc[min_inde:min_inde+1]
            result_df_new=pd.concat([result_df_new,best_test_df])
       
    
    best_sf=result_df_new.smooth_factor.values
    dist_df_new = pd.DataFrame([best_sf,con_list, new_hamming, new_jaccard,new_hausdorff]).T        ### no norm dist
    dist_df_new.columns = ['smooth_factor','con_size', 'Hamming', 'Jaccard','Hausdorff']
    dist_df_new.index =  result_df_new.index
    
    return result_df_new, dist_df_new




def recalc_dist_to_tissue(locs,data_norm, cellGraph, gmmDict, test_genes, tissue_mat_new,
                           unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    
    size_factor=200
    result_df_new=pd.DataFrame()
    new_hamming=[]
    new_jaccard=[]
    new_hausdorff=[]
   
    
    noise_size_estimate =9 # min(np.min(np.sum(cand_seed_mat, axis=1)), 9)  ## #noise<=9
    # print(noise_size_estimate)
    for geneID in test_genes: #zeor_boundGenes: #de_counts:
        #if geneID not in result_df.index:
        exp =  data_norm.loc[:,geneID].values
        gmm=gmmDict[geneID]
        #print(geneID,len(np.where(exp > 0)[0]))

        test_dist_list=[]
        hamming_dist_list=[]
        jaccard_dist_list=[]
        hausdorff_dist_list=[]
        p_min=[]
        test_df_list=pd.DataFrame()
       
        #noise_cutoff=np.inf
        
        temp_factor = 0   # 'Cldn9' need start from sf=0 
        newLabels = cut_graph_general(cellGraph, exp,gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)

        p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)
        num_isolate = count_isolate(locs,cellGraph, newLabels)  

        noise_size = sum(num_isolate[0:int(2*noise_size_estimate)]) 
        normalized_noise = noise_size*(size_factor/len(exp))
        noise_cutoff=normalized_noise
        
        
        labels_array = np.array(newLabels).reshape(1, -1)
        data_array = np.array((geneID, p, min(p),temp_factor, node), dtype=object).reshape(1,-1)
        t_array = np.hstack((data_array, labels_array))
        c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes']
        for i in np.arange(labels_array.shape[1]) + 1:
            temp_label = 'label_cell_' + str(i)
            c_labels.append(temp_label)
        test_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                          columns=c_labels)

        dist_df,false_genes=calc_distance_df( locs, data_norm,cellGraph, gmmDict,test_df, tissue_mat_new)
     #   print(geneID,temp_factor,noise_cutoff,dist_df,false_genes)
        
        if noise_cutoff<6 and noise_cutoff>0:  ## 'Vstm4' have no 0<noise<5 , just skip 5.
#                     print(dist_df)
#                     print('---')
            hamming_dist_list.append(dist_df.iloc[:,0].values)
            jaccard_dist_list.append(dist_df.iloc[:,1].values)
            hausdorff_dist_list.append(dist_df.iloc[:,2].values)
            p_min.append(min(p))
            test_df_list=pd.concat([test_df_list,test_df])
            
        while noise_cutoff >0 and len(false_genes)==0 and temp_factor<=80:
      #  for sf_add in np.arange(10,add_sf,10):   
            
            temp_factor = temp_factor+5
            newLabels = cut_graph_general(cellGraph, exp,gmm, unary_scale_factor, 
                                               temp_factor, label_cost, algorithm)

            p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)
            num_isolate = count_isolate(locs,cellGraph, newLabels)  
            
            noise_size = sum(num_isolate[0:int(2*noise_size_estimate)]) 
            normalized_noise = noise_size*(size_factor/len(exp))
            
            noise_cutoff=normalized_noise
           # print(geneID,temp_factor,noise_cutoff)
            #plot_voronoi_boundary(geneID,locs,exp,newLabels,min(p))
            
            labels_array = np.array(newLabels).reshape(1, -1)
            data_array = np.array((geneID, p, min(p),temp_factor, node), dtype=object).reshape(1,-1)
            t_array = np.hstack((data_array, labels_array))
            c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes']
            for i in np.arange(labels_array.shape[1]) + 1:
                temp_label = 'label_cell_' + str(i)
                c_labels.append(temp_label)
            test_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                              columns=c_labels)

            dist_df,false_genes=calc_distance_df( locs, data_norm,cellGraph, gmmDict,test_df, tissue_mat_new)
            
            
            if noise_cutoff<6 and noise_cutoff>0:
#                     print(dist_df)
#                     print('---')
                hamming_dist_list.append(dist_df.iloc[:,0].values)
                jaccard_dist_list.append(dist_df.iloc[:,1].values)
                hausdorff_dist_list.append(dist_df.iloc[:,2].values)
                p_min.append(min(p))
                test_df_list=pd.concat([test_df_list,test_df])
                    
#             if len(false_genes)!=0: #or noise_cutoff==0:
#                 print('false_genes:',false_genes)
#                 break;


        if len(hausdorff_dist_list)>0:
            best_p=min(p_min)  ## multiple min(p) select min(jaccard_dist) amony
            best_p_index=np.where(np.array(p_min)==best_p)[0]
            min_p_haus=[hausdorff_dist_list[i] for i in best_p_index]
            min_inde=np.where(np.array(hausdorff_dist_list)==min(min_p_haus))[0][0]
            #min_inde=np.argmin(p_min)  #jaccard_dist_list)   ## min(p) as selection standard.
            #min_inde_haus=np.argmin(hausdorff_dist_list)
            new_hamming.append(hamming_dist_list[min_inde][0])
            new_jaccard.append(jaccard_dist_list[min_inde][0])
            new_hausdorff.append(hausdorff_dist_list[min_inde][0])
            best_test_df=test_df_list.iloc[min_inde:min_inde+1]
            result_df_new=pd.concat([result_df_new,best_test_df])

    dist_df_new = pd.DataFrame([new_hamming, new_jaccard,new_hausdorff]).T        ### no norm dist
    dist_df_new.columns = ['Hamming', 'Jaccard','Hausdorff']
    dist_df_new.index =  result_df_new.index

    return result_df_new, dist_df_new




## abundan
def computer_norm_jaccard_to_tissue(tissue_mat, target_df, fdr_opt):
    '''
    computer jaccard distance
    
    '''
    jaccard_result = list()
    temp_result_df = fdr_opt[fdr_opt.fdr < 0.01].sort_values(by=['fdr'])
    p_cutoff = min(temp_result_df.iloc[temp_result_df.shape[0]-1].p_value)
#    p_cutoff = min(fdr_opt.p_value[int(np.where(fdr_opt.fdr == max(fdr_opt.fdr))[0])])
    for geneID in target_df.index:
        p = fdr_opt.loc[geneID].p_value
        node = fdr_opt.loc[geneID].nodes
        sizes=list()
        for nn in node:
            sizes.append(len(nn))

        temp_mat = target_df.loc[geneID].mat
        overlap_hdist = cdist(tissue_mat, temp_mat, compute_inclusion_new)
    #    mapping_index_c = np.argmax(overlap_hdist, axis=0)
        mapping_index_r = np.argmax(overlap_hdist, axis=1)
        
        jaccard_val = list()
        for ts_index in np.arange(tissue_mat.shape[0]):
            match_index = np.where(overlap_hdist[ts_index] > 0.6)[0]
            # use small seg first, use same number of seg as seed:
            # until size match
            if len(match_index) > 0:
                temp_vec= np.zeros(tissue_mat.shape[1])
                for m_index in match_index:
                    if sum(temp_vec) < 1.2*sum(tissue_mat[ts_index]):
                        temp_vec = temp_vec + temp_mat[m_index]
                temp_jaccard = distance.jaccard(tissue_mat[ts_index], temp_vec)
                jaccard_val.append(temp_jaccard)         
        p = fdr_opt.loc[geneID].p_value
        node = fdr_opt.loc[geneID].nodes
        if geneID == 'Fgfr1':
            print(p_cutoff, p)
        for p_index in np.arange(len(p)):
            if p[p_index] <= p_cutoff:
                temp_vec= np.zeros(tissue_mat.shape[1])
                temp_vec[node[p_index]] = 1
                for ts_index in np.arange(tissue_mat.shape[0]):
                    temp_jaccard = distance.jaccard(tissue_mat[ts_index], temp_vec) #,compute_jaccard)
                    jaccard_val.append(temp_jaccard)  
                    if geneID == 'Fgfr1':
                        print(p_cutoff, p)
                        print(p_index, ts_index, sum(temp_vec), sum(tissue_mat[ts_index]), temp_jaccard)

                        
        if len(jaccard_val) == 0:
            ts_index = np.argmin(np.sum(tissue_mat, axis=1))
            temp_vec= np.zeros(tissue_mat.shape[1])
            one_index = random.sample(range(0, tissue_mat.shape[1]), sum(tissue_mat[ts_index]).astype(np.int))
            temp_vec[one_index] = 1
            temp_jaccard = distance.jaccard(tissue_mat[ts_index], temp_vec) #,compute_jaccard)# temp_hamming = 10 #random.uniform(2.5, 4)   #compute_diff_vs_common_new(tissue_mat[ts_index], temp_vec)
            jaccard_val.append(temp_jaccard)
        jaccard_result.append(min(jaccard_val))
    jaccard_df = pd.DataFrame([target_df.index, jaccard_result]).T        
    jaccard_df.columns = ['geneID', 'jaccard_dist']
    jaccard_df.index =  target_df.index
    return jaccard_df

def compute_inclusion_new(u, v):
#    if len(np.where((u+v) == 2)[0]) == 0:
#        return len(np.where(abs(u-v) == 1)[0])
#    else:
        return len(np.where((u+v) == 2)[0])/sum(v)


## abundan
def computer_norm_hamming_to_tissue(tissue_mat, target_df, fdr_opt):
    '''
    computer hamming distance
    '''

    hamming_result = list()
    temp_result_df = fdr_opt[fdr_opt.fdr < 0.01].sort_values(by=['fdr'])
    p_cutoff = min(temp_result_df.iloc[temp_result_df.shape[0]-1].p_value)
    for geneID in target_df.index:
        p = fdr_opt.loc[geneID].p_value
        node = fdr_opt.loc[geneID].nodes
        sizes=list()
        for nn in node:
            sizes.append(len(nn))
        temp_mat = target_df.loc[geneID].mat
        overlap_hdist = cdist(tissue_mat, temp_mat, compute_inclusion_new)
    #    mapping_index_c = np.argmax(overlap_hdist, axis=0)
        mapping_index_r = np.argmax(overlap_hdist, axis=1)
        
        hamming_val = list()
        for ts_index in np.arange(tissue_mat.shape[0]):
            match_index = np.where(overlap_hdist[ts_index] > 0.6)[0]
            # use small seg first, use same number of seg as seed:
            # until size match
            if len(match_index) > 0:
                temp_vec= np.zeros(tissue_mat.shape[1])
                for m_index in match_index:
                    if sum(temp_vec) < 1.2*sum(tissue_mat[ts_index]):
                        temp_vec = temp_vec + temp_mat[m_index]
                temp_hamming = compute_diff_vs_common_new(tissue_mat[ts_index], temp_vec)
                hamming_val.append(temp_hamming)         
        
        if geneID == 'Fgfr1':
            print(p_cutoff, p)
        for p_index in np.arange(len(p)):
            if p[p_index] <= p_cutoff:
                temp_vec= np.zeros(tissue_mat.shape[1])
                temp_vec[node[p_index]] = 1
                for ts_index in np.arange(tissue_mat.shape[0]):
                    temp_hamming = compute_diff_vs_common_new(tissue_mat[ts_index], temp_vec)
                    hamming_val.append(temp_hamming)  
                    if geneID == 'Fgfr1':
                        print(p_cutoff, p)
                        print(p_index, ts_index, sum(temp_vec), sum(tissue_mat[ts_index]), temp_hamming)

                        
        if len(hamming_val) == 0:
            ts_index = np.argmin(np.sum(tissue_mat, axis=1))
            temp_vec= np.zeros(tissue_mat.shape[1])
            one_index = random.sample(range(0, tissue_mat.shape[1]), sum(tissue_mat[ts_index]).astype(np.int))
            temp_vec[one_index] = 1
            temp_hamming = compute_diff_vs_common_new(tissue_mat[ts_index], temp_vec)# temp_hamming = 10 #random.uniform(2.5, 4)   #compute_diff_vs_common_new(tissue_mat[ts_index], temp_vec)
            hamming_val.append(temp_hamming)
        hamming_result.append(min(hamming_val))
    hamming_df = pd.DataFrame([target_df.index, hamming_result]).T        
    hamming_df.columns = ['geneID', 'hamming']
    hamming_df.index =  target_df.index
    return hamming_df

def compute_diff_vs_common_const_new(u, v):
    return len(np.where((u+v) == 1)[0])/(2*len(np.where((u+v) == 2)[0]) + 10)

def compute_diff_vs_common_new(u, v):
    return len(np.where((u+v) == 1)[0])/(2*len(np.where((u+v) == 2)[0]) + 10)



## hausdorff distance

def compute_norm_dist_to_tissue_parallel(locs, tissue_mat_new, target_df, result_df):
    
    num_cores = mp.cpu_count()
    if num_cores > math.floor(target_df.shape[0]/2):
         num_cores=int(math.floor(target_df.shape[0]/2))
    ttt = np.array_split(target_df,num_cores,axis=0)
    tuples = [(l, d, u,  c) for l, d, u, c in zip(repeat(locs, num_cores),
                                                        repeat(tissue_mat_new, num_cores),
                                                        ttt,     
                                    repeat(result_df, num_cores))]
                                   # repeat(p_cutoff, num_cores))] 
                                    
    dist_results = parmap.starmap(compute_norm_dist_to_tissue, tuples,
                             pm_processes=num_cores, pm_pbar=True)
    
    dd=[dist_results[i][0] for i in range(len(dist_results))]
    dist_val=reduce(operator.add,dd)
    
    gg=[dist_results[i][1] for i in range(len(dist_results))]
    genes=reduce(operator.add,gg)
    
    re=[dist_results[i][2] for i in range(len(dist_results))]
    recal_genes=reduce(operator.add,re)
    
    dist_norm=[val /max(dist_val) for val in dist_val]
    dist_df = pd.DataFrame([genes, dist_val,dist_norm]).T        
    dist_df.columns = ['geneID', 'dist','norm_dist']
    dist_df.index =  genes
    
    return dist_df,recal_genes

    
def compute_norm_dist_to_tissue(locs,tissue_mat_new,target_df,result_df):
    
    '''
    u: targrt vector
    v: tissue vector
    '''
    
    dist_result=[]
    recal_genes=[]
    genelist=[]
    for geneID in target_df.index:
        
        #exp=data_norm.loc[:,geneID].values
        fdr = result_df.loc[geneID].fdr
        p=result_df.loc[geneID].p_value
        node = result_df.loc[geneID].nodes
        newLabels=result_df.loc[geneID][4:].values.astype(int)
        sf=result_df.loc[geneID,'smooth_factor']
        #plot_voronoi_boundary(geneID,locs,exp,newLabels,p=fdr)

       
        ### 1. sum(exp>0)<3,just one or two noise, no in target df, comparing noise and all tissue patterns.
        ###  no exist len(p)==1 and len(node[0])<3. 
        # geneID='Gm16532'
        if len(p)==1 and len(node[0])>locs.shape[0]-3 and len(node[0])<locs.shape[0]: 
            genelist.append(geneID)
            dist_temp=[]
            temp_vec=np.zeros(locs.shape[0])
            temp_vec[:]=1
            temp_vec[node[0]]=0
            u=locs[np.where(temp_vec==1)]
            for ts_index in range(len(tissue_mat_new)):
                v=locs[np.where(tissue_mat_new[ts_index]==1)]
                dist=compute_hausdorff(u,v)
                dist_temp.append(dist)
            dist_result.append(min(dist_temp))
       # dist_result.append(dist_val)
            # zero_boundGenes.append(geneID)
            
            # temp_hull=ConvexHull(locs)
            # u=locs[temp_hull.simplices.reshape(1,-1)[0]] 

            # dist_temp=[]
            # for ts_index in np.arange(len(tissue_mat_new)):
            #     v=locs[np.where(tissue_mat_new[ts_index]==1)]
            #     norm_factor=compute_norm_factor(v)
            #     dist=distance.directed_hausdorff(u,v)[0]
            #     dist_norm=dist/norm_factor
            #     # print(ts_index,norm_factor,dist,dist_norm)
            #     if dist_norm>1:
            #         dist_norm=1
            #     dist_temp.append(dist_norm)
            # dist_val=max(dist_temp)

        ## 2. no boundarys (with big smooth factor) , recalc with a series of small sf, not in target df.
        # geneID='Map1a'
        elif len(p)==1 and len(node[0])==locs.shape[0]:
            #dist_temp=[]
            recal_genes.append(geneID)

        ## 3. normal graph cuts, each target match to tissue.                
                
        else:
            genelist.append(geneID)
            dist_temp=[]            
            temp_mat=target_df.loc[geneID,'mat']
            
            overlap_hdist=cdist(tissue_mat_new,temp_mat,compute_inclusion_min)
            
            for tg_index in np.arange(len(temp_mat)):   ## tg_index: target index; match_ts_index: tissue index
                    match_ts_index=np.where(overlap_hdist.T[tg_index].T > 0.42)[0]
                   # print(match_ts_index,tg_index)
                    ## one targrt -to -one tissue
                    if len(match_ts_index)==1:  # match one tissue
                        #print(match_ts_index,tg_index)

                        u=locs[np.where(temp_mat[tg_index]==1)]  ## target
                        v=locs[np.where(tissue_mat_new[match_ts_index[0]]==1)]  ## tissue
                    # print(distance.directed_hausdorff(u,v)[0])
                        dist=compute_hausdorff(u,v)
                       # print(dist)
                        dist_temp.append(dist)
                        
                    ## one target -to multiple tissue
            #      ## one target match multiple tissue, should merge tissue
                    elif len(match_ts_index)>=2:  
                        #print(match_ts_index,tg_index)
                        tissue_vec = np.zeros(locs.shape[0])  ## tissue merge
                        for m_index in match_ts_index:
                            tissue_vec =tissue_vec+ tissue_mat_new[m_index]

                        u=locs[np.where(temp_mat[tg_index]==1)] 
                        v=locs[np.where(tissue_vec==1)]
                        dist=compute_hausdorff(u,v)
                       # print(dist)
                        dist_temp.append(dist)
                        
                    else:    ## target match zero tissue, (don't match tissue).
                    #    print('{} pattern no match:'.format(tg_index))      
                        for ts_index in np.arange(len(tissue_mat_new)):
                            u=locs[np.where(temp_mat[tg_index]==1)] 
                            v=locs[np.where(tissue_mat_new[ts_index]==1)]
                            dist=compute_hausdorff(u,v)
                           # print(dist)
                            dist_temp.append(dist)
            dist_result.append(min(dist_temp))
#     dist_df = pd.DataFrame([target_df.index, dist_result]).T        
#     dist_df.columns = ['geneID', 'dist']
#     dist_df.index =  target_df.index
    return dist_result,genelist, recal_genes


def compute_inclusion_min(u, v):
#    if len(np.where((u+v) == 2)[0]) == 0:
#        return len(np.where(abs(u-v) == 1)[0])
#    else:
    return len(np.where((u+v) == 2)[0])/min(sum(v),sum(u))

def compute_hausdorff(u,v):
    '''
    Compute norm hausdorff geometry distance between two pattern.
    u: target_mat
    v: tissue_mat
    
    '''
    
    dist_u_v=distance.directed_hausdorff(u,v)[0]
    dist_v_u=distance.directed_hausdorff(v,u)[0]
    
    # if norm_factor:
    #     dist= max(dist_u_v/compute_norm_factor(v),dist_v_u/compute_norm_factor(u))
    # else:
     
        
    return max(dist_u_v,dist_v_u)
    

def compute_norm_factor(coord):
    norm_dist=[]
    for i in range(coord.shape[0]):
        for j in range(coord.shape[0]):
            temp_dist = distance.euclidean(coord[i], coord[j])
            norm_dist.append(temp_dist)
    return max(norm_dist)


def compute_norm_hausdorff(u,v,norm_factor=True):
    '''
    Compute norm hausdorff geometry distance between two pattern.
    u: target_mat
    v: tissue_mat
    norm_factor: the number of pattern in positive tissue
    '''
    
    dist_u_v=distance.directed_hausdorff(u,v)[0]
    dist_v_u=distance.directed_hausdorff(v,u)[0]
    
    if norm_factor:
        dist= max(dist_u_v/compute_norm_factor(v),dist_v_u/compute_norm_factor(u))
    else:
        dist= max(dist_u_v,dist_v_u)
        
    return dist


def create_tissue_mat_new(locs,cellGraph,tissue_mat):
    '''
    devide tissue mat into single connect patterns
    '''
    
    effect_ind=[]
    
    for ts in range(len(tissue_mat)):
        temp_tissue=tissue_mat[ts]
        com=count_component(locs,cellGraph,temp_tissue)
        connectInd=[i for i in range(len(com)) if len(com[i])>=5 and len(com[i])<tissue_mat.shape[1]/4]
        #print(ts,connectInd)
        if len(connectInd)==2 or len(connectInd)==3:
            print(ts)
            effect_ind.append(ts)
            
    temp_tissue_other=np.zeros(tissue_mat.shape[1])
    for ee in effect_ind:
        temp_tissue_other=temp_tissue_other+tissue_mat[ee]
    #
    plt.scatter(locs[:,0],locs[:,1],c=temp_tissue_other)  
    #plt.show()
    com=count_component(locs,cellGraph,temp_tissue_other)
    connectInd=[i for i in range(len(com)) if len(com[i])>=3 and len(com[i])<tissue_mat.shape[1]/2]
    #connectInd
    tissue_mat_new=list()
    for i in connectInd:
        temp_graph=np.zeros(tissue_mat.shape[1])
        temp_graph[list(com[i])]=1
        plt.scatter(locs[:,0],locs[:,1],c=temp_graph)

        print('-----com_rank{}---com_size{}'.format(i,len(com[i])))
        tissue_mat_new.append(temp_graph)
        plt.show()
        
    return tissue_mat_new



## recalculate no cuts genes
def recalc_dist_to_tissue_parallel(locs,data_norm, cellGraph, gmmDict, test_genes,tissue_mat_new, add_sf=30,
                           unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    num_cores = mp.cpu_count()
    if num_cores > math.floor(len(test_genes)/2):
         num_cores=int(math.floor(len(test_genes)/2))
    ttt = np.array_split(test_genes,num_cores,axis=0)
    tuples = [(l, d,c,g,t,ts, a, u, ll, al) for l, d,c,g,t,ts, a, u, ll, al in zip(repeat(locs, num_cores),
                                                        repeat(data_norm, num_cores),
                                                        repeat(cellGraph, num_cores),
                                                        repeat(gmmDict, num_cores),                             
                                                        ttt, 
                                                        repeat(tissue_mat_new, num_cores),
                                                        repeat(add_sf, num_cores),
                                                        repeat(unary_scale_factor, num_cores),
                                                        repeat(label_cost, num_cores),
                                                        repeat(algorithm, num_cores))]
    dist_results = parmap.starmap(recalc_dist_to_tissue, tuples,
                             pm_processes=num_cores, pm_pbar=True)
    
    result_df_new=pd.DataFrame()
    best_dist_df=pd.DataFrame()
    for i in range(len(dist_results)):
        result_df_new=pd.concat([result_df_new,dist_results[i][0]])
        best_dist_df=pd.concat([best_dist_df,dist_results[i][1]])
    
    return result_df_new,best_dist_df

def cumcdf(data,num_bin=300):
    counts,bin_edges=np.histogram(data,bins=num_bin)
    cdf=np.cumsum(counts)
    x=bin_edges[1:]
    y=cdf/cdf[-1]
    return x,y