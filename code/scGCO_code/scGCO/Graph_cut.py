
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


def create_graph_with_weight(points, normCount):
    '''
    Returns a graph created from cell coordiantes.
    edge weights set by normalized counts.
    
    :param points: shape (n,2); normCount: shape (n)
    :rtype: ndarray shape (n ,3)
    
    '''
    edges = {}   
    var = normCount.var()
    delauny = Delaunay(points)
#    cellGraph = np.zeros((delauny.simplices.shape[0]*delauny.simplices.shape[1], 4))
    cellGraph = np.zeros((points.shape[0]*10, 4))

    for simplex in delauny.simplices:
        simplex.sort()
        edge0 = str(simplex[0]) + " " + str(simplex[1])
        edge1 = str(simplex[0]) + " " + str(simplex[2])
        edge2 = str(simplex[1]) + " " + str(simplex[2])
        edges[edge0] = 1
        edges[edge1] = 1
        edges[edge2] = 1
    ## remove repetitives edges among triangle
        
    i = 0
    for kk in edges.keys():  
        node0 = int(kk.split(sep=" ")[0])
        node1 = int(kk.split(sep=" ")[1])
        edgeDiff = normCount[node0] - normCount[node1]
        energy = np.exp((0 - edgeDiff**2)/(2*var))
        dist = distance.euclidean(points[node0,:], points[node1,:])
        cellGraph[i] = [node0, node1, energy, dist]       
        i = i + 1
    
    tempGraph = cellGraph[0:i]
    n_components_range = range(1,5)
    best_component = 1
    lowest_bic=np.infty
    temp_data = tempGraph[:,3].reshape(-1,1)  ## GMM of dist 
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components = n_components)
        gmm.fit(temp_data)
        gmm_bic = gmm.bic(temp_data)
        if gmm_bic < lowest_bic:
            best_gmm = gmm
            lowest_bic = gmm_bic
            best_component = n_components  
    
    mIndex = np.where(best_gmm.weights_ == max(best_gmm.weights_))[0]
    cutoff = best_gmm.means_[mIndex] + 2*np.sqrt(best_gmm.covariances_[mIndex])

    for simplex in delauny.simplices:
        simplex.sort()          
        dist0 = distance.euclidean(points[simplex[0],:], points[simplex[1],:])
        dist1 = distance.euclidean(points[simplex[0],:], points[simplex[2],:])
        dist2 = distance.euclidean(points[simplex[1],:], points[simplex[2],:])
        tempArray = np.array((dist0, dist1, dist2))
        badIndex = np.where(tempArray == max(tempArray))[0][0]  ## remove longest edges among simplex taiangle.
        if tempArray[badIndex] > cutoff:
            edge0 = str(simplex[0]) + " " + str(simplex[1])  
            edge1 = str(simplex[0]) + " " + str(simplex[2])       
            edge2 = str(simplex[1]) + " " + str(simplex[2])
            edgeCount = 0
            if edge0 in edges and edge1 in edges and edge2 in edges:
                if badIndex == 0:
                    del edges[edge0]
                elif badIndex == 1:
                    del edges[edge1]
                elif badIndex == 2:
                    del edges[edge2]     ## remove longest edges from edges

    i = 0
    for kk in edges.keys():         ## recrete cellGraph with new edges
        node0 = int(kk.split(sep=" ")[0])
        node1 = int(kk.split(sep=" ")[1])
        edgeDiff = normCount[node0] - normCount[node1]
        energy = np.exp((0 - edgeDiff**2)/(2*var))
        dist = distance.euclidean(points[node0,:], points[node1,:])
        cellGraph[i] = [node0, node1, energy, dist]       
        i = i + 1   
      
    tempGraph = cellGraph[0:i]
    temp_data = tempGraph[:,3].reshape(-1,1)    
    gmm = mixture.GaussianMixture(n_components = 1)
    gmm.fit(temp_data)    
    cutoff = gmm.means_[0] + 2*np.sqrt(gmm.covariances_[0])
    finalGraph = tempGraph.copy()
    j=0
    for i in np.arange(tempGraph.shape[0]):    
        if tempGraph[i, 3] < cutoff:     ### re-test all edges' dist have similar distribution. 
            finalGraph[j] = tempGraph[i]
            j = j + 1
         
    return finalGraph

def find_mixture(data):
    '''
    estimate expression clusters
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''

    #n_components_range = range(2,5)
    best_component = 2
    lowest_bic=np.infty
    temp_data = data.reshape(-1,1)

    if len(temp_data)<=2:
        gmm = mixture.GaussianMixture(n_components = 2)
        gmm.fit(temp_data)
        best_gmm=gmm
    else:
        if len(temp_data)<5 and len(temp_data)>2:
            n_components_range=range(2,len(temp_data))

        else:
             n_components_range = range(2,5)

        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components = n_components)
            gmm.fit(temp_data)
            gmm_bic = gmm.bic(temp_data)
            if gmm_bic < lowest_bic:
                best_gmm = gmm
                lowest_bic = gmm_bic
                best_component = n_components      

    return best_gmm
def find_mixture_2(data):
    '''
    estimate expression clusters, use k=2
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    gmm = mixture.GaussianMixture(n_components = 2)
    gmm.fit(data.reshape(-1,1))

    return gmm

def TSNE_gmm(data):
    n_components_range = range(2,10)
    best_component = 2
    lowest_bic=np.infty
    temp_data = data
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components = n_components)
        gmm.fit(temp_data)
        gmm_bic = gmm.bic(temp_data)
        if gmm_bic < lowest_bic:
            best_gmm = gmm
            lowest_bic = gmm_bic
            best_component = n_components     
    return best_gmm

def perform_gmm(count):
    '''
    reture one trained gmm model
    
    '''
    a = count.copy()
#    if sum(a>1)/len(count) <= 0.1 and sum(a>1)<=30:
#        np.place(a, a==0, (np.random.rand(sum(a==0))*0.25)) 
#        gmm = find_mixture_2(a)       
#    else:
    if True:
        a=a[a>0]
        gmm = find_mixture(a)
        gmm_pred = gmm.predict(count.reshape(-1,1))
        unique, counts = np.unique(gmm_pred,return_counts=True)
        if np.min(counts) < 0.1*len(count):
            gmm = find_mixture_2(a) 
    return gmm

def gmm_model(data_norm):
    gmmDict_={}
    for geneID in data_norm.columns:
        count=data_norm.loc[:,geneID].values
        gmm=perform_gmm(count)   
        gmmDict_[geneID]=gmm
    return gmmDict_

def multiGMM(data_norm):
    num_cores = mp.cpu_count()
    if num_cores > math.floor(data_norm.shape[1]/2):
        num_cores=int(math.floor(data_norm.shape[1]/2))
   # print(num_cores)
    ttt = np.array_split(data_norm,num_cores,axis=1)
    #print(ttt)

    tuples = [d for d in zip(ttt)] 
    gmmDict_=parmap.starmap(gmm_model, tuples,
                             pm_processes=num_cores, pm_pbar=True)
    gmmDict={}
    for i in np.arange(len(gmmDict_)):
        gmmDict.update(gmmDict_[i])   ## dict.update()  add dict to dict
    return gmmDict


def first_neg_index(a):
    '''
    deprecated
    '''
    for i in np.arange(a.shape[0]):
        if a[i] < 0:
            return i
    return a.shape[0] - 1                

def calc_u_cost(a, mid_points):
    '''
    deprecated
    '''
    neg_index = int(a[0])
    x = a[1]
    m_arr = np.concatenate((0 - mid_points[0:neg_index+1], 
                            mid_points[neg_index:]), axis=0)
    x_arr = np.concatenate((np.repeat(x, neg_index+1), 
                0 - np.repeat(x, mid_points.shape[0] - neg_index)), axis=0)
    return m_arr+x_arr   


def compute_pairwise_cost(size, smooth_factor):
    '''
    Returns pairwise energy.
    
    :param points: size: scalar; smooth_factor: scalar

    :rtype: pairwise energy matrix.
    '''
    pairwise_size = size
    pairwise = -smooth_factor * np.eye(pairwise_size, dtype=np.int32)
    step_weight = -smooth_factor*np.arange(pairwise_size)[::-1]
    for i in range(pairwise_size): 
        pairwise[i,:] += np.roll(step_weight,i) 
    temp = np.triu(pairwise).T + np.triu(pairwise)
    np.fill_diagonal(temp, np.diag(temp)/2)
    return temp

def cut_graph_general_profile(cellGraph, count,gmm, unary_scale_factor=100, 
                      smooth_factor=50, label_cost=10, algorithm='expansion'):
    '''
    Returns new labels and gmm for the cut with gmm profile.
    
    :param points: cellGraph (n,3); count: shape (n,); 
    :unary_scale_factor, scalar; smooth_factor, scalar; 
    :label_cost: scalar; algorithm='expansion'
    :rtype: label shape (n,); gmm object.
    '''
    unary_scale_factor = unary_scale_factor
    label_cost = label_cost
    algorithm = algorithm
    smooth_factor = smooth_factor
    gmm=gmm      
    unary_cost = compute_unary_cost_simple_profile(count, gmm, unary_scale_factor)
    
    pairwise_cost = compute_pairwise_cost(gmm.means_.shape[0], smooth_factor)
    edges = cellGraph[:,0:2].astype(np.int32)
    labels = pygco.cut_from_graph(edges, unary_cost, pairwise_cost, label_cost)
#    energy = compute_energy(unary_cost, pairwise_cost, edges, labels)

    return labels

def compute_unary_cost_simple_profile(count, gmm, unary_scale_factor):
    '''
    Returns unary cost energy.
    
    :param points: count: shape (n,); gmm: gmm object; scale_factor: scalar

    :rtype: unary energy matrix.
    '''    
    exp = count
    a= exp[exp > 0]
    gmm_pred = gmm.predict(a.reshape(-1,1))
    zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
    labels_pred = gmm.predict(exp.reshape(-1,1))
    if len(np.where(exp == 0)[0]) > 0:
        np.place(labels_pred, exp==0, zero_label)
    uniq, count = np.unique(labels_pred, return_counts = True)    

    if len(uniq)<2:  ## when just one uniq,the will mismatch dimension with pairwaise label
        uniq_modify =np.append(uniq,uniq)
    else:
        uniq_modify =uniq
    uninary_mat = np.zeros((len(labels_pred), len(uniq_modify)))
    for i in np.arange(uninary_mat.shape[0]):
        for j in np.arange(len(uniq_modify)):
            if uniq_modify[j] == labels_pred[i]:  ## same ,energy -1; imsame ,energy 1.
                uninary_mat[i, j] = -1
            else:
                uninary_mat[i, j] = 1   
    return (unary_scale_factor*uninary_mat).astype(np.int32)


def cut_graph_general(cellGraph, count,gmm, unary_scale_factor=100, 
                      smooth_factor=30, label_cost=10, algorithm='expansion',
                      profile=False):
    '''
    Returns new labels and gmm for the cut.
    
    :param points: cellGraph (n,3); count: shape (n,); 
    :unary_scale_factor, scalar; smooth_factor, scalar; 
    :label_cost: scalar; algorithm='expansion'
    :rtype: label shape (n,); gmm object.
    '''
    unary_scale_factor = unary_scale_factor
    label_cost = label_cost
    algorithm = algorithm
    smooth_factor = smooth_factor
    gmm=gmm
#     a = count.copy() 
#     if sum(a>1)/len(count) <= 0.1 and sum(a>1)<=30:             ###  1-1, 0.25 can modify 
#         np.place(a, a==0, (np.random.rand(sum(a==0))*0.25)) # using 0.1 gives layer 4 300; 0.25 gives 150; 0.5 gives 80?
#         gmm = find_mixture_2(a)       
#     else:
#         a=a[a>0]
#         gmm = find_mixture(a)
    
    if profile==False:
        unary_cost = compute_unary_cost_simple(count,  gmm,  unary_scale_factor)
    else:
        unary_cost = compute_unary_cost_simple_profile(count, gmm, unary_scale_factor)
    
    pairwise_cost = compute_pairwise_cost(gmm.means_.shape[0], smooth_factor)
    edges = cellGraph[:,0:2].astype(np.int32)
    labels = pygco.cut_from_graph(edges, unary_cost,
                                                 pairwise_cost.astype(np.int32), np.int32(label_cost))
#    energy = compute_energy(unary_cost, pairwise_cost, edges, labels)

    return labels



def compute_unary_cost_simple(count, gmm, scale_factor):
    '''
    Returns unary cost energy.
    
    :param points: count: shape (n,); gmm: gmm object; scale_factor: scalar

    :rtype: unary energy matrix.
    '''    
    exp = count
    a= exp[exp > 0]

    gmm_pred = gmm.predict(a.reshape(-1,1))
    zero_label = gmm.predict(np.min(a).reshape(-1,1))[0]
    labels_pred = gmm.predict(exp.reshape(-1,1))
    if len(np.where(exp == 0)[0]) > 0:
        np.place(labels_pred, exp==0, zero_label)
    
    temp_means = np.sort(gmm.means_, axis=None)
   # new_index = np.where(gmm.means_ == temp_means)[1]
    # temp_covs = gmm.covariances_.copy()
    # for i in np.arange(new_index.shape[0]):
    #     temp_covs[i] = gmm.covariances_[new_index[i]]  ##sorted gmm.covariances as np.sort(gmm.means_)
    new_index= np.argsort(gmm.means_,axis=0).reshape(1,-1)[0]
    temp_covs = gmm.covariances_[new_index]

    new_labels = np.zeros(labels_pred.shape[0], dtype=np.int32)
    for i in np.arange(new_index.shape[0]):
        temp_index = np.where(labels_pred == i)[0]
        new_labels[temp_index] = new_index[i]  ## the label of min(gmm.means_) is 0; label of max is 1.

    mid_points = np.zeros(len(new_index) - 1)
    for i in np.arange(len(mid_points)):
        mid_points[i] = (temp_means[i]*np.sqrt(temp_covs[i+1]) + 
                     temp_means[i+1]*np.sqrt(temp_covs[i])
                    )/(np.sqrt(temp_covs[i]) + np.sqrt(temp_covs[i+1]))
    temp = count[:, np.newaxis] - temp_means.T[1:]
    neg_indices = np.apply_along_axis(first_neg_index, 1, temp)
    ind_count_arr = np.vstack((neg_indices, count)).T        
    unary_cost =  (scale_factor*np.apply_along_axis(calc_u_cost, 1, 
                                    ind_count_arr, mid_points)).astype(np.int32)
    a = count.copy()
    if sum(a>1)/len(count) <= 0.1 and sum(a>1)<=30:
        unique_cost, counts_cost = np.unique(unary_cost, return_counts=True)
        val_to_replace1 = unique_cost[np.argmax(counts_cost)]
        val_to_replace2 = 0 - unique_cost[np.argmax(counts_cost)]
        if val_to_replace1 < 10 and val_to_replace1 > 0:
            np.place(unary_cost, unary_cost == val_to_replace1, np.median(abs(unique_cost)))
        elif val_to_replace1 < 0 and val_to_replace1 > -10:
            np.place(unary_cost, unary_cost == val_to_replace1, 0-np.median(abs(unique_cost)))
        if val_to_replace2 < 10 and val_to_replace2 > 0:
            np.place(unary_cost, unary_cost == val_to_replace2, np.median(abs(unique_cost)))
        elif val_to_replace2 < 0 and val_to_replace2 > -10:
            np.place(unary_cost, unary_cost == val_to_replace2, 0-np.median(abs(unique_cost)))
    return unary_cost


def noise_inside(a, b, cellGraph):
    idx0 = np.in1d(cellGraph[:,0], np.array(list(a))).nonzero()[0]
    idx1 = np.in1d(cellGraph[:,1], np.array(list(a))).nonzero()[0]
    neighbor0 = cellGraph[idx0, 1]
    neighbor1 = cellGraph[idx1, 0]
    neighbors = set(neighbor0.tolist() + neighbor1.tolist())   
    out_neighbors = neighbors.difference(set(a))
    not_a_neighbors = out_neighbors.difference(set(b))
    return (len(not_a_neighbors) == 0)


# added noise
def compute_p_CSR(locs, newLabels, gmm, exp, cellGraph): 
    '''
    Returns p_value of the cut.
    
    :param points: newLabels: shape (n,); gmm: gmm object
                   exp: ndarray shape (n ,3); cellGraph: shape (n,3)

    :rtype: p_value.
    '''
    com_factor = 1
    p_values = list()
    node_lists = list()
    gmm_pred=gmm_predict(exp,gmm)
    unique, counts = np.unique(gmm_pred,return_counts=True)
    con_components = count_component(locs,cellGraph, newLabels)  ## nodes index in subgraphs
    noise = dict()
    # now calculate p for all comp without considering noise
    if True:
        min_sig_p_size = np.inf
        for j in np.arange(len(con_components)):
            if len(con_components[j]) >=3: # we want to score the object not the back ground
                node_list = con_components[j]
                com_size = len(node_list)
                gmm_pred_com=gmm_pred[list(node_list)]          
    
        # check 0s
                unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)
                major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
                label_count = counts[np.where(unique == major_label)[0]]  ## Ci
# check 0s
                unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)  
                major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
                label_count = counts[np.where(unique == major_label)[0]]  ##  real counts that in subgraphs,Ci
                count_in_com =  counts_com.max()   ## counts by graph cuts,k
                cover = exp.shape[0]/com_size
                p0 = poisson.sf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
                p1 = poisson.pmf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
                prob=min((p0+p1)*cover,1)  
                p_values.append(prob)
                if prob <0.1 and len(con_components[j]) < min_sig_p_size:
                    min_sig_p_size = len(con_components[j])
            else: # set small comp p=1
                p_values.append(1)            
            node_lists.append(np.array(list(node_list)))
#        print(p_values)
        for j in np.arange(len(con_components)):
            if p_values[j] >= 0.1 and len(con_components[j]) < 10: # min_sig_p_size:
                noise[j] = con_components[j]
    # now re-calculate p considering noise
        for j in np.arange(len(con_components)):
            if p_values[j] < 0.1: # small p let correct by consider noise
                noise_size = 0
                used_com = list()
                for jj, cc in noise.items():
                    if noise_inside(cc, con_components[j], cellGraph):
                        noise_size = noise_size + len(cc)
                        used_com.append(jj)
                if noise_size > 0:
                    for jj in used_com:
                        noise.pop(jj)
                    node_list = con_components[j]
                    com_size = len(node_list)
                    gmm_pred_com=gmm_pred[list(node_list)]
                    unique_com, counts_com = np.unique(gmm_pred_com, return_counts=True)  
                    major_label = unique_com[np.where(counts_com == counts_com.max())[0][0]]
                    label_count = counts[np.where(unique == major_label)[0]]  ##  real counts that in subgraphs,Ci
                    count_in_com =  counts_com.max()   ## counts by graph cuts,k
                    com_size = com_size + noise_size
                    cover = exp.shape[0]/com_size
                
                    p0 = poisson.sf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
                    p1 = poisson.pmf(count_in_com, com_size*(label_count/exp.shape[0]))[0]
                    prob=min((p0+p1)*cover,1)  
                    p_values[j] = prob      
            
    return p_values, node_lists, con_components


def count_component(locs, cellGraph, newLabels):
    '''
    Returns number of subgraphs.
    
    :param points: cellGraph: shape (n,3); newLabels: ndarray shape (n,); locs: shape (n, 2) 

    :rtype: scalar. 
    
    '''

    G_cut = nx.Graph()
    tempGraph = cellGraph.copy()

    tempGraph = np.apply_along_axis(remove_egdes, 1, tempGraph, newLabels)   ## reassign the edges energy between two nodes,save cellGraph[:,2]
    G_cut.add_nodes_from(list(set(list(tempGraph[:,0].astype(np.int32)) + list(tempGraph[:,1].astype(np.int32)))))    
    G_cut.add_edges_from(tempGraph[np.where(tempGraph[:,2] == 1)[0],0:2].astype(np.int32))  ## connect same label to one subgraph.
    
    com = sorted(nx.connected_components(G_cut),    ## sort by len(# nodes in subgraphs), com[1] is second largest subgraphs
                                  key = len, reverse=True)  

    return com  

def count_isolate(locs, cellGraph, newLabels):
    '''
    Returns number of subgraphs.
    
    :param points: cellGraph: shape (n,3); newLabels: ndarray shape (n,); locs: shape (n, 2) 

    :rtype: scalar. 
    
    '''

    G_full = nx.Graph()
    G_cut = nx.Graph()
    tempGraph = cellGraph.copy()
    G_full.add_nodes_from(list(set(list(tempGraph[:,0].astype(np.int32)) + list(tempGraph[:,1].astype(np.int32)))))
    G_full.add_edges_from(tempGraph[:, 0:2].astype(np.int32))   
    tempGraph = np.apply_along_axis(remove_egdes, 1, tempGraph, newLabels)
    G_cut.add_nodes_from(list(set(list(tempGraph[:,0].astype(np.int32)) + list(tempGraph[:,1].astype(np.int32)))))    
    G_cut.add_edges_from(tempGraph[np.where(tempGraph[:,2] == 1)[0],0:2].astype(np.int32))
    degree_cutoff = int(min(nx.average_degree_connectivity(G_full).values()))
    com = sorted(nx.connected_components(G_cut), 
                                  key = len, reverse=True)  
    isolated_nodes = list(nx.isolates(G_cut))
    seg_size = np.zeros(18)
    if len(isolated_nodes) > 0:
        deg_dict = {key: value for (key, value) in list(G_full.degree(list(nx.isolates(G_cut))))}
#    seg_size[0] = (np.array(list(deg_dict.values())) > degree_cutoff).sum()             
#    t_com = len(com) + locs.shape[0] - sum_nodes 
#    t_com = locs.shape[0] - sum_nodes 
        seg_size[0] = (np.array(list(deg_dict.values())) <= degree_cutoff).sum() 
        seg_size[1] = (np.array(list(deg_dict.values())) > degree_cutoff).sum() 
    for cc in com:
        if len(cc) == 2:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() > 0:
                seg_size[2] = seg_size[2] + 1
            else:
                seg_size[3] = seg_size[3] + 1
                
        if len(cc) == 3:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[4] = seg_size[4] + 1
            else:
                seg_size[5] = seg_size[5] + 1  
    
        if len(cc) == 4:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[6] = seg_size[6] + 1
            else:
                seg_size[7] = seg_size[7] + 1 
        if len(cc) == 5:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[8] = seg_size[8] + 1
            else:
                seg_size[9] = seg_size[9] + 1 
        if len(cc) == 6:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[10] = seg_size[10] + 1
            else:
                seg_size[11] = seg_size[11] + 1 
        if len(cc) == 7:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[12] = seg_size[12] + 1
            else:
                seg_size[13] = seg_size[13] + 1 
        if len(cc) == 8:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[14] = seg_size[14] + 1
            else:
                seg_size[15] = seg_size[15] + 1 
        if len(cc) == 9:
            deg_dict = {key: value for (key, value) in list(G_full.degree(list(cc)))}
            if (np.array(list(deg_dict.values())) <= degree_cutoff).sum() >= 2:
                seg_size[16] = seg_size[16] + 1
            else:
                seg_size[17] = seg_size[17] + 1 
        
    return seg_size #(np.array(list(deg_dict.values())) > degree_cutoff).sum()


def remove_egdes(edges, newLabels):
    '''
    Mark boundary of the cut.
    
    :param points: edges: shape (n,); newLabels: shape(k,)

    :rtype: marked edges.
    '''
    if newLabels[int(edges[0])] != newLabels[int(edges[1])]:   
        edges[2] = 0
    else:
        edges[2] = 1
    return edges



def gmm_predict(exp,gmm):
    """Predict the labels for the data samples in X using trained model and replace the zreo using the minimum data label.

    Parameters
    ----------
    exp : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    gmm : Gussian Mixture model

    Returns
    -------
    labels : array, shape (n_samples,)
        Component labels.
    """
    a=exp[exp>0]
    gmm_pred=gmm.predict(a.reshape(-1,1))
    zero_label=gmm.predict(np.min(a).reshape(-1,1))[0]
    label_pred = gmm.predict(exp.reshape(-1,1))
    if len(np.where(exp == 0)[0]) > 0:
        np.place(label_pred, exp==0, zero_label)
    gmm_pred =label_pred
    return gmm_pred



## 20200629
def compute_spatial_genomewise_optimize_gmm(locs, data_norm, cellGraph, gmmDict, smooth_factor=10, 
                                         unary_scale_factor=100, label_cost=10, algorithm='expansion'):
    
    
    size_factor=200
    noise_size_estimate= 9
    genes = list()
    nodes = list()
    p_values = list()
    smooth_factors = list()
    pred_labels = list()
    start_factor=smooth_factor
    
    for geneID in data_norm.columns:
#         print(geneID)
        exp =  data_norm.loc[:,geneID].values
        gmm=gmmDict[geneID]
        ## 0. init cuts graph
        temp_factor = start_factor
        newLabels = cut_graph_general(cellGraph, exp, gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)
        p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)
        num_isolate = count_isolate(locs,cellGraph, newLabels) 

        noise_size_inside = sum([num_isolate[num] for num in np.arange(1,2*noise_size_estimate, 2)]) # 0629, 16:16
        noise_size_border = sum([num_isolate[num] for num in np.arange(2,2*noise_size_estimate, 2)])
        if (noise_size_inside>1 and noise_size_border< noise_size_estimate) and (noise_size_border> 2*noise_size_inside and noise_size_border< 3*noise_size_inside): #0629 16:42
            noise_size= noise_size_inside
        elif (noise_size_border>1 and noise_size_inside< noise_size_estimate) and  (noise_size_inside> 2*noise_size_border and noise_size_inside< 3*noise_size_border) :
            noise_size = noise_size_border
        else:
            noise_size= noise_size_inside + noise_size_border

        noise_size_norm = noise_size*(size_factor/len(exp))

               
        # 1. cuts too many noise
        while  noise_size_norm > 2 and len(p)>0 and min(p) < 0.01 : 
            temp_factor = temp_factor + 10   ## can speed up with +10
            newLabels = cut_graph_general(cellGraph, exp, gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)
            p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)  
            num_isolate = count_isolate(locs,cellGraph, newLabels)  

            noise_size_inside = sum([num_isolate[num] for num in np.arange(1,2*noise_size_estimate, 2)])
            noise_size_border = sum([num_isolate[num] for num in np.arange(2,2*noise_size_estimate, 2)])

            if (noise_size_inside>1 and noise_size_border< noise_size_estimate) and (noise_size_border> 2*noise_size_inside and noise_size_border< 3*noise_size_inside):
                noise_size= noise_size_inside
            elif (noise_size_border>1 and noise_size_inside< noise_size_estimate) and (noise_size_inside> 2*noise_size_border and noise_size_inside< 3*noise_size_border) :
                noise_size = noise_size_border
            else:
                noise_size= noise_size_inside + noise_size_border

            noise_size_norm = noise_size*(size_factor/len(exp))

        if len(p)>0 and min(p) >= 0.01:
            logP = (0-np.log10(min(p)))
        else:
            logP = 0 - sum(np.log10(np.array(p)[np.array(p)<0.01]))                         
        obj_val = logP - noise_size_norm


        p_best = p
        node_best = node
        newLabels_best = newLabels
        temp_factor_best = temp_factor  
        obj_val_best = obj_val
        noise_size_best = noise_size_norm
        com_best=com
        
        # 2. recuts effect patterns, get best obj_val pattern
        obj_bad = 0
        while len(p)>0 and min(p) < 0.01:   
            temp_factor = temp_factor + 5   ## can speed up with +10
            newLabels = cut_graph_general(cellGraph, exp, gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)
            p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)

            num_isolate = count_isolate(locs,cellGraph, newLabels)  

            noise_size_inside = sum([num_isolate[num] for num in np.arange(1,2*noise_size_estimate, 2)])
            noise_size_border = sum([num_isolate[num] for num in np.arange(2,2*noise_size_estimate, 2)])
            if (noise_size_inside>1 and noise_size_border< noise_size_estimate) and (noise_size_border> 2*noise_size_inside and noise_size_border< 3*noise_size_inside):
                noise_size= noise_size_inside
            elif (noise_size_border>1 and noise_size_inside< noise_size_estimate) and  (noise_size_inside> 2*noise_size_border and noise_size_inside< 3*noise_size_border) :
                noise_size = noise_size_border
            else:
                noise_size= noise_size_inside + noise_size_border
   
            noise_size_norm = noise_size*(size_factor/len(exp))


            if len(p)>0 and min(p) >= 0.01:
                logP = (0-np.log10(min(p)))
            else:
                logP = 0 - sum(np.log10(np.array(p)[np.array(p)<0.01]))    
            obj_val = logP - noise_size_norm


            if obj_val > obj_val_best:
                p_best = p
                node_best = node
                newLabels_best = newLabels
                temp_factor_best = temp_factor      
                obj_val_best = obj_val
                noise_size_best = noise_size_norm 
                com_best=com
            else:
                obj_bad+=1
            if noise_size <= 1 or obj_bad>2:   ## 0629 16:16
                break

        
        # 3. For bad obj_val , need recuts
        if obj_val_best < 5: # modify
            p = p_best
            node = node_best
            newLabels = newLabels_best
            temp_factor = temp_factor_best      
            obj_val = obj_val_best
            noise_size_norm = noise_size_best 
            com = com_best
            while  noise_size_norm > 0.7 and len(p)>0 and min(p) < 0.01 : 
                temp_factor = temp_factor + 10   ## can speed up with +10
                newLabels = cut_graph_general(cellGraph, exp, gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)
                p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)  
                num_isolate = count_isolate(locs,cellGraph, newLabels)         
                
                noise_size_inside = sum([num_isolate[num] for num in np.arange(1,2*noise_size_estimate, 2)])
                noise_size_border = sum([num_isolate[num] for num in np.arange(2,2*noise_size_estimate, 2)])
                if (noise_size_inside>1 and noise_size_border< noise_size_estimate) and (noise_size_border> 2*noise_size_inside and noise_size_border< 3*noise_size_inside):
                    noise_size= noise_size_inside
                elif (noise_size_border>1 and noise_size_inside< noise_size_estimate) and  (noise_size_inside> 2*noise_size_border and noise_size_inside< 3*noise_size_border) :
                    noise_size = noise_size_border
                else:
                    noise_size= noise_size_inside + noise_size_border
                noise_size_norm = noise_size*(size_factor/len(exp))
                
                if noise_size_norm < 0.01:
                    break
            p_best = p
            node_best = node
            newLabels_best = newLabels
            temp_factor_best = temp_factor      
            noise_size_best = noise_size_norm
            com_best=com
        
        ## 4. For small significate p_value pattern, need bigger sf
        if len(p_best)>0 and len(node_best[np.argmin(p_best)]) < noise_size_estimate:
            p = p_best
            node = node_best
            newLabels = newLabels_best
            temp_factor = temp_factor_best      
            obj_val = obj_val_best
            noise_size_norm = noise_size_best
            com_best=com

            while min(p) < 0.01: 
                temp_factor = temp_factor + 50   
                newLabels = cut_graph_general(cellGraph, exp, gmm, unary_scale_factor, 
                                           temp_factor, label_cost, algorithm)
                p, node, com = compute_p_CSR(locs, newLabels, gmm, exp, cellGraph)   
  
                if temp_factor > 120:
                    break

            p_best = p
            node_best = node
            newLabels_best = newLabels
            temp_factor_best = temp_factor      
            com_best=com
            
        if len(p_best)>0:
            final_factor = temp_factor_best
            p_values.append(p_best)
            nodes.append(node_best)
            genes.append(geneID)
            smooth_factors.append(final_factor)
            pred_labels.append(newLabels_best)
            
    return nodes, p_values, genes, smooth_factors, pred_labels



def identify_spatial_genes(locs, data_norm, cellGraph, gmmDict, smooth_factor=10,
                      unary_scale_factor=100, label_cost=10, algorithm='expansion'):
#    pool = mp.Pool()
    '''
    main function to identify spatially variable genes
    :param file:locs: spatial coordinates (n, 2); data_norm: normalized gene expression;
        smooth_factor=10; unary_scale_factor=100; label_cost=10; algorithm='expansion' 
    :rtype: prediction: a dataframe
    '''    
    
    num_cores = mp.cpu_count()
    if num_cores > math.floor(data_norm.shape[1]/2):
         num_cores=int(math.floor(data_norm.shape[1]/2))
    ttt = np.array_split(data_norm,num_cores,axis=1)
    tuples = [(l, d, c, g, s, u, b, a) for l, d, c, g, s, u, b, a in zip(
                                    repeat(locs, num_cores), 
                                    ttt,
                                    repeat(cellGraph, num_cores),
                                    repeat(gmmDict, num_cores),
                                    repeat(smooth_factor, num_cores),
                                    repeat(unary_scale_factor, num_cores), 
                                    repeat(label_cost, num_cores),
                                    repeat(algorithm, num_cores))] 
    
    results = parmap.starmap(compute_spatial_genomewise_optimize_gmm, tuples,
                                pm_processes=num_cores, pm_pbar=True)
    
#    pool.close()
# p_values, genes, diff_p_values, exp_diff, smooth_factors, pred_labels
    nnn = [results[i][0] for i in np.arange(len(results))]
    nodes = reduce(operator.add, nnn)
    ppp = [results[i][1] for i in np.arange(len(results))]
    p_values=reduce(operator.add, ppp)
    ggg = [results[i][2] for i in np.arange(len(results))]
    genes = reduce(operator.add, ggg)
    # exp_ppp = [results[i][3] for i in np.arange(len(results))]
    # exp_pvalues = reduce(operator.add, exp_ppp)  
    # exp_ddd = [results[i][4] for i in np.arange(len(results))]
    # exp_diffs = reduce(operator.add, exp_ddd)      
    fff = [results[i][3] for i in np.arange(len(results))]
    s_factors = reduce(operator.add, fff)
    lll = [results[i][4] for i in np.arange(len(results))]
    pred_labels = reduce(operator.add, lll)

    best_p_values=[min(i) for i in p_values]
    fdr = multi.multipletests(np.array(best_p_values), method='fdr_bh')[1]
    #exp_fdr = multi.multipletests(np.array(exp_pvalues), method='fdr_bh')[1]    
    
    labels_array = np.array(pred_labels).reshape(len(genes), pred_labels[0].shape[0])
    data_array = np.array((genes, p_values, fdr,s_factors, nodes), dtype=object).T
    t_array = np.hstack((data_array, labels_array))
    c_labels = ['p_value', 'fdr',  'smooth_factor', 'nodes']
    for i in np.arange(labels_array.shape[1]) + 1:
        temp_label = 'label_cell_' + str(i)
        c_labels.append(temp_label)
    result_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                      columns=c_labels)
    
    return result_df