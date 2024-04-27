import numpy as np
import pandas as pd
from sklearn import mixture
import statsmodels.stats.multitest as multi
import multiprocessing as mp
import math
import parmap
from itertools import repeat

def find_mixture(data, random_state = 2020):
    '''
    estimate expression clusters
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''

    #n_components_range = range(2,5)
    best_component = 2
    lowest_bic=np.infty

    data = np.unique(data)  # all duplications, when data[data>0]
    data = list(data).copy()

    if len(data)<=2:   # 20211119 coco
        # len(data) == 0, data=[0,0], n = 1
        # len(data) == 1, data=[x,0], n=2
        # len(data) == 2, data[x1,x2], n= 2
        # len(data) == 3/4,  , n = [2,3/4]
        # len(data) >=5, , n= range(2,5)
        zero = np.repeat([0],3-len(data))
        data.extend(zero)
        temp_data = np.array(data).reshape(-1,1)
        n_components = min(len(np.unique(data)),2)
        gmm = mixture.GaussianMixture(n_components = n_components,random_state=random_state)
        gmm.fit(temp_data)
        best_gmm=gmm
    else:
        temp_data = np.array(data).reshape(-1,1)
        if len(temp_data)<5 and len(temp_data)>2:
            n_components_range=range(2,len(temp_data))

        else:
             n_components_range = range(2,5)

        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components = n_components,random_state=random_state)
            gmm.fit(temp_data)
            gmm_bic = gmm.bic(temp_data)
            if gmm_bic < lowest_bic:
                best_gmm = gmm
                lowest_bic = gmm_bic
                best_component = n_components      

    return best_gmm


def find_mixture_2(data,random_state=2020):
    '''
    estimate expression clusters, use k=2
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    data = np.unique(data)
    data = list(data).copy()

    if len(data)<=2:
        zero = np.repeat([0],3-len(data))
        data.extend(zero)
    n_components = min(len(np.unique(data)), 2)
    temp_data = np.array(data).reshape(-1,1)
    gmm = mixture.GaussianMixture(n_components = n_components,random_state=random_state)
    gmm.fit(temp_data)

    return gmm


def find_mixture_n(data,n_components = 2, random_state=2020):
    '''
    estimate expression clusters, use k=2
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    data = np.unique(data)
    data = list(data).copy()

    if len(data)<=2:
        zero = np.repeat([0],3-len(data))
        data.extend(zero)
    n_components = min(len(np.unique(data)), n_components)
    temp_data = np.array(data).reshape(-1,1)
    gmm = mixture.GaussianMixture(n_components = n_components,random_state=random_state)
    gmm.fit(temp_data)

    return gmm



def perform_gmm(count, random_state = 2020):
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
        gmm = find_mixture(a, random_state=random_state)
        gmm_pred = gmm.predict(count.reshape(-1,1))
        unique, counts = np.unique(gmm_pred,return_counts=True)
        if np.min(counts) < 0.1*len(count):
            gmm = find_mixture_2(a, random_state= random_state) 
    return gmm


def gmm_model(data_norm, random_state = 2020):
    gmmDict_={}
    for geneID in data_norm.columns:
        count=data_norm.loc[:,geneID].values
        gmm=perform_gmm(count, random_state = random_state)   
        gmmDict_[geneID]=gmm
    return gmmDict_


def multiGMM(data_norm,random_state=2020, ncores = None):
    all_cores = mp.cpu_count()
    # ncores = input('You can select your number of cores: ')
    # print(ncores)
    if ncores !=None:
        num_cores = ncores
    else:
        num_cores = int(all_cores*0.6)
    if num_cores > math.floor(data_norm.shape[1]/2):
        num_cores=int(math.floor(data_norm.shape[1]/2))
    print(f'scGCO used {num_cores} out of {all_cores} cores')

    ttt = np.array_split(data_norm,num_cores,axis=1)
    tuples = [(d, rd) for d, rd in zip(ttt, 
                                    repeat(random_state, num_cores))] 
    gmmDict_=parmap.starmap(gmm_model, tuples,
                             pm_processes=num_cores, pm_pbar=True)
    gmmDict={}
    for i in np.arange(len(gmmDict_)):
        gmmDict.update(gmmDict_[i])   ## dict.update()  add dict to dict
    return gmmDict


def gmm_model_2(data_norm,random_state=2020 ):
    gmmDict_={}
    for geneID in data_norm.columns:
        count=data_norm.loc[:,geneID].values
        gmm=find_mixture_2(count,random_state=random_state)   
        gmmDict_[geneID]=gmm
    return gmmDict_


def multiGMM_2(data_norm,random_state=2020):
    '''
    GMM with conponents =2
    '''
    num_cores = mp.cpu_count()
    num_cores = int(num_cores*0.5)
    if num_cores > math.floor(data_norm.shape[1]/2):
        num_cores=int(math.floor(data_norm.shape[1]/2))
   # print(num_cores)
    ttt = np.array_split(data_norm,num_cores,axis=1)
    #print(ttt)

    tuples = [(d, rd) for d, rd in zip(ttt, 
                                    repeat(random_state, num_cores))] 
    gmmDict_=parmap.starmap(gmm_model, tuples,
                             pm_processes=num_cores, pm_pbar=True)
    gmmDict={}
    for i in np.arange(len(gmmDict_)):
        gmmDict.update(gmmDict_[i])   ## dict.update()  add dict to dict
    return gmmDict




def TSNE_gmm(data):
    n_components_range = range(2,10)
    best_component = 2
    lowest_bic=np.infty
    temp_data = data
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components = n_components,random_state=2020)
        gmm.fit(temp_data)
        gmm_bic = gmm.bic(temp_data)
        if gmm_bic < lowest_bic:
            best_gmm = gmm
            lowest_bic = gmm_bic
            best_component = n_components     
    return best_gmm
