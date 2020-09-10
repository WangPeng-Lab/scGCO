import itertools
import operator
from functools import reduce
from scipy.spatial import distance
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy import stats
from ast import literal_eval
import pickle
from .Graph_cut import create_graph_with_weight


def store_gmm(gmmDict,fileName):
    with open(fileName,"wb") as fw:
        pickle.dump(gmmDict,fw)

def grab_gmm(fileName):
    with open(fileName,"rb") as fr:
        return pickle.load(fr)
       

def read_spatial_expression(file,sep='\s+',num_exp_genes=0.01, num_exp_spots=0.05, min_expression=1):
    
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
#     print("Number of expressed genes a spot must have to be kept " \
#     "({}% of total expressed genes) {}".format(num_exp_genes, min_genes_spot_exp))
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
#     print("Dropped {} spots".format(num_spots - len(counts.index)))
          
    # Spots are columns and genes are rows
    counts = counts.transpose()
  
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots) 
#     print("Removing genes that are expressed in less than {} " \
#     "spots with a count of at least {}".format(min_features_gene, min_expression))
    counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
#     print("Dropped {} genes".format(num_genes - len(counts.index)))
    
    data=counts.transpose()
    temp = [val.split('x') for val in data.index.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])
    
    return coord,data



def write_result_to_csv(df,fileName):

    '''
    For convenience, user should output and save scGCO result dataframe with this function.
    Meanwhile, output result with scGCO can be resued and readed across platforms 
            with **read_result_to_dataframe()** function.
     More detail can see **read_result_to_dataframe()**.
    '''
    df_=df.copy()
    df_.nodes=df.nodes.apply(conver_list)
    df_.to_csv(fileName)

def read_result_to_dataframe(fileName,sep=','):
    
    '''
    Read and use scGCO output file cross-platform .
    More detail can see **write_result_to_csv()**.
    '''
    converters={"p_value":converter,"nodes":converter}
    df=pd.read_csv(fileName,converters=converters,index_col=0,sep=sep)
    df.nodes=df.nodes.apply(conver_array)
    return df

def conver_list(x):
    return [list(xx) for xx in x ]

def conver_array(x):
    return [np.array(xx) for xx in x] 

def converter(x):
    #define format of datetime
    return literal_eval(x)


def normalize_count_cellranger(data,Log=True):
    '''
    normalize count as in cellranger
    
    :param file: data: A dataframe of shape (m, n);
    :rtype: data shape (m, n);
    '''
    normalizing_factor = np.sum(data, axis = 1)/np.median(np.sum(data, axis = 1))
    data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    if Log==True:
        data=log1p(data)
    else:
        data=data

    return data


def log1p(data):
    '''
    log transform normalized count data
    
    :param file: data (m, n); 
    :rtype: data (m, n);
    '''
    if not issparse(data):
        return np.log1p(data)
    else:
        return data.log1p()


## Adding points  ##20191010
def Find_line(locs,cellGraph,axis=1):

    '''
    group points by the same curve.
    axis=1 represents adding points on horizontal(row) with group_yy;
    axis=0 represents adding points on vertical(columns) with group_xx. 
    
    '''

    axis=axis
    locs_xy=locs[:,axis]
    xy_sort=sorted(locs_xy)
    group_=[]
    group_xy=[]
    
    edge_down=min(cellGraph[:,3])
    level=xy_sort[0]+edge_down

    for cc,xx in enumerate(xy_sort):

        if xx<=level:
            locs_index=np.argsort(locs_xy)[cc]
            group_.append(locs_index)

        else:
            group_xy.append(group_)
            group_=[]
            if cc<len(xy_sort)-1:
                level=xy_sort[cc+1]+edge_down
            else:
                level=xy_sort[cc]+edge_down
            if xx<=level:
                locs_index=np.argsort(locs_xy)[cc]
                group_.append(locs_index)

    group_xy.append(group_)   
    return group_xy


def Add_Points(group_xy,locs,axis=1):
    
    '''
    Add holes with missed points on the curve.
    axis=1 represents adding points on horizontal(row) with group_yy;
    axis=0 represents adding points on vertical(columns) with group_xx. 
    '''
    
    newPoints={}
    axis=axis
    ax=int(abs(1-axis))
    for k,curve in enumerate(group_xy):
        addNode=[]
        for gg in curve:
    #         plt.scatter(locs[gg,0],locs[gg,1])
            addNode.append(locs[gg,ax])  ## locs_x of group_yy[g]

        
        dist=[]   
        add_sort=sorted(addNode)
        for k in range(0,len(add_sort)-1):

            dd=add_sort[k+1]-add_sort[k]   
            dist.append(dd)

        for k in range(len(dist)):
            
            if dist[k]> 1.5*np.median(dist) and dist[k]<3*min(dist):
                #print(np.median(dist),min(dist))
                newPoint1=curve[np.argsort(addNode)[k]]
                newPoint2=curve[np.argsort(addNode)[k+1]]
                addPoints=str(newPoint1)+' '+ str(newPoint2)
              #  print(k,addPoints)
                x_=(locs[newPoint1,0],locs[newPoint2,0])
                y_=(locs[newPoint1,1],locs[newPoint2,1])
                #plt.scatter(x_,y_,c='b',s=100)
                newLocs=[reduce(operator.add,x_)/2,reduce(operator.add,y_)/2]
                newPoints[addPoints]=newLocs

    return newPoints

def Assign_exp_to_newPoints(locs,data_norm,newPoints):
    
    '''
    assign expression to newPoints with their neighbors nodes' expression.
    para: locs is the coordinate;
          data is the clean dataFrame expression with normalization.
          newPoints is dictionary, whose keys is the two sides nodes of addPoints
                     and values is the coordinate of newPoints.
    '''
    
    
    ## 1st, appending newPoints to data
    data_norm_temp =data_norm.copy()
    newPoints_sort=sorted(newPoints.items(), key=lambda d: d[1])  ## sort by values and reture keys and values list. 
    for n,di in enumerate(newPoints_sort):
        key=di[0]
        values=di[1]
        index_=str(values[0])+'x'+str(values[1])
        #print(key,index_)

        node1=int(key.split(sep=' ')[0])
        node2=int(key.split(sep=' ')[1])
        node=list([node1,node2])
        node_=[v for v in node if v< data_norm.shape[0]]
        Add_exp=np.median(data_norm.iloc[node_,],axis=0)
        data_norm_temp.loc[index_]=Add_exp

    locs_new=Get_coord(data_norm_temp)
    
    
    ## 2nd, update expression with newPoints neighbors node
    
    exp=data_norm_temp.iloc[:,1].values
    cellGraph_new=create_graph_with_weight(locs_new,exp)

    G_add = nx.Graph()
    tempGraph = cellGraph_new.copy()
    G_add.add_nodes_from(list(set(list(tempGraph[:,0].astype(np.int32)) + list(tempGraph[:,1].astype(np.int32)))))
    G_add.add_edges_from(tempGraph[:, 0:2].astype(np.int32))  

    
    data_norm_new=data_norm.copy()
    newPoints_sort=sorted(newPoints.items(), key=lambda d: d[1])  ## sort by values and reture keys and values list.
    
    for n,di in enumerate(newPoints_sort):
        key=di[0]
        values=di[1]
        index_=str(values[0])+'x'+str(values[1])
        #print(n,key,index_)
        
        nodeInd=np.where(data_norm_temp.index==index_)[0][0]
        node=list(G_add.neighbors(nodeInd)) ## adding points index
        #print(node)
        node_=[v for v in node if v< data_norm.shape[0]]
       # print(node_)
        Add_exp=np.median(data_norm.iloc[node_,:],axis=0)
        data_norm_new.loc[index_]=Add_exp
    
    locs_new=Get_coord(data_norm_new)
    
    return locs_new,data_norm_new

def Get_coord(data):
    '''
    row is spots/cells; columns is genes/samples
    '''
    #data=counts.transpose()
    temp = [val.split('x') for val in data.index.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])
    
    return coord

def AddPoints_and_update_data(locs,data_norm,cellGraph,axis=1):
    
    '''
    Fix the holes on graph cuts by adding points.
    axis=1 represents adding points on horizontal(row) with group_yy;
    axis=0 represents adding points on vertical(columns) with group_xx. 
    
    '''
    
    ## 1. finding holes on row or columns.
    axis=axis
    group_xy=Find_line(locs,cellGraph,axis)
           
    ## 2. adding missed points
    
    newPoints=Add_Points(group_xy,locs,axis)
       
    
    ## 3. assign exprossion for newPoints
    locs_new,data_norm_new=Assign_exp_to_newPoints(locs,data_norm,newPoints)
    
    return locs_new,data_norm_new,newPoints


def AddPoints_XY_and_update_data(locs,data_norm,cellGraph,axis=1):
    '''
    Adding points as horizontal and vertical;
    return new locs, data_norm and newPoints.
    axis=1 represents adding points on horizontal(row) with group_yy;
    axis=0 represents adding points on vertical(columns) with group_xx. 
    
    '''
    newPoints={}
    locs_new_temp,data_norm_new_temp,newPoints_temp = AddPoints_and_update_data(locs,data_norm,cellGraph,axis=1)
    newPoints.update(newPoints_temp)
    
    i=1
    while len(newPoints_temp) >0:
        i=i+1
        ## 3rd, adding points on vertical with group_xx.
        exp_new=data_norm_new_temp.iloc[:,1].values
        cellGraph_new=create_graph_with_weight(locs_new_temp,exp_new)

        group_xy=Find_line(locs_new_temp,cellGraph_new,axis=int(i%2))
        newPoints_temp =Add_Points(group_xy,locs_new_temp,axis=int(i%2))
      
        
        ## 4th, update newPoints and locs_new as x_axis. 
        newPoints.update(newPoints_temp)
        locs_new_temp,data_norm_new_temp=Assign_exp_to_newPoints(locs_new_temp,data_norm,newPoints)
     
    return locs_new_temp,data_norm_new_temp,newPoints
    



def update_result_df_delPoints(result_df_new,data_norm_new,data_norm):
    '''
    Update result_df_new with added points.
    return the same spots/cells result_df.
    
    '''
    noisePoints=[ val for val in data_norm_new.index.values if val not in data_norm.index.values]
    noiseInd=[np.where(val==data_norm_new.index.values)[0][0] for val in noisePoints if val in data_norm_new.index.values]
    print('del_points_number: ',len(noiseInd))
    
    nodes_array=[]
    newLabels_array=[]
    p_array = []
    genes=[]
    for geneID in result_df_new.index.values:  #range(result_df_new.shape[0]):
        # print(geneID)
        genes.append(geneID)
        node_=result_df_new.loc[geneID,'nodes']
        p = result_df_new.loc[geneID,'p_value']
        Labels=result_df_new.loc[geneID,][4:].values
        newLabels=[Labels[v] for v in range(len(Labels)) if v not in noiseInd]
        newLabels_array.append(newLabels)

        nodes=[]
        zeroInd = []
        for n in range(len(node_)):
            node=np.array([v for v in node_[n] if v not in noiseInd])
            if len(node)>0:
                nodes.append(node)
            else:
                zeroInd.append(n)
        nodes_array.append(nodes)
        if len(zeroInd)>0:
            zeroInd.sort(reverse=True)
            for nn in zeroInd:
                del p[nn]
        p_array.append(p)
        
    labels_array = np.array(newLabels_array).reshape(len(genes), data_norm.shape[0])
    data_array = np.array((genes,p_array, nodes_array), dtype=object).T
    t_array = np.hstack((data_array, labels_array))
    c_labels = ['p_value','nodes']
    for i in np.arange(labels_array.shape[1]) + 1:
        temp_label = 'label_cell_' + str(i)
        c_labels.append(temp_label)
    temp_df = pd.DataFrame(t_array[:,1:], index=t_array[:,0], 
                      columns=c_labels)

    result_df_delPoints = pd.concat([result_df_new.iloc[:,1:3],temp_df],axis=1)
    return result_df_delPoints