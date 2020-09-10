import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition 
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN,KMeans
import pygco

from .Graph_cut import compute_pairwise_cost


# clustering genes

def spatial_pca_tsne_kmeans_cluster_gene(data_norm, gene_lists,marker_genes, perplexity = 30,fileName=None):

    '''
    perform standard PCA + tsne
    :param file: data_norm: normalized gene expression; gene_lists: list shape(k,)
        perplexity = 30
    :rtype: tsne_proj: shape (m, 2)
    '''          
    data_s = StandardScaler().fit_transform(data_norm.loc[:, gene_lists])
    pca = decomposition.PCA()
    pca.fit(data_s.T)
    pca_proj = pca.fit_transform(data_s.T)
    num_comp = np.where(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_)
                    > 0.9)[0][0]

#    RS=20180824
    tsne=manifold.TSNE(n_components=2, perplexity=perplexity)
    tsne_proj = tsne.fit_transform(pca_proj[:,0:num_comp])
    print(tsne_proj.shape)
#    tsne_proj = tsne.fit_transform(pca_proj[:,0:num_comp])
    tsne_proj_df=pd.DataFrame(index=gene_lists)
    tsne_proj_df["TSNE1"]=tsne_proj[:,0]
    tsne_proj_df["TSNE2"]=tsne_proj[:,1]
    init = tsne_proj_df.loc[marker_genes].values
    num_clusters=len(marker_genes)
    kmeans=KMeans(n_clusters=num_clusters,init = init, random_state=0).fit(tsne_proj)

    tsne_proj_df["cluster"]=kmeans.labels_
    gene_subset_lists=list()
    for geneID in marker_genes:
        gene_subset = tsne_proj_df.index[np.where(tsne_proj_df.cluster == tsne_proj_df.loc[geneID,"cluster"])]
        gene_subset_lists.append(gene_subset)

    for i ,gene_subset in enumerate(gene_subset_lists):
        tsne_proj_df.loc[gene_subset,"cluster"]=i
        
    final_labels = tsne_proj_df.cluster.values
    final_tsne = np.c_[tsne_proj, final_labels]
    palette = sns.color_palette('deep', final_labels.max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in final_tsne[:,2].astype(int)]
    plt.scatter(final_tsne[:,0], final_tsne[:,1], c=colors, s=28)
    plt.xlabel('TSNE component 1')
    plt.ylabel('TSNE component 2')
    for i in final_labels:
        position = np.max(final_tsne[ final_tsne[:,2]== i], axis=0)
        plt.gcf().gca().text(position[0], position[1]-1,str(i), fontsize=12)
    if fileName != None:
        plt.savefig(fileName,format="pdf",dpi=300)
    plt.show()
    return tsne_proj_df


def dbScan(tsne_proj, z, threshold, eps=1):
    '''
    deprecated
    '''      

    fig, ax = plt.subplots(figsize = (6, 6))
    tsne_proj_sel = tsne_proj[z > threshold]
    db = DBSCAN(eps=eps, min_samples=5).fit(tsne_proj_sel)
    ax.scatter(tsne_proj_sel[:,0], tsne_proj_sel[:,1], c=db.labels_, marker='.')
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    for i in np.arange(n_clusters_):
        position = np.max(tsne_proj_sel[db.labels_ == i], axis=0)
        plt.gcf().gca().text(position[0], position[1]-1,str(i), fontsize=12)
    plt.show()
    return tsne_proj_sel, db.labels_


def spatial_pca_tsne(data_norm, gene_lists, perplexity = 30, random_state=None):
    '''
    perform standard PCA + tsne
    :param file: data_norm: normalized gene expression; gene_lists: list shape(k,)
        perplexity = 30 
    :rtype: tsne_proj: shape (m, 2)
    '''           
    data_s = StandardScaler().fit_transform(data_norm.loc[:, gene_lists])  ## Input matrix (n_sample,n_feature)
    pca = decomposition.PCA()
    pca.fit(data_s.T)
    pca_proj = pca.fit_transform(data_s.T)
    num_comp = np.where(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_) 
                    > 0.9)[0][0]

#    RS=20180824
    tsne=manifold.TSNE(n_components=2,random_state=None, perplexity=perplexity)
    tsne_proj = tsne.fit_transform(pca_proj[:,0:num_comp])
    return tsne_proj




#clustering cells/spots

def create_labels(locs,data_norm,geneList,cellGraph, cluster_size=5,unary_scale_factor=100,smooth_factor=10,rs=0):
    
    
    X=data_norm.loc[:,geneList]
    
    cluster_KM=cluster_size
    kmeans=KMeans(n_clusters=cluster_KM,random_state=rs).fit(X)
    hmrf_labels = cut_graph_profile(cellGraph, kmeans.labels_, unary_scale_factor=unary_scale_factor,
                  smooth_factor=smooth_factor) ## smooth_factor can adjust
    return kmeans.labels_,hmrf_labels

def cut_graph_profile(cellGraph, Kmean_labels, unary_scale_factor=100, 
                      smooth_factor=50, label_cost=10, algorithm='expansion'):
    '''
    Returns new labels and gmm for the cut.
    
    :param points: cellGraph (n,3); count: shape (n,); 
    :unary_scale_factor, scalar; smooth_factor, scalar; 
    :label_cost: scalar; algorithm='expansion'
    :rtype: label shape (n,); gmm object.
    '''

    smooth_factor = smooth_factor
    unary_scale_factor = unary_scale_factor
    label_cost = label_cost
    algorithm = algorithm
    uniq, count = np.unique(Kmean_labels, return_counts = True)  
    unary_cost = compute_unary_cost_profile(Kmean_labels, unary_scale_factor)
    pairwise_cost = compute_pairwise_cost(len(uniq), smooth_factor)
    edges = cellGraph[:,0:2].astype(np.int32)
    labels = pygco.cut_from_graph(edges, unary_cost, pairwise_cost, label_cost)
#    energy = compute_energy(unary_cost, pairwise_cost, edges, labels)

    return labels


def compute_unary_cost_profile(Kmean_labels, scale_factor):
    '''
    Returns unary cost energy.
    
    :param points: count: shape (n,); gmm: gmm object; scale_factor: scalar

    :rtype: unary energy matrix.
    '''    
    labels_pred = Kmean_labels
    uniq, count = np.unique(Kmean_labels, return_counts = True)    
    uninary_mat = np.zeros((len(labels_pred), len(uniq)))
    for i in np.arange(uninary_mat.shape[0]):
        for j in np.arange(len(uniq)):
            if uniq[j] == labels_pred[i]:
                uninary_mat[i, j] = -1
            else:
                uninary_mat[i, j] = 1   
    return (scale_factor*uninary_mat).astype(np.int32)

def count_neighbors(a, b, cellGraph):
    
    idx0 = np.in1d(cellGraph[:,0], a).nonzero()[0]
    idx1 = np.in1d(cellGraph[:,1], a).nonzero()[0]
    neighbor0 = cellGraph[idx0, 1]
    neighbor1 = cellGraph[idx1, 0]
    neighbors = set(neighbor0.tolist() + neighbor1.tolist())   
    
    return len(neighbors.intersection(set(b)))


