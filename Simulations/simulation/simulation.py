import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, KDTree, ConvexHull
from scipy.spatial.distance import cdist
import scGCO
from sklearn import mixture




class Points:
    
    def __init__(self,numCell,numGenes):
        self.numCell=numCell
        self.numGenes=numGenes
    
    def grid(self):
        n_x = int(np.sqrt(self.numCell))
        n_y = n_x
        self.numCell = n_x * n_y
        x = np.arange(n_x)
        y = np.arange(n_y)
        self.xv,self.yv = np.meshgrid(x,y)
        grid_points = np.hstack((self.xv.reshape(-1,1),self.yv.reshape(-1,1)))
        self.locs =grid_points
    
    def mob(self,sample_info):
        locs = sample_info[["x","y"]].values
        self.locs=locs
        self.numCell = locs.shape[0]
        
    def padding(self,sample_info):
        # first estimate mean distance between points--
        locs = sample_info[["x","y"]].values
        points =locs
        p_dist = cdist(points, points)    
        p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
        norm_dist = np.mean(np.min(p_dist, axis = 0))

        # find points at edge, add three layers of new points 
        x_min = np.min(points, axis = 0)[0] - 0*norm_dist
        y_min = np.min(points, axis = 0)[1] - 0*norm_dist
        x_max = np.max(points, axis = 0)[0] + 0*norm_dist
        y_max = np.max(points, axis = 0)[1] + 0*norm_dist

        n_x = int((x_max - x_min)/norm_dist) + 1
        n_y = int((y_max - y_min)/norm_dist) + 1

        # create a mesh
        x = np.around(np.linspace(x_min, x_max, n_x), decimals=3)
        y = np.around(np.linspace(y_min, y_max, n_y), decimals=3)
        xv, yv = np.meshgrid(x, y)
        # now select points outside of hull, and merge
        hull = Delaunay(points)
        grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
#         pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
#         pad_dist = cdist(pad_points, points)   
#         pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
#         all_points = np.vstack((points, pad_points))
        
        self.norm_dist = norm_dist
        self.locs = grid_points
        
        sample_info_padding = pd.DataFrame(grid_points,columns=["x","y"])
        sample_info_padding.index = [*map(lambda x:str(x[0])+"x"+str(x[1]),grid_points)]
        
        np.random.seed(1)
        total_counts = np.random.choice(sample_info["total_counts"].values,size=grid_points.shape[0])
        point_dist = cdist(points,grid_points)
        
        point_ind,grid_ind = np.where(point_dist< 0.6* norm_dist)
        total_counts[grid_ind] = sample_info["total_counts"].values[point_ind]
        sample_info_padding["total_counts"] = total_counts
        
        self.numCell = grid_points.shape[0]
        
        return sample_info_padding
    
    
    def poisson_point(self):
        #Simulation window parameters
        xMin=0;xMax=1;
        yMin=0;yMax=1;
        xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
        areaTotal=xDelta*yDelta

        #Point process parameters
        lambda0=self.numCell; #intensity (ie mean density) of the Poisson process

        #Simulate Poisson point process
        numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
        xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
        yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points
        
        self.numCell = numbPoints
        self.locs = np.hstack((xx,yy))
        
    def plot_cellGraph(self):
        
        cellGraph = scGCO.create_graph_with_weight(self.locs,np.array([1]*self.numCell))
        fig,ax =plt.subplots(1,1,figsize=(5,5))
        ax.set_aspect("equal")
        ax.scatter(self.locs[:,0],self.locs[:,1],s=1,color="black")
        for i in np.arange(cellGraph.shape[0]):
            x=(self.locs[int(cellGraph[i,0]),0],self.locs[int(cellGraph[i,1]),0])
            y=(self.locs[int(cellGraph[i,0]),1],self.locs[int(cellGraph[i,1]),1])
            ax.plot(x,y,color="black",linewidth=0.5)
    
    def add_markdist_hotpot(self,low_marks=0,high_marks=1,cell_proportion = 0.3):
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_half = (x_min + x_max) * 0.5
        y_half = (y_min + y_max) * 0.5
        
        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        all_dist = (x-x[center_point])**2 + (y-y[center_point])**2
        
        sortedDistIndicies = all_dist.argsort()
        
        high_ind = sortedDistIndicies[:int(self.numCell*cell_proportion)]
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks

    def add_markdist_S(self,low_marks=0,high_marks=1, cell_proportion = 0.3):
        A = cell_proportion* 0.5
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_half = (x_min + x_max) * 0.5
        y_half = (y_min + y_max) * 0.5

        x_len = x_max - x_min
        y_len = y_max - y_min

        mask = y> A*y_len*np.sin(2*np.pi*(x-x_half)/x_len) + y_half - A*y_len
        mask&= y< A*y_len*np.sin(2*np.pi*(x-x_half)/x_len) + y_half + A*y_len
        
        self.exp = np.zeros(self.numCell)+ low_marks
        self.exp[mask] = high_marks


    
    
    def add_markdist_circles(self,low_marks=0,high_marks=1,cell_proportion = 0.2):
        
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_1 = x_min + (x_max-x_min) * 0.2
        
        x_2 = x_min + (x_max-x_min) * 0.7
        
        y_half = (y_min + y_max) * 0.5
        
        center_point1 = np.argmin((x- x_1)**2 + (y-y_half)**2)
        
        all_dist1 = (x-x[center_point1])**2 + (y-y[center_point1])**2
        
        sortedDistIndicies1 = all_dist1.argsort()
        
        high_ind1 = sortedDistIndicies1[:int(self.numCell*cell_proportion)]
        
        
        center_point2 = np.argmin((x- x_2)**2 + (y-y_half)**2)
        
        all_dist2 = (x-x[center_point2])**2 + (y-y[center_point2])**2
        
        sortedDistIndicies2 = all_dist2.argsort()
        
        high_ind2 = sortedDistIndicies2[:int(self.numCell*cell_proportion)]
        
        high_ind = np.concatenate([high_ind1,high_ind2])
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
    
    
    def add_markdist_streak(self,low_marks=0,high_marks=1,cell_proportion = 0.2):
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_half = (x_min + x_max) * 0.5
        y_half = (y_min + y_max) * 0.5
        
        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        x_dist = np.abs(x[center_point] - x)
        
        sortedDistIndicies = x_dist.argsort()
        
        high_ind = sortedDistIndicies[:int(self.numCell*cell_proportion)]
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
    
    def add_markdist_streaks(self,low_marks=0,high_marks=1,cell_proportion = 0.2):
        
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_1 = x_min + (x_max-x_min) * 0.3
        
        x_2 = x_min + (x_max-x_min) * 0.7
        
        
        x_dist1 = np.abs(x_1 - x)
        
        sortedDistIndicies1 = x_dist1.argsort()
        
        high_ind1 = sortedDistIndicies1[:int(self.numCell*cell_proportion)]
        
        
        x_dist2 = np.abs(x_2 - x)
        
        sortedDistIndicies2 = x_dist2.argsort()
        
        high_ind2 = sortedDistIndicies2[:int(self.numCell*cell_proportion)]
        
        
        high_ind = np.concatenate([high_ind1,high_ind2])
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
    
    def add_markdist_annuluses(self,low_marks=0,high_marks=1,cell_proportion = 0.1):
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_1 = x_min + (x_max-x_min) * 0.2
        
        x_2 = x_min + (x_max-x_min) * 0.8
        y_half = (y_min + y_max) * 0.5
        
        
        
        center_point1 = np.argmin((x- x_1)**2 + (y-y_half)**2)
        
        all_dist1 = (x-x[center_point1])**2 + (y-y[center_point1])**2
        
        sortedDistIndicies1 = all_dist1.argsort()
        
        high_ind1 = sortedDistIndicies1[int(self.numCell*cell_proportion):int(self.numCell*2.5*cell_proportion)]
        
        
        center_point2 = np.argmin((x- x_2)**2 + (y-y_half)**2)
        
        all_dist2 = (x-x[center_point2])**2 + (y-y[center_point2])**2
        
        sortedDistIndicies2 = all_dist2.argsort()
        
        high_ind2 = sortedDistIndicies2[int(self.numCell*cell_proportion):int(self.numCell*2.5*cell_proportion)]
        
        high_ind = np.concatenate([high_ind1,high_ind2])
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
        
        
        
        
    def add_markdist_annulus(self,low_marks=0,high_marks=1,cell_proportion = 0.2):
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        
        x_half = (x_min + x_max) * 0.5
        y_half = (y_min + y_max) * 0.5
        
        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        all_dist = (x-x[center_point])**2 + (y-y[center_point])**2
        
        sortedDistIndicies = all_dist.argsort()
        
        high_ind = sortedDistIndicies[int(self.numCell*0.5*cell_proportion):int(self.numCell*2*cell_proportion)]
        
        
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks

    
    def add_markdist_swiss_roll(self,low_marks=0,high_marks=1):
        
        
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_half = (x_min + x_max) * 0.5
        y_half = (y_min + y_max) * 0.5

        x_len = x_max - x_min
        y_len = y_max - y_min

        R = y_len*0.1

        cell_proportion = np.pi * R**2/(x_len * y_len)


        x_half = x_min + x_len*0.58 -1
        y_half = y_min + y_len * 0.5

        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        all_dist = (x-x[center_point])**2 + (y-y[center_point])**2
        
        sortedDistIndicies = all_dist.argsort()
        
        high_ind1 = sortedDistIndicies[int(self.numCell*cell_proportion):int(self.numCell*5*cell_proportion)]
        mask = y[high_ind1] > 13
        mask&= x[high_ind1] < 17
        high_ind1 = high_ind1[~mask]
        
        
        x_half = x_half + 1*R -1
        y_half = y_min + y_len * 0.55

        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        all_dist = (x-x[center_point])**2 + (y-y[center_point])**2

        sortedDistIndicies = all_dist.argsort()
        
        high_ind2 = sortedDistIndicies[int(self.numCell*10*cell_proportion):int(self.numCell*20*cell_proportion)]
        mask = y[high_ind2] < 12
        high_ind2 = high_ind2[~mask]
        
        
        
        
        x_half = x_half + 0.4*R-2 
        y_half = y_min + y_len * 0.7

        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        all_dist = (x-x[center_point])**2 + (y-y[center_point])**2

        sortedDistIndicies = all_dist.argsort()
        
        high_ind3 = sortedDistIndicies[int(self.numCell*23.9*cell_proportion):int(self.numCell*33*cell_proportion)]
        
        mask = y[high_ind3] > 18
        high_ind3 = high_ind3[~mask]
        
        high_ind = np.unique(np.concatenate([high_ind1,high_ind2,high_ind3]))
        
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks








    
    def add_markdist_squares(self,low_marks=0,high_marks=1):
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        square1_mask = (x >= x_min) & (x<= x_min + 0.2*(x_max - x_min)) 
        square1_mask &= (y >= y_min) & (y <= y_min + 0.2*(y_max - y_min))
        
        high_ind1 = np.where(square1_mask==True)[0]
        
        
        square2_mask = (x >= x_min + 0.4*(x_max - x_min)) & (x<= x_min + 0.6*(x_max - x_min)) 
        square2_mask &= (y >= y_min + 0.4*(y_max - y_min)) & (y <= y_min + 0.6*(y_max - y_min))
        
        high_ind2 = np.where(square2_mask==True)[0]
        
        
        square3_mask = (x >= x_min + 0.8*(x_max - x_min)) & (x<= x_max ) 
        square3_mask &= (y >= y_min + 0.8*(y_max - y_min)) & (y <= y_max)
        
        high_ind3 = np.where(square3_mask==True)[0]
        
        
#         square4_mask = (x > x_min + 0.8*(x_max - x_min)) & (x< x_max ) 
#         square4_mask &= (y > y_min) & (y < y_min + 0.2*(y_max - x_min))
        
#         high_ind4 = np.where(square4_mask==True)[0]
        
        
#         square5_mask = (x > x_min) & (x< x_min + 0.2*(x_max - x_min)) 
#         square5_mask &= (y > y_min + 0.8*(y_max - x_min)) & (y < y_max)
        
#         high_ind5 = np.where(square5_mask==True)[0]
        
        
        high_ind = np.concatenate([high_ind1,high_ind2,high_ind3])
        
        
        
        
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
    
    
    def add_markdist_pattern1(self,low_marks=0,high_marks=1):
        
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        square1_mask = (x > x_min + 0.6*(x_max - x_min)) & (x< x_min + 0.8*(x_max - x_min)) 
        square1_mask &= (y > y_min + 0.2*(y_max - x_min)) & (y < y_min + 0.8*(y_max - x_min))
        
        high_ind1 = np.where(square1_mask==True)[0]
        
        
        square2_mask = (x > x_min + 0.2*(x_max - x_min)) & (x< x_min + 0.4*(x_max - x_min)) 
        square2_mask &= (y > y_min + 0.2*(y_max - x_min)) & (y < y_min + 0.8*(y_max - x_min))
        
        high_ind2 = np.where(square2_mask==True)[0]
        
        high_ind = np.concatenate([high_ind1,high_ind2])
        
        
        
        
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
    
    
    def add_markdist_pattern2(self,low_marks=0,high_marks=1,cell_proportion=0.05):
        
        x = self.locs[:,0]
        y = self.locs[:,1]
        
        x_min = np.min(x)
        y_min = np.min(x)
        
        x_max = np.max(x)
        y_max = np.max(y)
        
        square1_mask = (x > x_min + 0.5*(x_max - x_min)) & (x< x_min + 0.6*(x_max - x_min)) 
        
        high_ind1 = np.where(square1_mask==True)[0]
        
        
        
        
        center_point1 = np.argmin((x- x_min)**2 + (y-y_min)**2)
        all_dist1 = (x-x[center_point1])**2 + (y-y[center_point1])**2
        
        sortedDistIndicies1 = all_dist1.argsort()
        
        high_ind2 = sortedDistIndicies1[:int(self.numCell*cell_proportion)]
        
        
        
        center_point2 = np.argmin((x- x_max)**2 + (y-y_min)**2)
        
        all_dist2 = (x-x[center_point2])**2 + ((y-y[center_point2])/3)**2
        
        sortedDistIndicies2 = all_dist2.argsort()
        
        high_ind3 = sortedDistIndicies2[:int(self.numCell*1.5*cell_proportion)]
        
        
        high_ind = np.concatenate([high_ind1,high_ind2,high_ind3])
        
        
        
        
        
        low_ind = np.setdiff1d(np.arange(self.numCell),high_ind)
        
        self.exp = np.zeros(self.numCell)+ low_marks
        
        self.exp[high_ind] = high_marks
        
    
    def add_markdist_pattern3(self):
        self.add_markdist_pattern1(0,1)
        p1_exp = self.exp
        
        self.add_markdist_pattern2(0,1)
        p2_exp = self.exp
        
        p3_exp = 1- (p1_exp+p2_exp)
        self.exp = p3_exp
    
        
    
    def add_markdist_gradient(self):
        x = self.locs[:,0]
        y = self.locs[:,1]
        x_max = np.max(x)
        y_max = np.max(y)
        
        x_half = x_max * 0.5
        y_half = y_max * 0.5
        
        center_point = np.argmin((x- x_half)**2 + (y-y_half)**2)
        
        out = self.exp.copy()
        
        pass
    
    
        
    
    def set_mob_pattern(self,sample_info,tissue_mat,pattern=0):
        exp = np.zeros(self.locs.shape[0])
        tissue_ind = np.where(tissue_mat[pattern]==1)
        locs = sample_info[["x","y"]].values
        tissue_dist = cdist(locs[tissue_ind],self.locs)
        #_,match_ind = np.where(tissue_dist< dist_cutoff* self.norm_dist)
        match_ind = np.argmin(tissue_dist,axis=1)
        exp[match_ind] = 1
        self.exp = exp
    
    def set_mob_pattern_new(self,sample_info,tissue_mat,pattern=0,dist_cutoff=0.6):
        exp = np.zeros(self.locs.shape[0])
        tissue_ind = np.where(tissue_mat[pattern]==1)
        locs = sample_info[["x","y"]].values
        tissue_dist = cdist(locs[tissue_ind],self.locs)
        _,match_ind = np.where(tissue_dist< dist_cutoff* self.norm_dist)
        #match_ind = np.argmin(tissue_dist,axis=1)
        exp[match_ind] = 1
        self.exp = exp
    
    def set_pattern(self,a):
        self.exp = a
    
    def plot_scatter_with_exp(self,ax=None):
        colors = ["g","r"]
        if ax:
            ax.scatter(self.locs[:,0],self.locs[:,1],c=self.exp,cmap=matplotlib.colors.ListedColormap(colors))
        else:
            fig,ax =plt.subplots(1,1,figsize=(5,5))
            ax.scatter(self.locs[:,0],self.locs[:,1],c=self.exp,cmap=matplotlib.colors.ListedColormap(colors))
            ax.set_axis_off()



def gmm_model2(data_norm,numGenes=1000,max_iters=5):
    gmmDict_={}
    for i,geneID in enumerate(data_norm.columns):
        count=data_norm.loc[:,geneID].values
        if i < numGenes:
            gmm=find_mixture_2(count)
            iters = 0
            while gmm.covariances_[0]/gmm.covariances_[1] >1e3 or gmm.covariances_[0]/gmm.covariances_[1] <1e-3 or iters <= max_iters:
                gmm=find_mixture_2(count)
                iters +=1
        else:
            gmm=find_mixture_2(count)
        
        gmmDict_[geneID]=gmm
    return gmmDict_


def find_mixture_2(data):
    '''
    estimate expression clusters, use k=2
    
    :param file: data (n,); 
    :rtype: gmm object;
    '''
    gmm = mixture.GaussianMixture(n_components = 2)
    gmm.fit(data.reshape(-1,1))

    return gmm

def find_mixture_2_new(data, max_iters=5):
    gmm = mixture.GaussianMixture(n_components = 2)
    gmm.fit(data.reshape(-1,1))
    iters = 0
    while gmm.covariances_[0]/gmm.covariances_[1] >1e3 or gmm.covariances_[0]/gmm.covariances_[1] <1e-3 or iters <= max_iters:
        gmm = mixture.GaussianMixture(n_components = 2)
        gmm.fit(data.reshape(-1,1))
        iters +=1
    return gmm