import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, KDTree, ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde

import seaborn as sns
from scipy.spatial.distance import cdist
from .Graph_cut import *

def plot_tsne(tsne_locs,tsne_labels,fileName=None):
    palette = sns.color_palette('deep', tsne_labels.max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in tsne_labels.astype(int)]
    plt.scatter(tsne_locs[:,0],tsne_locs[:,1], c=colors, s=28)
    plt.xlabel('TSNE component 1')
    plt.ylabel('TSNE component 2')
    for i in tsne_labels:
        position = np.max(tsne_locs[tsne_labels== i], axis=0)
        plt.gcf().gca().text(position[0], position[1]-1,str(i), fontsize=12)

    if fileName !=None:
        plt.savefig(fileName)
    plt.show()

def visualize_tsne_density(tsne_proj, threshold=0.001, bins=100, fileName=None,title=None,ax=None,fig=None):
    '''
    perform kde density estimationg for tsne projection to visualize genes clusters.
    :param file: tsne_proj: shape (m, 2)
    threshold=0.001, bins=100, fileName=None
    '''  
   # fig,ax=plt.subplots()
    tsne_proj=tsne_proj.copy()
    kde = gaussian_kde(tsne_proj.T, bw_method = 'scott')
    z = kde(tsne_proj.T)    
    x = np.ma.masked_where(z > threshold, tsne_proj[:,0])
    y = np.ma.masked_where(z > threshold, tsne_proj[:,1])

    # plot unmasked points
    ax.scatter(list(tsne_proj[:,0]), list(tsne_proj[:,1]), c='black', marker='o', s=5)

    # get bounds from axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # prepare grid for density map
    xedges = np.linspace(xmin, xmax, bins)
    yedges = np.linspace(ymin, ymax, bins)
    xx, yy = np.meshgrid(xedges, yedges)
    gridpoints = np.array([xx.ravel(), yy.ravel()])

    # compute density map
    zz = np.reshape(kde(gridpoints), xx.shape)

    # plot density map
    im = ax.imshow(zz, cmap='Spectral_r', interpolation='nearest',
               origin='lower', extent=[xmin, xmax, ymin, ymax],
                  aspect='auto')
    # plot threshold contour
    cs = ax.contour(xx, yy, zz, levels=[threshold], colors='black', line_width=10)
    # show
    fig.colorbar(im,ax=ax)   
    if title !=None:
        ax.set_title(title,fontsize=12)

    if fileName != None:
        plt.savefig(fileName)
   # plt.show()
    return z


def visualize_spatial_genes(df, locs, data_norm, point_size= 0.5):
    '''
    plot Voronoi tessellation of cells, highlight boundaries of graph cut
    
    :param file: df: dataframe of graph cut results; locs: spatial coordinates (n, 2);
    data_norm: normalized count: shape (n, m); 
    point_size = 0.5; 
    '''    

    i = 0
    while i < df.shape[0]:
        plt.figure(figsize=(6,2.5), dpi=300)
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)

        geneID = df.index[i]
        exp =  data_norm.loc[:,geneID].values
        best_Labels = df.loc[geneID,][5:].values.astype(int)
        subplot_voronoi_boundary(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=p1, 
                                 fdr=True, point_size = point_size, class_line_width=2)
        i = i + 1
        if i < df.shape[0]:
            geneID = df.index[i]
            exp =  data_norm.loc[:,geneID].values
            best_Labels = df.loc[geneID,][5:].values.astype(int)
            subplot_voronoi_boundary(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=p2, fdr=True, 
                                     point_size = point_size)    
        else:
            p2.axis('off')
        plt.show()
        i= i + 1


def plot_voronoi_boundary(geneID, coord, count, classLabel, p, fdr=False, 
                            fileName=None, point_size=5,
                          line_colors="k", class_line_width=2.5,
                          line_width=0.5, line_alpha=1.0,**kw):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n);
                predicted cell class calls shape (n); prediction p-value.
                fdr=False; line_colors = 'k'; class_line_width = 3; 
                line_width = 0.5; line_alpha = 1.0
    '''
    points = coord
    count = count
    newLabels =classLabel

    # first estimate mean distance between points--
    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))

    # find points at edge, add three layers of new points 
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    n_x = int((x_max - x_min)/norm_dist) + 1
    n_y = int((y_max - y_min)/norm_dist) + 1

    # create a mesh
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    xv, yv = np.meshgrid(x, y)
    # now select points outside of hull, and merge
    hull = Delaunay(points)
    grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
    pad_dist = cdist(pad_points, points)   
    pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
    all_points = np.vstack((points, pad_points))

    ori_len = points.shape[0]
    vor = Voronoi(all_points)

    if kw.get("show_points",True):
        plt.plot(points[0:ori_len,0], points[0:ori_len,1], ".", markersize=point_size)     
    patches = []
    # but we onl use the original points fot plotting
    for i in np.arange(ori_len):
        good_ver = vor.vertices[vor.regions[vor.point_region[i]]]
        polygon = Polygon(good_ver, True)
        patches.append(polygon)

    pc = PatchCollection(patches, cmap= cm.PiYG, alpha=1)

    pc.set_array(np.array(count))

    plt.gca().add_collection(pc)
    # for loop for plotting is slow, consider to vectorize to speedup
    # doesn;t mater for now unless you have many point or genes
    finite_segments=[]
    boundary_segments=[]
    for kk, ii in vor.ridge_dict.items():
        if kk[0] < ori_len and kk[1] < ori_len:
            if newLabels[kk[0]] !=  newLabels[kk[1]]:
                boundary_segments.append(vor.vertices[ii])
            else:
                finite_segments.append(vor.vertices[ii])

    plt.gca().add_collection(LineCollection(boundary_segments,
                                            colors="k",           
                                            lw=class_line_width,
                                            alpha=1,
                                            linestyles="solid"))
    plt.gca().add_collection(LineCollection(finite_segments,
                                            colors=line_colors,
                                            lw=line_width,
                                            alpha=line_alpha,
                                            linestyle="solid"))

    plt.xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    plt.ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)

    # also remember to add color bar
    plt.colorbar(pc)
    
    if fdr:
            titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
            titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))
    
    titleText=kw.get("set_title",titleText)
    fontsize=kw.get("fontsize",12)
    plt.title(titleText, fontname="Arial", fontsize=fontsize)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    if fileName!=None:
        plt.savefig(fileName)
    plt.show() 



def pdf_voronoi_boundary(geneID, coord, count, classLabel, p ,fileName, fdr=False, point_size=5,
                          line_colors="k", class_line_width=2.5,
                          line_width=0.5, line_alpha=1.0,**kw):
    '''
    save spatial expression as voronoi tessellation to pdf
    highlight boundary between classes.
    
    :param file: geneID; spatial coordinates shape (n, 2); normalized count: shape (n);
                predicted cell class calls shape (n); prediction p-value; pdf fileName;
                fdr=False; line_colors = 'k'; class_line_width = 3; 
                line_width = 0.5; line_alpha = 1.0
    '''

    points = coord
    count = count
    newLabels =classLabel

    # first estimate mean distance between points--
    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))

    # find points at edge, add three layers of new points 
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    n_x = int((x_max - x_min)/norm_dist) + 1
    n_y = int((y_max - y_min)/norm_dist) + 1

    # create a mesh
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    xv, yv = np.meshgrid(x, y)
    # now select points outside of hull, and merge
    hull = Delaunay(points)
    grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
    pad_dist = cdist(pad_points, points)   
    pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
    all_points = np.vstack((points, pad_points))

    ori_len = points.shape[0]
    vor = Voronoi(all_points)

    if kw.get("show_points",True):
        plt.plot(points[0:ori_len,0], points[0:ori_len,1], ".", markersize=point_size)     
    patches = []
    # but we onl use the original points fot plotting
    for i in np.arange(ori_len):
        good_ver = vor.vertices[vor.regions[vor.point_region[i]]]
        polygon = Polygon(good_ver, True)
        patches.append(polygon)

    pc = PatchCollection(patches, cmap=cm.PiYG, alpha=1)

    pc.set_array(np.array(count))

    plt.gca().add_collection(pc)
    # for loop for plotting is slow, consider to vectorize to speedup
    # doesn;t mater for now unless you have many point or genes
    finite_segments=[]
    boundary_segments=[]
    for kk, ii in vor.ridge_dict.items():
        if kk[0] < ori_len and kk[1] < ori_len:
            if newLabels[kk[0]] !=  newLabels[kk[1]]:
                boundary_segments.append(vor.vertices[ii])
            else:
                finite_segments.append(vor.vertices[ii])

    plt.gca().add_collection(LineCollection(boundary_segments,
                                            colors="k",           
                                            lw=class_line_width,
                                            alpha=1,
                                            linestyles="solid"))
    plt.gca().add_collection(LineCollection(finite_segments,
                                            colors=line_colors,
                                            lw=line_width,
                                            alpha=line_alpha,
                                            linestyle="solid"))
    plt.xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    plt.ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)

    # also remember to add color bar
    plt.colorbar(pc)
    
    if fdr:
            titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
            titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))
    
    titleText=kw.get("set_title",titleText)
    fontsize=kw.get("fontsize",12)
    plt.title(titleText, fontname="Arial", fontsize=fontsize)
    plt.axis('off')
#    plt.xlabel('X coordinate')
#    plt.ylabel('Y coordinate')
    if fileName != None:
        plt.savefig(fileName)
    else:
        print('ERROR! Please supply a file name.')



def subplot_voronoi_boundary(geneID, coord, count, classLabel, p  ,ax ,fdr=False, point_size=5,
                          line_colors="k", class_line_width=2.5,
                          line_width=0.5, line_alpha=1.0,**kw):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; spatial coordinates (n, 2); normalized gene expression: count;
            predicted cell class calls (n); p_value; ax number;
    '''
    points = coord
    count = count
    newLabels =classLabel

    # first estimate mean distance between points--
    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))

    # find points at edge, add three layers of new points 
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    n_x = int((x_max - x_min)/norm_dist) + 1
    n_y = int((y_max - y_min)/norm_dist) + 1

    # create a mesh
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    xv, yv = np.meshgrid(x, y)
    # now select points outside of hull, and merge
    hull = Delaunay(points)
    grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
    pad_dist = cdist(pad_points, points)   
    pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
    all_points = np.vstack((points, pad_points))

    ori_len = points.shape[0]
    vor = Voronoi(all_points)

    if kw.get("show_points",True):
        ax.plot(points[0:ori_len,0], points[0:ori_len,1], ".", markersize=point_size)  

    ## plt.full(color)   
    patches = []
    # but we onl use the original points fot plotting
    for i in np.arange(ori_len):
        good_ver = vor.vertices[vor.regions[vor.point_region[i]]]
        polygon = Polygon(good_ver, True)
        patches.append(polygon)

    pc = PatchCollection(patches, cmap=cm.PiYG, alpha=1)

    pc.set_array(np.array(count))

    ax.add_collection(pc)


    # for loop for plotting is slow, consider to vectorize to speedup
    # doesn;t mater for now unless you have many point or genes
    finite_segments=[]
    boundary_segments=[]
    for kk, ii in vor.ridge_dict.items():
        if kk[0] < ori_len and kk[1] < ori_len:
            if newLabels[kk[0]] !=  newLabels[kk[1]]:
                boundary_segments.append(vor.vertices[ii])
            else:
                finite_segments.append(vor.vertices[ii])

    ax.add_collection(LineCollection(boundary_segments,
                                            colors="k",           
                                            lw=class_line_width,
                                            alpha=1,
                                            linestyles="solid"))
    ax.add_collection(LineCollection(finite_segments,
                                            colors=line_colors,
                                            lw=line_width,
                                            alpha=line_alpha,
                                            linestyle="solid"))
    ax.set_xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    ax.set_ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)

    # also remember to add color bar
    #plt.colorbar(pc)
    

    if fdr:
            titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
            titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))
    
    titleText=kw.get("set_title",titleText)
    fontsize=kw.get("fontsize",8)
    ax.set_title(titleText, fontname="Arial", fontsize=fontsize)



def subplot_voronoi_boundary_12x18(geneID, coord, count, 
                          classLabel, p, ax, fdr=False, point_size = 0.5,  
                          line_colors = 'k', class_line_width = 0.8, 
                          line_width = 0.05, line_alpha = 1.0,**kw):
    '''
    plot spatial expression as voronoi tessellation
    highlight boundary between classes
    
    :param file: geneID; coord: spatial coordinates (n, 2); count: normalized gene expression;
        predicted cell class calls (n); p: graph cut p-value. 
    '''
    points = coord
    count = count
    newLabels =classLabel

    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))

    # find points at edge, add three layers of new points 
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    n_x = int((x_max - x_min)/norm_dist) + 1
    n_y = int((y_max - y_min)/norm_dist) + 1

    # create a mesh
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    xv, yv = np.meshgrid(x, y)
    # now select points outside of hull, and merge
    hull = Delaunay(points)
    grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
    pad_dist = cdist(pad_points, points)   
    pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
    all_points = np.vstack((points, pad_points))

    ori_len = points.shape[0]
    vor = Voronoi(all_points)
    if kw.get("show_points",True):
        ax.plot(points[0:ori_len,0], points[0:ori_len,1], ".", markersize=point_size)     
    patches = []
    # but we onl use the original points fot plotting
    for i in np.arange(ori_len):
        good_ver = vor.vertices[vor.regions[vor.point_region[i]]]
        polygon = Polygon(good_ver, True)
        patches.append(polygon)

    pc = PatchCollection(patches, cmap=cm.PiYG, alpha=1)

    pc.set_array(np.array(count))

    ax.add_collection(pc)
    # for loop for plotting is slow, consider to vectorize to speedup
    # doesn;t mater for now unless you have many point or genes
    finite_segments=[]
    boundary_segments=[]
    for kk, ii in vor.ridge_dict.items():
        if kk[0] < ori_len and kk[1] < ori_len:
            if newLabels[kk[0]] !=  newLabels[kk[1]]:
                boundary_segments.append(vor.vertices[ii])
            else:
                finite_segments.append(vor.vertices[ii])

    ax.add_collection(LineCollection(boundary_segments,   ### boundary
                                            colors="k",           
                                            lw=class_line_width,
                                            alpha=1,
                                            linestyles="solid"))
    ax.add_collection(LineCollection(finite_segments,               ## other line in loop
                                            colors=line_colors,
                                            lw=line_width,
                                            alpha=line_alpha,
                                            linestyle="solid"))
    ax.set_xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    ax.set_ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)

    # also remember to add color bar
    #plt.colorbar(pc)

    if fdr:
        titleText = geneID + ' ' + '' + str("{0:.1e}".format(p))
    else:
        titleText = geneID + ' ' + 'p_value: ' + str("{0:1e}".format(p))
    titleText=kw.get("set_title",titleText)
    fontsize=kw.get("fontsize",3.5)
    ax.set_title(titleText, fontname="Arial", fontsize=fontsize, y = 0.85)



def multipage_pdf_visualize_spatial_genes(df, locs, data_norm, cellGraph, fileName, 
                     point_size=0.,**kw):
    '''
    save spatial expression as voronoi tessellation to pdf highlight boundary between classes
    format: 12 by 18.
    :param file: df: graph cuts results; locs: spatial coordinates (n, 2); data_norm: normalized gene expression;
        pdf filename; point_size=0.5. 
    '''    
 
    points = locs
    vor = Voronoi(points)

    nb_plots = int(df.shape[0])
    numCols = 12  
    numRows = 18
    nb_plots_per_page =  numCols*numRows
    t_numRows = int(df.shape[0]/numCols) + 1
    
    with PdfPages(fileName) as pdf:    
        for i in np.arange(df.shape[0]):
            if i % nb_plots_per_page == 0:
                fig, axs = plt.subplots(numRows, numCols, # 8 11
                                    figsize = (8,11))   
                fig.subplots_adjust(hspace=0.3, wspace=0.3,
                                top=0.925, right=0.925, bottom=0.075, left = 0.075)
                  
            geneID = df.index[i]
            exp =  data_norm.loc[:,geneID].values
            if np.isnan(df.loc[geneID,].fdr):
                best_Labels = np.zeros(data_norm.shape[0])
            else:
                best_Labels = df.loc[geneID,][5:].values.astype(int)
            m = int(i/numCols) % numRows
            n = i % numCols 
            ax = axs[m,n]
            subplot_voronoi_boundary_12x18(geneID, locs, exp, best_Labels,
                                 df.loc[geneID,].fdr, ax=ax, fdr=True,
                                 point_size = point_size,**kw)

            if (i + 1) % nb_plots_per_page == 0 or (i + 1) == nb_plots:
                for ii in np.arange(numRows):
                    for jj in np.arange(numCols):        
                        axs[ii,jj].axis('off')
                pdf.savefig(fig)
                fig.clear()
                plt.close()


def add_HE_image(image,ax):
    img=Image.open(image)
    extent_size = [1,33,1,35]
    img_transpose=img.transpose(Image.FLIP_TOP_BOTTOM)
    ax.imshow(img_transpose,extent=extent_size)


def subplot_boundary(geneID, coord, count, classLabel, p, ax=None,
                    fdr=False, point_size=5,
                    class_line_width=2.5,
                    **kw):
    
    '''
    plot spatial expression as voronoi tessellation.
    :param file: geneID; spatial coordinates (n, 2); normalized count: shape (n); 
    '''
    
    points = coord
    count = count
    newLabels =classLabel

    # first estimate mean distance between points--
    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))

    # find points at edge, add three layers of new points 
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    n_x = int((x_max - x_min)/norm_dist) + 1
    n_y = int((y_max - y_min)/norm_dist) + 1

    # create a mesh
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    xv, yv = np.meshgrid(x, y)
    # now select points outside of hull, and merge
    hull = Delaunay(points)
    grid_points = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    pad_points = grid_points[np.where(hull.find_simplex(grid_points)< 0)[0]]
    pad_dist = cdist(pad_points, points)   
    pad_points = pad_points[np.where(np.min(pad_dist, axis = 1) > norm_dist)[0]]
    all_points = np.vstack((points, pad_points))

    ori_len = points.shape[0]
    vor = Voronoi(all_points)

    if kw.get("show_points",False):
        ax.plot(points[0:ori_len,0], points[0:ori_len,1], ".", markersize=point_size)
    
    # for loop for plotting is slow, consider to vectorize to speedup
    # doesn;t mater for now unless you have many point or genes
    boundary_segments=[]
    for kk, ii in vor.ridge_dict.items():
        if kk[0] < ori_len and kk[1] < ori_len:
            if newLabels[kk[0]] !=  newLabels[kk[1]]:
                boundary_segments.append(vor.vertices[ii])

    ax.add_collection(LineCollection(boundary_segments,
                                            colors="k",           
                                            lw=class_line_width,
                                            alpha=1,
                                            linestyles="solid"))
    
    ax.set_xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    ax.set_ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)
    
    if fdr:
            titleText = geneID + '\n' + 'fdr: ' + str("{:.2e}".format(p))
    else:
            titleText = geneID + '\n' + 'p_value: ' + str("{:.2e}".format(p))
    
    titleText=kw.get("set_title",titleText)
    fontsize=kw.get("fontsize",8)
    ax.set_title(titleText, fontname="Arial", fontsize=8)


def plot_tissue_pattern(locs,data_norm,tissue_mat,image,colors,title,nrows=4,ncols=5,s=15):
        ## Task2: Tissue mat
    nb_plots=tissue_mat.shape[0]
    nrows=nrows
    ncols=ncols
    nb_box=nrows*ncols
  
    fig,ax=plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*3),dpi=180)
    fig.subplots_adjust(hspace=0.3, wspace=0.3,
                                    top=0.925, right=0.925, bottom=0.075, left = 0.075)
    
    for i in range(tissue_mat.shape[0]):
        x=int(i/ncols)
        y=i%ncols
        axes=ax[x,y]
        
        add_HE_image(image,axes)
        axes.scatter(locs[:,0], locs[:,1], c=tissue_mat[i],
                     cmap=matplotlib.colors.ListedColormap(colors) ,s=s)
        
        axes.set_title(title,fontsize=8)


        points=locs
        p_dist = cdist(points, points)    
        p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
        norm_dist = np.mean(np.min(p_dist, axis = 0))

        # find points at edge, add three layers of new points 
        x_min = np.min(points, axis = 0)[0] - 3*norm_dist
        y_min = np.min(points, axis = 0)[1] - 3*norm_dist
        x_max = np.max(points, axis = 0)[0] + 3*norm_dist
        y_max = np.max(points, axis = 0)[1] + 3*norm_dist

        axes.set_xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
        axes.set_ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)

        if (i + 1) == nb_plots:
            for ii in np.arange(nb_plots,nb_box):        
                    ax[int(ii/ncols),ii%ncols].axis('off')

def subplot_HE_with_labels(locs,labels,image,ax,colors,title,s=30):
    
   # import matplotlib
    add_HE_image(image,ax)  
    ax.scatter(locs[:,0], locs[:,1], c=labels,
                 cmap=matplotlib.colors.ListedColormap(colors) ,s=s)
    
    ax.set_title(title,fontsize=8)

    points=locs
    p_dist = cdist(points, points)    
    p_dist[p_dist == 0] = np.max(p_dist, axis = 0)[0]
    norm_dist = np.mean(np.min(p_dist, axis = 0))
    x_min = np.min(points, axis = 0)[0] - 3*norm_dist
    y_min = np.min(points, axis = 0)[1] - 3*norm_dist
    x_max = np.max(points, axis = 0)[0] + 3*norm_dist
    y_max = np.max(points, axis = 0)[1] + 3*norm_dist

    ax.set_xlim(x_min + 1*norm_dist, x_max - 1*norm_dist)
    ax.set_ylim(y_min + 1*norm_dist, y_max - 1*norm_dist)