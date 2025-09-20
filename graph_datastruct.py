#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:27:17 2022

@author: yigongqin
"""

import os
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 24})
coolwarm = cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256*100))
ly = np.array([255/256, 255/256, 255/256, 1])
newcolors[0, :] = ly
newcmp = ListedColormap(newcolors)
from scipy.spatial import Voronoi
from scipy.stats import truncnorm
from math import pi
#from shapely.geometry.polygon import Polygon
#from shapely.geometry import Point
from collections import defaultdict
import math
import argparse
import networkx as nx
from shapely.geometry import Polygon
from user_generate import user_defined_config
from shapely.geometry import Polygon, Point, LineString
from matplotlib.path import Path
import torch

def angle_norm(angular):

    return - ( 2*(angular + pi/2)/(pi/2) - 1 )

eps = 1e-12
def in_bound(x, y, max_y=1, cone_ratio = 0):
    
    if x>=-eps and x<=1+eps and y>=-eps + cone_ratio*(1-x) and y<=max_y - cone_ratio*(1-x) +eps:
        return True
    else:
        return False
    
def sort_polygon(coords):
    cx = sum([p[0] for p in coords]) / len(coords)
    cy = sum([p[1] for p in coords]) / len(coords)
            
    # Sort the points by the angle they make with the centroid
    sorted_coords = sorted(coords, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return sorted_coords

def periodic_move_p(p, pc):

    if p[0]<pc[0]-0.5-eps: p[0]+=1
    if p[0]>pc[0]+0.5+eps: p[0]-=1
    if p[1]<pc[1]-0.5-eps: p[1]+=1
    if p[1]>pc[1]+0.5+eps: p[1]-=1    



def periodic_move(p, pc):
    x,  y  = p
    xc, yc = pc
    """
    if x<xc-0.5-eps: x+=1
    if x>xc+0.5+eps: x-=1
    if y<yc-0.5-eps: y+=1
    if y>yc+0.5+eps: y-=1    
    """
    rel_x = x - xc
    rel_y = y - yc
    x += -1*(rel_x>0.5) + 1*(rel_x<-0.5) 
    y += -1*(rel_y>0.5) + 1*(rel_y<-0.5) 
    
    
    assert -0.5<x - xc<0.5
    assert -0.5<y - yc<0.5
    return [x, y]


def periodic_dist_(p, pc):
    
    x,  y  = p
    xc, yc = pc
    
    if x<xc-0.5-eps: x+=1
    if x>xc+0.5+eps: x-=1
    if y<yc-0.5-eps: y+=1
    if y>yc+0.5+eps: y-=1         
           
    return math.sqrt((x-xc)**2 + (y-yc)**2)


def counterclock(point, center):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-center[0], point[1]-center[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    angle = math.atan2(vector[1], vector[0])
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def angle_of_vector(v):
    """
    Return the angle (in degrees, between 0 and 360) of vector v = (vx, vy).
    """
    return math.degrees(math.atan2(v[1], v[0])) % 360

def all_angles_convex(P, P_prev, P_next, H, tol=1e-6):
    """
    Given a vertex P with adjacent vertices P_prev and P_next (from the pentagon),
    and a candidate hexagon vertex H (outside the pentagon),
    check if the three rays (from P to P_prev, P to P_next, and P to H) are arranged
    such that all circular gaps (angles) at P are less than 180 degrees.
    
    Returns True if all three gaps are less than 180°, False otherwise.
    """
    # Compute the three vectors originating at P.
    v_prev = (P_prev[0] - P[0], P_prev[1] - P[1])
    v_next = (P_next[0] - P[0], P_next[1] - P[1])
    v_H    = (H[0]    - P[0], H[1]    - P[1])
    
    # Get their angles (in degrees) in the range [0, 360).
    angles = [angle_of_vector(v_prev), angle_of_vector(v_next), angle_of_vector(v_H)]
    angles.sort()
    
    # Compute the three gaps (accounting for circularity)
    gap1 = angles[1] - angles[0]
    gap2 = angles[2] - angles[1]
    gap3 = angles[0] + 360 - angles[2]
    
    # Debug (optional): print(angles, gap1, gap2, gap3)
    return (gap1 < 180 - tol) and (gap2 < 180 - tol) and (gap3 < 180 - tol)


def plot_overlap_polygons(self, region_polygons, overlaps, timestep=None, fname=None, save_dir='overlap_plots'):
    os.makedirs(save_dir, exist_ok=True)
    if fname is None:
        if timestep is not None:
            fname = f"overlap_plot_t{timestep}.png"
        else:
            fname = "overlap_plot.png"
    full_path = os.path.join(save_dir, fname)
    fig, ax = plt.subplots(figsize=(8, 6))
    for region, poly in region_polygons.items():
        if region in overlaps and overlaps[region]:
            coords = list(poly.exterior.coords)
            patch = patches.Polygon(coords, closed=True,
                                            facecolor='red', edgecolor='black', alpha=0.5)
            ax.add_patch(patch)
            centroid = poly.centroid.coords[0]
            ax.text(centroid[0], centroid[1], str(region),
                fontsize=10, color='black', ha='center', va='center')
        else:
            coords = list(poly.exterior.coords)
            patch = patches.Polygon(coords, closed=True,
                                    facecolor='none', edgecolor='blue', alpha=0.5)
            ax.add_patch(patch)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(full_path, dpi=300)


def linked_edge_by_junction(j1, j2):
    
    
    if len(set(j1).intersection(set(j2)))==2:
        return True
    else:
        return False
    

def counterclock(point, center):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-center[0], point[1]-center[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    angle = math.atan2(vector[1], vector[0])
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def hexagonal_lattice(dx=0.05, noise=0.0001, BC='periodic', max_y = 1, cone_ratio = 0):
    # Assemble a hexagonal lattice
    rows, cols = int(1/dx)+1, int(1/dx)
    print('cols and rows of grains: ', cols, rows)
    shiftx, shifty = 0.1*dx, 0.25*dx
    points = []
    in_points = []
    randNoise = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*noise, size=rows*cols*5)
    count = 0
    for row in range(rows*2):
        for col in range(cols):
            count+=1
            x = ( (col + (0.5 * (row % 2)))*np.sqrt(3) )*dx + shiftx
            y = row*0.5 *dx + shifty
         
            x += randNoise[count,0]
            y += randNoise[count,1]
            
            if in_bound(x, y, max_y, cone_ratio):
              in_points.append([x,y])
              points.append([x,y])
              if BC == 'noflux':
                  points.append([-x,y])
                  points.append([2-x,y])
                  # mirror against CRx + y - CR = 0 
                  
                  points.append([-( 2*cone_ratio*y + (cone_ratio**2-1)*x -2*cone_ratio**2 )/(1+cone_ratio**2),
                                 -( (1-cone_ratio**2)*y +2*cone_ratio*x -2*cone_ratio )/(1+cone_ratio**2)])    
                  
                  #  mirror against CRx - y +max_y - CR = 0
                  points.append([-( -2*cone_ratio*y + (cone_ratio**2-1)*x +2*cone_ratio*(max_y - cone_ratio))/(1+cone_ratio**2),
                                 -( (1-cone_ratio**2)*y -2*cone_ratio*x -2*cone_ratio*(max_y - cone_ratio) )/(1+cone_ratio**2) ])
              if BC == 'periodic':
                  points.append([x+1,y])
                  points.append([x-1,y])
                  points.append([x,y+1])
                  points.append([x,y-1])                  
                  points.append([x+1,y+1])
                  points.append([x-1,y-1])
                  points.append([x-1,y+1])
                  points.append([x+1,y-1])                     

    return points, in_points


def random_lattice(dx=0.05, noise=0.0001, BC='periodic', max_y = 1, cone_ratio = 0):
    # Assemble a hexagonal lattice
    rows, cols = int(1/dx), int(1/dx)
    print('cols and rows of grains: ', cols, rows)
    points = []
    in_points = []
    randNoise = np.random.rand(rows*cols,2)
    count = 0
    for row in range(rows*cols):
  
            
            x = randNoise[count,0]
            y = randNoise[count,1]
            
            if in_bound(x, y, max_y, cone_ratio):
              in_points.append([x,y])
              points.append([x,y])
              if BC == 'noflux':
                  points.append([-x,y])
                  points.append([2-x,y])
                  # mirror against CRx + y - CR = 0 
                  
                  points.append([-( 2*cone_ratio*y + (cone_ratio**2-1)*x -2*cone_ratio**2 )/(1+cone_ratio**2),
                                 -( (1-cone_ratio**2)*y +2*cone_ratio*x -2*cone_ratio )/(1+cone_ratio**2)])    
                  
                  #  mirror against CRx - y +max_y - CR = 0
                  points.append([-( -2*cone_ratio*y + (cone_ratio**2-1)*x +2*cone_ratio*(max_y - cone_ratio))/(1+cone_ratio**2),
                                 -( (1-cone_ratio**2)*y -2*cone_ratio*x -2*(max_y - cone_ratio) )/(1+cone_ratio**2) ])
              if BC == 'periodic':
                  points.append([x+1,y])
                  points.append([x-1,y])
                  points.append([x,y+1])
                  points.append([x,y-1])                  
                  points.append([x+1,y+1])
                  points.append([x-1,y-1])
                  points.append([x-1,y+1])
                  points.append([x+1,y-1])                     
            count+=1
    return points, in_points



        
class graph:
    def __init__(self, lxd: float = 40, randInit: bool = True, seed: int = 1, noise: float = 0.01, BC: str = 'periodic',\
                 adjust_grain_size = False, adjust_grain_orien = False,
                 user_defined_config = None):
        
        if user_defined_config:
            self.BC = user_defined_config['boundary']
            
            self.lxd = user_defined_config['geometry']['lxd']
            self.lyd = self.lxd*user_defined_config['geometry']['yx_asp_ratio']
            self.lzd = self.lxd*user_defined_config['geometry']['zx_asp_ratio']
            self.ini_height = user_defined_config['geometry']['z0']
            self.final_height = self.ini_height + self.lzd 
            self.cone_ratio = user_defined_config['geometry']['cone_ratio']
            
            self.mesh_size = user_defined_config['initial_parameters']['mesh_size']          
            self.ini_grain_size = user_defined_config['initial_parameters']['grain_size_mean'] 
            self.seed = user_defined_config['initial_parameters']['seed'] 
            self.noise = user_defined_config['initial_parameters']['noise_level']       
            
        else:      
            self.BC = BC
            
            self.lxd = lxd
            self.lyd = self.lxd
            self.ini_height, self.final_height = 2, 50
            self.cone_ratio = 0
            
            self.mesh_size = 0.08           
            self.ini_grain_size = 4
            self.seed = seed
            self.noise = noise
            
            
        if adjust_grain_size:
            self.ini_grain_size = 2 + (seed%10)/5*3        
            
        self.patch_size = 40    
        self.patch_grid_size = int(round(self.patch_size/self.mesh_size))
        self.imagesize = (int(self.lxd/self.mesh_size)+1, int(self.lyd/self.mesh_size)+1)
        self.vertices = defaultdict(list) ## vertices coordinates
        self.vertex2joint = defaultdict(set) ## (vertex index, x coordiante, y coordinate)  -> (region1, region2, region3)
        self.vertex_neighbor = defaultdict(set)
        self.grain_neighbor=defaultdict(set)
        self.edges = []  ## index linkage
       # self.edge_len = []
        self.regions = defaultdict(list) ## index group
        self.region_coors = defaultdict(list)
        self.region_edge = defaultdict(set)
        self.region_center = defaultdict(list)
        self.vertex_coord_to_index = {}
        self.index_to_vertex_coord = {}
        self.elim_grain_to_vertices={}
        self.nucle_grain_to_vertices={}
        self.edge_prob_dict = {}
        self.junct_gradient_dict={}

        self.pentagon_region_id=defaultdict(list)
        self.added_region=defaultdict(list)
        self.nucle_fall_region=[]
        self.intersection_points=[]
        self.regions_edges=defaultdict(list)
        self.covered=[]
        self.replaced=[]
        self.cutted_edges=[]
        self.nucle_map =defaultdict(list)
        self.added_edge=[]
        self.projected_vertices=[]
        self.inter_point=[]
        self.region2newidx=defaultdict(set)

        self.quadruples = {}
        self.corner_grains = [0, 0, 0, 0]
       # self.region_coors = [] ## region corner coordinates
        self.density = self.ini_grain_size/self.lxd
        self.noise = self.noise/self.lxd/(self.lxd/self.patch_size)
        
        self.alpha_field = np.zeros((self.imagesize[1], self.imagesize[0]), dtype=int)  # use Image convention [ny, nx]
      #  self.alpha_field_dummy = np.zeros((2*self.imagesize[0], 2*self.imagesize[1]), dtype=int)
        self.error_layer = 0
        
        self.raise_err = True
        self.save = None

        
        if randInit:
            np.random.seed(seed)
            if self.BC == 'noflux':
                self.random_voronoi_noflux()
            elif self.BC == 'periodic':
                self.random_voronoi_periodic()
            else:
                raise KeyError()
            self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
            self.alpha_pde = self.alpha_field.copy()
            self.update(init=True)
            
 
            self.num_regions = len(self.regions)
            self.num_vertices = len(self.vertices)
            self.num_edges = len(self.edges)
            
            cur_grain, counts = np.unique(self.alpha_field, return_counts=True)
            self.area_counts = dict(zip(cur_grain, counts))

            
            # sample orientations
            ux = np.random.randn(self.num_regions)
            uy = np.random.randn(self.num_regions)
            uz = np.random.randn(self.num_regions)
            self.theta_x = np.zeros(1 + self.num_regions)
            self.theta_z = np.zeros(1 + self.num_regions)
            self.theta_x[1:] = np.arctan2(uy, ux)%(pi/2)
            
            if adjust_grain_orien:
                low, up = 0, pi/2
                mean, sd = 0+pi/36*(seed%10), 0.4
                gen = truncnorm((low - mean) / sd, (up - mean) / sd, loc=mean, scale=sd)
                self.theta_z[1:] = gen.rvs(self.num_regions)
            else:
                self.theta_z[1:] = np.arctan2(np.sqrt(ux**2+uy**2), uz)%(pi/2)

            self.layer_grain_distribution()    

    def layer_grain_distribution(self):
        
        grain_area = np.array(list(self.area_counts.values()))*self.mesh_size**2#*self.ini_height
        grain_size = np.sqrt(4*grain_area/pi)
       
        mu = np.mean(grain_size)
        std = np.std(grain_size)
        print('initial grain size mean', mu, ', standard deviation', std)
        print('max and min', np.max(grain_size), np.min(grain_size))
        
        self.ini_grain_dis = grain_size

    def plot_grain_distribution(self):
        bins = np.arange(0,20,1)
        
        dis, bin_edge = np.histogram(self.ini_grain_dis, bins=bins, density=True)
        bin_edge = 0.5*(bin_edge[:-1] + bin_edge[1:])
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(bin_edge, dis*np.diff(bin_edge)[0], 'b', label='GNN')
        ax.set_xlim(0, 20)
        ax.set_xlabel(r'$d\ (\mu m)$')
        ax.set_ylabel(r'$P$')  
      #  ax.legend(fontsize=15)  
        plt.savefig('seed'+str(self.seed)+'_ini_size_dis' +'.png', dpi=400, bbox_inches='tight')

        bins = np.arange(0,90,10)
        
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.hist(self.theta_z[1:]*180/pi, bins, density=False, facecolor='g', alpha=0.75, edgecolor='black')
        ax.set_xlim(0, 90)
        ax.set_xlabel(r'$\theta_z$')
       # ax.set_ylabel(r'$P$')  
      #  ax.legend(fontsize=15)  
        plt.savefig('seed'+str(self.seed)+'_ini_orien_dis' +'.png', dpi=400, bbox_inches='tight')



    def compute_error_layer(self):
        self.error_layer = np.sum(self.alpha_pde!=self.alpha_field)/len(self.alpha_pde.flatten())
        print('pointwise error at current layer: ', self.error_layer)

    def random_voronoi_periodic(self):

        mirrored_seeds, seeds = hexagonal_lattice(dx=self.density, noise = self.noise, BC = self.BC)
        vor = Voronoi(mirrored_seeds)     
    
       # regions = []
        reordered_regions = []
       # vertices = []
        vert_map = {}
        vert_count = 0
       # edge_count = 0
        alpha = 0
       # edges = []
        
        for region in vor.regions:
            flag = True
            inboundpoints = 0
           # upper_bound = 2 if self.BC == 'periodic' else 1
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if x<=-0.5-eps or y<=-0.5-eps or x>=1.5+eps or y>=1.5+eps:
                        flag = False
                        break
                    if x<=1+eps and y<=1+eps:
                        inboundpoints +=1
                           
            
                        
            if region != [] and flag:
               # polygon =  Polygon(vor.vertices[region]) 
               # if inboundpoints ==0 and not polygon.contains(Point(1,1)): continue
                '''
                valid region propertities
                '''
                
            #    regions.append(region)
                reordered_region = []
                

                
                for index in region:

                  #  point = tuple(vor.vertices[index])
                    x, y = round(vor.vertices[index][0]%1, 4), round(vor.vertices[index][1]%1, 4)
                    point = (x, y)
                    if point not in vert_map:
                        self.vertices[vert_count] = point
                        vert_map[point] = vert_count
                        reordered_region.append(vert_count)
                        vert_count += 1
                    else:
                        reordered_region.append(vert_map[point])
                
                if tuple(sorted(reordered_region)) not in reordered_regions:
                    reordered_regions.append(tuple(sorted(reordered_region)))

                else:
                    continue

                alpha += 1                           
              #  sorted_vert = reordered_region    
                for i in range(len(reordered_region)):

                    self.vertex2joint[reordered_region[i]].add(alpha)  
                    """
                    cur = sorted_vert[i]
                    nxt = sorted_vert[i+1] if i<len(sorted_vert)-1 else sorted_vert[0]
                    self.edges.update({edge_count:[cur, nxt]})
                    edge_count += 1
                    """                    
        
        ''' deal with quadruples '''            
          
        self.quadruples = {}

        for k, v in self.vertex2joint.copy().items():
            if len(v)>3:
                
                grains = list(v)
               # print('quadruple', k, grains)
                num_vertices = len(self.vertex2joint)
                
            
                first = grains[0]
                v.remove(first)
                self.vertex2joint[num_vertices] = v.copy()
                v.add(first)
                
                self.vertices[num_vertices] = self.vertices[k]

                
                n1 = reordered_regions[first-1]
               # print(grains)
                for test_grains in grains[1:]:

                    if len(set(n1).intersection(set(reordered_regions[test_grains-1])))==1:
                        remove_grain = test_grains

                        break

                v.remove(remove_grain)

                self.vertex2joint[k] = v.copy()
                v.remove(first)
                v = list(v)
                
                self.quadruples.update({v[0]:(k,num_vertices), v[1]:(k,num_vertices)})     
                
        for k, v in self.vertex2joint.items():
            if len(v)!=3:
                print(k, v)    
    def random_voronoi_noflux(self):

        """ 
        Output: self.vertices, self.vertex2joint
        
        """
        max_y = self.lyd/self.lxd 
        self.max_y = max_y
        cone_ratio = self.cone_ratio
        mirrored_seeds, seeds = random_lattice(dx=self.density, noise = self.noise, BC = self.BC, max_y = max_y, cone_ratio=cone_ratio)
        vor = Voronoi(mirrored_seeds)     
    
        vert_map = {}
        vert_count = 0
       # edge_count = 0
        alpha = 1
       # edges = []
        
       
        for region in vor.regions:
            flag = True
            indomain_count = 0
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if x<=-eps or y<=cone_ratio*(1-x)-eps or x>=1.+eps or y>=max_y-cone_ratio*(1-x)+eps:
                        flag = False
                        break
                    if x>eps and x<1-eps and y>eps + cone_ratio*(1-x) and y<max_y-cone_ratio*(1-x)-eps:
                        indomain_count += 1
                           
            if region != [] and flag and indomain_count>0:

                reordered_region = []
                
                for index in region:
                    x, y = vor.vertices[index][0], vor.vertices[index][1]
                    
                    if (abs(x)<eps or abs(1-x)<eps) and (abs(y- cone_ratio)<eps or abs(max_y-cone_ratio-y)<eps):
                        
                        if abs(x)<eps and abs(y - cone_ratio)<eps:
                            self.corner_grains[0] = alpha + 1
                        if abs(1-x)<eps and abs(y)<eps:
                            self.corner_grains[1] = alpha + 1
                        if abs(x)<eps and abs(max_y - cone_ratio - y)<eps:
                            self.corner_grains[2] = alpha + 1
                        if abs(1-x)<eps and abs(max_y - y)<eps:
                            self.corner_grains[3] = alpha + 1
                        
                        continue
                
                    
                    point = (x, y)
                    if point not in vert_map:
                        self.vertices[vert_count] = point
                        vert_map[point] = vert_count
                        reordered_region.append(vert_count)
                        vert_count += 1
                    else:
                        reordered_region.append(vert_map[point])
                

                alpha += 1                           
              #  sorted_vert = reordered_region    
                for i in range(len(reordered_region)):

                    self.vertex2joint[reordered_region[i]].add(alpha)  
          
          
        for k, v in self.vertex2joint.copy().items():
            if len(v)<3:
                self.vertex2joint[k].add(1)



        for k, v in self.vertex2joint.copy().items():
            if len(v)<3:
                del self.vertex2joint[k]
               # print(k, self.vertices[k], v)
                
                

        
    def plot_polygons(self, imagesize = None):
        """
        Input: region_coors
        Output: alpha_field, just index

        """
        if not imagesize:
            imagesize = self.imagesize
        s = imagesize[0]
        
        if self.BC == 'noflux':
            image = Image.new("RGB", (imagesize[0], imagesize[1])) 
        if self.BC == 'periodic':           
            image = Image.new("RGB", (2*s, 2*s))       
        draw = ImageDraw.Draw(image)
          
        # Add polygons 
        for region_id, poly in self.region_coors.items():
            
            if self.BC == 'noflux' and region_id == 1: continue
            
            Rid = region_id//(255*255)
            Gid = (region_id - Rid*255*255)//255
            Bid = region_id - Rid*255*255 - Gid*255
            orientation = tuple([Rid, Gid, Bid])
            p = []            

            #poly = np.asarray(poly*pic_size[0], dtype=int) 
            for i in range(len(poly)):
                if self.BC == 'noflux':
                    coor = np.asarray(np.round(np.array(poly[i])*s), dtype=int)
                if self.BC == 'periodic':
                    coor = np.asarray(np.array(poly[i])*s, dtype=int)
                p.append(tuple(coor))
          #  print(p)
            if len(p)>1:
                draw.polygon(p, fill=orientation) 

        img = np.array(image, dtype=int)
        img = img[:,:,0]*255*255 + img[:,:,1]*255 + img[:,:,2]


        if self.BC == 'periodic':
            img = np.stack([img[:s,:s] ,img[s:,:s] ,img[:s,s:] ,img[s:,s:] ])    
            self.alpha_field = np.max(img, axis=0)    
        

        
        if self.BC == 'noflux':
            patch = np.meshgrid(np.arange(0, imagesize[0]), np.arange(0, imagesize[1]))
            patch = 2*patch[0]//imagesize[0] + 2*(2*patch[1]//imagesize[1])
  
            self.alpha_field  = img + np.array(self.corner_grains)[patch]*(img==0)

        if self.raise_err:
            assert np.all(self.alpha_field>0), self.seed
   
        self.compute_error_layer()
    def snap(self,fname='snap'):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]], color='k')

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fname, dpi=300)

    def snap_nucle(self, fname='snap'): # function to plot the nucleation

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Iterate over regions with their IDs.
        for region_id, coors in self.region_coors.items():
            # Convert any integer entry into a coordinate pair using self.vertices.
            converted = []
            for pt in coors:
                if isinstance(pt, (int, np.integer)):
                    pt = self.vertices[pt]
                converted.append(pt)
            
            # If the region has fewer than 2 vertices, skip it.
            if len(converted) < 2:
                continue
            
            # Draw the region by connecting vertices cyclically.
            for i in range(len(converted)):
                cur = converted[i]
                nxt = converted[(i + 1) % len(converted)]
                ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]], color='k')
            
            # Determine the region's centroid.
            if region_id in self.region_center:
                cx, cy = self.region_center[region_id]
            else:
                xs, ys = zip(*converted)
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            

            ax.text(cx, cy, str(region_id), color='red', fontsize=8,
                    ha='center', va='center')


        for region_id, coors in self.added_region.items():
            # Convert any integer entry into a coordinate pair using self.vertices.
            converted = []
            for pt in coors:
                if isinstance(pt, (int, np.integer)):
                    pt = self.vertices[pt]
                converted.append(pt)
            
            # A region should have at least 3 vertices to form a polygon.
            if len(converted) < 3:
                continue
            
            # Unzip the coordinate tuples into separate lists of x and y coordinates.
            xs, ys = zip(*converted)
            
            # Option 1: Use ax.fill to fill the region with a face color.
            # You can choose a fixed facecolor or use a colormap based on region_id.
            facecolor = 'lightblue'  # or, for example: plt.cm.viridis(region_id / max_region_id)
            plt.gca().fill(xs, ys, facecolor=facecolor, edgecolor='k', alpha=0.5)



        # if hasattr(self, 'added_edge'):
        #     for edge in self.added_edge:
        #         v1, v2 = edge  # each edge is (vertex_id1, vertex_id2)
        #         pt1 = self.vertices[v1]
        #         pt2 = self.vertices[v2]
        #         print("Point", v1, v2)
        #         ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='green', linestyle='--', linewidth=1)

        if hasattr(self, 'cutted_edges'):
            for edge in self.cutted_edges:
                print("cuuted",edge)
                # Each edge is assumed to be a list like [v1, v2].
                v1, v2 = edge
                pt1 = self.vertices[v1]
                pt2 = self.vertices[v2]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color='blue', linewidth=2)
        for pt in self.intersection_points:
            ax.plot(pt.x, pt.y, 'ro', markersize=2)
        for vid, coord in self.vertices.items():
            ax.text(coord[0], coord[1], str(vid), color='purple', fontsize=5,
                    ha='center', va='center')
        # Draw reference lines and labels.
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        ax.axvline(0, color='gray', linewidth=1, linestyle='--')
        ax.text(1, 0, '1', va='center', ha='left', color='gray')
        ax.text(0, 1, '1', va='bottom', ha='center', color='gray')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fname, dpi=300)

    def check_overlap(self, edge_index_dict, timestep=None): # function to check whether we have overlapping grains 
        self.grain_neighbor.clear()

        for key, regions in self.vertex2joint.items():
        # For every vertex, regions is a set of region IDs that share that vertex.
            for region in regions:
                # Add all the other regions as neighbors for this region.
                self.grain_neighbor[region].update(set(regions) - {region})

        region_polygons = {}
        for region, coords in self.region_coors.items():
 
            sorted_coords = sort_polygon(coords)
            region_polygons[region] = Polygon(sorted_coords)
        overlaps = {}
        for region, neighbors in self.grain_neighbor.items():
            overlaps[region] = {}
            poly = region_polygons.get(region)
            if poly is None:
                continue
            for neighbor in neighbors:
                neighbor_poly = region_polygons.get(neighbor)
                if neighbor_poly is None:
                    continue
                # Use shapely's intersection method to check if there is an actual overlapping area.
                if poly.intersects(neighbor_poly):
                    overlap_region = poly.intersection(neighbor_poly)
                    overlap_area = overlap_region.area
                    # print("Overlap area:", overlap_area)
                    overlap_geom = poly.intersection(neighbor_poly)
                    if overlap_area>1e-5:
                        overlaps[region][neighbor] = True
        plot_overlap_polygons(self, region_polygons,overlaps, timestep)


    def check_planar(self, edge_index_dict):
        def plot_edges(self, edges):
            for u, v in edges:
                # Retrieve coordinates from self.vertices
                coord_u = self.vertices[u]
                coord_v = self.vertices[v]
                # Plot the edge between the two coordinates
                plt.plot([coord_u[0], coord_v[0]], [coord_u[1], coord_v[1]], 'k-')
            plt.title("Graph Plot")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig('planar_graph.png',dpi=600)

        E_pp = edge_index_dict['joint', 'connect', 'joint']
        G=nx.Graph()
        for v,coord in self.vertices.items():
            G.add_node(v, pos=coord)
        unique_edges = set()
        num_edges = E_pp.shape[1]
        for i in range(num_edges):
            src = int(E_pp[0, i].item())
            dst = int(E_pp[1, i].item())
            if(math.sqrt((self.vertices[src][0]-self.vertices[dst][0])**2+(self.vertices[dst][1]-self.vertices[src][1])**2)>0.8):
                continue
            edge = (min(src, dst), max(src, dst))
            unique_edges.add(edge)
        
        for edge in unique_edges:
            G.add_edge(*edge)

        is_planar,embedding =nx.check_planarity(G,counterexample=True)
        #plot_edges(self,counterexample.edges())
        if is_planar:
            #pos = nx.spring_layout(G)
            pos = nx.kamada_kawai_layout(G)
            nx.draw(G, pos, with_labels=False, node_size=20)
            plt.savefig('planar_graph.png',dpi=600)


        # if not is_planar:
        #     print("counterexample:",counterexample.edges())
        #     pos = nx.spring_layout(counterexample)
        #     nx.draw(counterexample, pos, with_labels=True)
        #     plt.savefig('planar_graph.png',dpi=600)

        print("Graph is planar:", is_planar)
    
    def snap_classifer(self,fname='snap_classifer', timestep=None): 
        os.makedirs("class_before", exist_ok=True)
        if timestep is not None:
            fname=f"classifier_plot_t{timestep}.png"
        else:
            fname="classifier_plot"
        full_path = os.path.join("class_before", fname)
        fig, ax = plt.subplots(figsize=(8, 6))
        color = 'blue'
        alpha = 0.5
        #print('elimgrains',self.elim_grain_to_vertices.items())
        for grain, vertex_indices in self.elim_grain_to_vertices.items():
            coords = [self.index_to_vertex_coord[v] for v in vertex_indices]
            if len(coords) < 3:
                continue
            ref=coords[0]

            adjusted_coords = [periodic_move(coord, ref) for coord in coords]
            
            # Now sort the adjusted coordinates into a proper polygon order.
            sorted_coords = sort_polygon(adjusted_coords)  # Assume sort_polygon is defined elsewhere
            
            # Create a polygon patch from the sorted coordinates
            polygon = patches.Polygon(sorted_coords, closed=True,
                                    facecolor=color, edgecolor='black', alpha=alpha)
            ax.add_patch(polygon)
        for grain, vertex_indices in self.nucle_grain_to_vertices.items():
            coords = [self.index_to_vertex_coord[v] for v in vertex_indices]
            if len(coords) < 3:
                continue
            ref=coords[0]
            adjusted_coords = [periodic_move(coord, ref) for coord in coords]
            # Now sort the adjusted coordinates into a proper polygon order.
            sorted_coords = sort_polygon(adjusted_coords)  # Assume sort_polygon is defined elsewhere
            # Create a polygon patch from the sorted coordinates
            polygon = patches.Polygon(sorted_coords, closed=True,
                                    facecolor='yellow', edgecolor='black', alpha=alpha)
            ax.add_patch(polygon)

        def scale_linewidth(prob, threshold=1e-8, base_width=1, max_width=10, scale_factor=0.25):
            if prob < threshold:
                return base_width
            # Compute the logarithmic scale factor
            lw = base_width + scale_factor * math.log10(prob / threshold)
            # Cap the linewidth at max_width
            return min(lw, max_width)
        # Create a single subplot (a single axis)
        
        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                
                if isinstance(cur, np.ndarray):
                    cur_key = tuple(cur.tolist())
                elif isinstance(cur, list):
                    cur_key = tuple(cur)
                else:
                    cur_key = cur 

                if isinstance(nxt, np.ndarray):
                    nxt_key = tuple(nxt.tolist())
                elif isinstance(nxt, list):
                    nxt_key = tuple(nxt)
                else:
                    nxt_key = nxt

                cur_index = self.vertex_coord_to_index.get(cur_key)
                nxt_index = self.vertex_coord_to_index.get(nxt_key)

                if cur_index is None or nxt_index is None:
                    prob=0
                else:
                    edge = (min(cur_index, nxt_index), max(cur_index, nxt_index))
                    prob = self.edge_prob_dict.get(edge, 0)

                lw = scale_linewidth(prob, threshold=1e-8)


                ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]], color='r', linewidth=lw)
        
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(full_path, dpi=300)

    def show_movement(self,fname='movement'):
        fig, ax = plt.subplots(figsize=(8, 6))
        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]], color='r')
        drawn = set()
        for coord, idx in self.vertex_coord_to_index.items():
            if idx in drawn:
                continue  # Skip if this vertex has already been processed
            drawn.add(idx)
            movement = self.junct_gradient_dict.get(idx)
            if (movement[0]**2 + movement[1]**2) < (0.05**2):
                continue
            # if isinstance(movement, torch.Tensor):
            #     movement = movement.detach().cpu().numpy()
            ax.arrow(coord[0], coord[1],
                    movement[0], movement[1],
                    head_width=0.02, head_length=0.03,
                    fc='blue', ec='blue')
    
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fname, dpi=300)

    def show_data_struct(self):
        

        fig, ax = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1], 'height_ratios': [1]})
        
        Q, V, E = len(self.regions), len(self.vertex_neighbor), len(self.edges)

        
        for coors in self.region_coors.values():
            for i in range(len(coors)):
                cur = coors[i]
                nxt = coors[i+1] if i<len(coors)-1 else coors[0]
                ax[0].plot([cur[0],nxt[0]], [cur[1],nxt[1]], 'k')
                
        x, y = zip(*self.region_center.values())     
    
      #  ax[0].scatter(list(x), list(y), c = 'k')
        ax[0].axis("equal")
        ax[0].axis('off')
       # ax[0].set_title('(Q, V, E)=(%d, %d, %d)'%(Q, V, E))
       # ax[0].set_title('Graph')
        ax[0].set_frame_on(False)
        
        ax[1].imshow(self.theta_z[self.alpha_field]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
      #  ax[1].set_title('GrainGNN') 
        
        ax[2].imshow(self.theta_z[self.alpha_pde]/pi*180, origin='lower', cmap=newcmp, vmin=0, vmax=90)
        ax[2].set_xticks([])
        ax[2].set_yticks([])
       # ax[2].set_title('Phase field')         
        
        ax[3].imshow(1*(self.alpha_pde!=self.alpha_field),cmap='Reds',origin='lower')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        p_err = int(np.round(self.error_layer*100))
      #  ax[3].set_title('error'+'%d'%(p_err)+'%')           

        if self.save:
            plt.savefig(self.save, dpi=400)
            
    def consolidate_and_remove_best(self):
        def polygon_area(coords):
            area = 0.0
            n = len(coords)
            for i in range(n):
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        for orig_id, added_ids in self.region2newidx.items():
            best_id, best_area = None, -1.0

            # 1) Find the added region with maximum area
            for rid in added_ids:
                coors = self.region_coors.get(rid)
                if not coors or len(coors) < 3:
                    continue
                pts = [(self.vertices[v] if isinstance(v, int) else v) for v in coors]
                area = polygon_area(pts)
                if area > best_area:
                    best_area, best_id = area, rid

            if best_id is None:
                continue  # no valid fragment

            # 2) Copy best_id’s geometry into orig_id
            self.region_coors[orig_id]   = self.region_coors[best_id]
            self.regions[orig_id]        = self.regions[best_id]
            self.region_center[orig_id]  = self.region_center[best_id]

            # 3) Update vertex2joint: any vertex of best_id now points to orig_id
            for vid in self.regions[orig_id]:
                joints = self.vertex2joint.get(vid)
                if joints and best_id in joints:
                    joints.discard(best_id)
                    joints.add(orig_id)

            # 4) Delete best_id from all data structures
            self.replaced.append(best_id)
            del self.region_coors[best_id]
            del self.regions[best_id]
            del self.region_center[best_id]
            # Remove best_id from vertex2joint globally
            for vid, joints in list(self.vertex2joint.items()):
                if best_id in joints:
                    joints.discard(best_id)
                    # If a vertex now has no regions, you may choose to delete it:
                    if not joints:
                        del self.vertex2joint[vid]
                        
    def add_polygon(self, center=(0.5, 0.5), radius=0.1, perturb_ratio=0.05):

        """
        Create a polygon region by adding either a pentagon (5 vertices) or a quadrilateral (4 vertices)
        with equal probability.
        
        For each new vertex, the function:
        - Computes its coordinate (based on the given center, radius, and a small random perturbation),
        - Adds the vertex to self.vertices and initializes its vertex2joint set,
        - Creates an edge between consecutive vertices,
        - Updates the new region (stored in self.regions, self.region_coors, and self.region_center),
        - And then updates the vertex2joint mapping.
        
        For joint2vertex, since each new vertex is only adjacent to the new region,
        a unique key is formed for each vertex by using a tuple (new_region, vertex_id).
        """
        print("Check number of vertices", len(self.vertex2joint))
        self.nucle_fall_region.clear() 
        self.nucle_map.clear()
        # shape_type = random.choice(["pentagon", "quadrilateral"])
        shape_type="pentagon"
        if shape_type == "pentagon":
            n_vertices = 5
        else:
            n_vertices = 4

        print(f"Adding a {shape_type} region.")    # decide the shape

        polygon_points = []
        for i in range(n_vertices):    # add the polygon
            angle = 2 * math.pi * i / n_vertices - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            # Add small random perturbation:
            perturb = perturb_ratio * radius 
            x += np.random.uniform(-perturb, perturb)
            y += np.random.uniform(-perturb, perturb)
            polygon_points.append((x, y))     #in polygon_points, stores the corrdinates for each points
        
        # Assign new vertex IDs for the polygon vertices.
        next_vertex_id = max(self.vertices.keys()) + 1 if self.vertices else 0
        new_vertex_ids = []
        for pt in polygon_points:
            self.vertices[next_vertex_id] = pt       # store the new element in map (id -> coordinates)
            self.vertex2joint[next_vertex_id] = set()  # create the map (vertex id -> adjacent grain id)
            new_vertex_ids.append(next_vertex_id)  # put it into the set
            next_vertex_id += 1
        
        for i in range(n_vertices):
            v1 = new_vertex_ids[i]
            v2 = new_vertex_ids[(i + 1) % n_vertices]
            self.edges.append([v1, v2]) # store all the internal edges of this polygon

        
        new_region = max(self.region_coors.keys()) + 1 if self.region_coors else 0  # create new added_region idx
        self.regions[new_region] = new_vertex_ids # store the map (region idx -> set of vertex idxs)
        print("new idx", new_region,  new_vertex_ids)
        self.region_coors[new_region] = polygon_points  # store the map (region idx -> set of new vertex's coordinates)
        self.added_region[new_region] = polygon_points   # same
        xs, ys = zip(*polygon_points)
        self.region_center[new_region] = [np.mean(xs), np.mean(ys)] 
        
        for v in new_vertex_ids:
            self.vertex2joint[v].add(new_region) # store the map ( vertex -> {new region})
        
        # for v in new_vertex_ids:
        #     # self.vertex2joint[v] now contains the set of regions (e.g. {new_region})
        #     self.joint2vertex[tuple(sorted(self.vertex2joint[v]))] = v
        for v in new_vertex_ids:
            self.joint2vertex[(new_region, v)] = v # store the reverse map ( {new region, v} -> vertex)  incomplete!

        # # Save the region id with an attribute depending on the shape.
        # if shape_type == "pentagon":
        #     self.pentagon_region_id[new_vertex_ids] = new_region
        # else:
        #     self.quadrilateral_region_id = new_region


        # Optionally, find which existing Voronoi vertices are covered by this new polygon.
        covered = self.find_vertices_covered_by_polygon(polygon_points,new_region) # search for all vertexs that covered by new polygon
        if covered:
            print("These existing Voronoi vertices are covered by the new", shape_type, ":", covered)
            for cv in covered:
                regions_for_cv = self.vertex2joint[cv]
                print(f"  Vertex {cv} is in region(s) {regions_for_cv}")
        else:
            print("No existing Voronoi vertices were covered by the new", shape_type, ".")
        self.covered=covered
        
        # For each new vertex, determine which region(s) it falls into.
        for v_id in new_vertex_ids:
            vx, vy = self.vertices[v_id]
            found_regions = self.find_regions_for_point((vx, vy), new_region) # find which region idx where the vertex fall into
            # self.joint2vertex[(new_region, found_regions)] = v_id
            self.nucle_fall_region.append(found_regions)  # store the idx of this region
            self.nucle_map[found_regions].append(v_id)  # create the map (region -> vertex)
            print(f"{shape_type.capitalize()} vertex {v_id} at ({vx:.3f}, {vy:.3f}) lies in region(s): {found_regions}")
        # print(self.nucle_fall_region)
        # print(self.nucle_map)



        self.intersection_points+=self.find_polygon_edge_intersections(polygon_points)

        self.connect()


        self.reconstruct(new_vertex_ids, new_region)
        # print("MAP",self.region2newidx)

        self.vertex2joint = {k: v for k, v in self.vertex2joint.items() if k not in self.covered}

        print("Check number of vertices", len(self.vertex2joint))
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())

        # print(self.vertex2joint)
        # self.nucle_fall_region=[]
        # self.intersection_points=[]
        # self.covered=[]
        # self.cutted_edges=[]
        # self.nucle_map =defaultdict(list)
        # self.added_edge=[]
        # self.projected_vertices=[]
        # self.inter_point=[]

        return new_region, new_vertex_ids       


    def sort_candidates_clockwise(self, candidates):

        coords = [self.vertices[v] for v in candidates]
        
        # Compute the centroid.
        cx = sum(x for x, y in coords) / len(coords)
        cy = sum(y for x, y in coords) / len(coords)
        
        # Define a function to compute the angle from the centroid to the vertex.
        def angle_from_centroid(v):
            x, y = self.vertices[v]
            # atan2 returns angle in [-pi, pi]; for clockwise sort, we use reverse order.
            return math.atan2(y - cy, x - cx)
        
        # Sort candidates in descending order of angle to get clockwise order.
        sorted_candidates = sorted(candidates, key=angle_from_centroid, reverse=True)
        return sorted_candidates

    def check_relative(self, v_id, v_vertex, connected_vertex):
        A=self.vertices[v_id]
        B=self.vertices[v_vertex]
        C=self.vertices[connected_vertex]
        Ax, Ay = A[0] - C[0], A[1] - C[1]
        Bx, By = B[0] - C[0], B[1] - C[1]
        cross = Ax * By - Ay * Bx
        if cross<0:
            return True
        elif cross>0:
            return False
        else:
            return None

    def reconstruct(self, new_vertex_ids, new_region):
        region_count=max(self.region_coors.keys()) if self.region_coors else 0
        print(self.covered)

        edge_map = {}
        for region in set(self.nucle_fall_region):
            store_idx=region
            candidates = list((set(self.regions[region]).union(set(self.projected_vertices))) - set(self.covered))
            print("Cand",candidates)
            sorted_candidates = self.sort_candidates_clockwise(candidates)
            for v_id in self.nucle_map[region]: 
                print("CHECK ID", v_id)

                # if v_id in vertex_map and vertex_map[v_id][0] > -1 and vertex_map[v_id][1] > -1:
                #     continue

                for edge in self.added_edge:
                    if v_id in edge:
                        connected_vertex = edge[0] if edge[1] == v_id else edge[1]
                        current_edge =edge
                        break  

                v_vertex=None
                for edge in self.cutted_edges:
                    if connected_vertex in edge:
                        v_vertex=edge[0] if edge[1]==connected_vertex else edge[1]
                        cut=edge
                        break
                
                if v_vertex is not None:
                    left=self.check_relative(v_id, v_vertex, connected_vertex )
                    if left is True:

                        if current_edge in edge_map:
                            edge_map[current_edge] = [region_count + 1, edge_map[current_edge][1]]
                            # print('Check',edge_map[current_edge])
                            region_count+=1
                        else:
                            edge_map[current_edge] = [region_count + 1, region_count + 2]
                            region_count+=2
                            region_count=self.process_right(edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex, v_id)

                        if tuple(cut) not in edge_map:
                            edge_map[tuple(cut)] = [-1, -1]  
                        edge_map[tuple(cut)][1] = edge_map[current_edge][0]

                    
                    elif left is False:

                        if current_edge in edge_map:
                            edge_map[current_edge]=[edge_map[current_edge][0],region_count + 1]
                            # print('Check',edge_map[current_edge])
                            region_count+=1
                        else:
                            edge_map[current_edge] = [region_count + 2, region_count + 1]
                            region_count+=2
                            region_count=self.process_left(edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex, v_id)

                        if tuple(cut) not in edge_map:
                            edge_map[tuple(cut)] = [-1, -1]  
                        edge_map[tuple(cut)][0] = edge_map[current_edge][1]

                else:
                    if current_edge in edge_map and edge_map[current_edge][0]>-1 and edge_map[current_edge][1]>-1:
                        self.vertex2joint[connected_vertex ].discard(region)
                        self.vertex2joint[connected_vertex ].update(edge_map[current_edge][:2])
                        inner = current_edge[1] if current_edge[0] == connected_vertex else current_edge[0]
                        self.vertex2joint[inner].clear()
                        self.vertex2joint[inner].update(edge_map[current_edge][:2] + [new_region])
                        print(current_edge)
                        # print("CHECK ID2", v_id)

                        # print('Map',edge_map)
                        self.update_connected( edge_map, current_edge, connected_vertex, new_region,region)
                        continue

                    if current_edge not in edge_map:
                        edge_map[current_edge]=[-1,-1]

                    if edge_map[current_edge][0] == -1:
                        region_count=self.process_left(edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex,  v_id)

                    if edge_map[current_edge][1] == -1:
                        region_count=self.process_right(edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex, v_id)

                # print('Map',edge_map)
                self.update_connected( edge_map, current_edge, connected_vertex, new_region,region)

        self.update_intersect(edge_map, current_edge, new_region)


    def update_intersect(self, edge_map, current_edge, new_region):
        print(edge_map)
        for intersec in self.inter_point:
            current_edge = next((edge for edge in self.cutted_edges if intersec in edge), None)
            if intersec not in self.vertex2joint:
                self.vertex2joint[intersec] = set()
            # print("SHOWWWWW",self.vertex2joint[intersec])
            self.vertex2joint[intersec].clear()
            self.vertex2joint[intersec].update(edge_map[tuple(current_edge)][:2] + [new_region])
            print("IN", intersec, self.vertex2joint[intersec])

            outer=current_edge[1] if current_edge[0]==intersec else current_edge[0]
            self.vertex2joint[outer].update(edge_map[tuple(current_edge)][:2])
            print("OUT", outer, self.vertex2joint[outer])


    def update_connected(self, edge_map, current_edge, connected_vertex, new_region, region):
        print(connected_vertex,self.vertex2joint[connected_vertex])
        self.vertex2joint[connected_vertex].discard(region)
        self.vertex2joint[connected_vertex].update(edge_map[current_edge][:2])
        self.region2newidx[region].update(edge_map[current_edge][:2])
        print(edge_map[current_edge])

        print("Outter", connected_vertex, self.vertex2joint[connected_vertex])

        inner = current_edge[1] if current_edge[0] == connected_vertex else current_edge[0]
        self.vertex2joint[inner].clear()
        self.vertex2joint[inner].update(edge_map[current_edge][:2] + [new_region])
        print("Inner", inner,self.vertex2joint[inner])


    def process_left(self, edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex, v_id ):
        edge_map[current_edge] = [region_count + 1, edge_map[current_edge][1]]
        region_count += 1

        idx = sorted_candidates.index(connected_vertex)
        n=len(sorted_candidates)

        left_candidate=sorted_candidates[(idx-1)%n]
        while (all(left_candidate not in edge for edge in self.cutted_edges) and 
        all(left_candidate not in edge for edge in self.added_edge)):
            print("MAP", self.vertex2joint)
            self.vertex2joint[left_candidate].discard(region)
            self.vertex2joint[left_candidate].add(edge_map[current_edge][0])

            idx = sorted_candidates.index(left_candidate)

            left_candidate=sorted_candidates[(idx-1)%n]

        if any(left_candidate in edge for edge in self.added_edge) and any(left_candidate in edge for edge in self.cutted_edges):
            connected_edge = next((edge for edge in self.added_edge if left_candidate in edge), None)
            cut_edge = next((edge for edge in self.cutted_edges if left_candidate in edge), None)

            con_v=connected_edge[0] if connected_edge[1]==left_candidate else connected_edge[1]
            cut_v=cut_edge[0] if cut_edge[1]==left_candidate else cut_edge[1]

            left=self.check_relative(con_v, cut_v, left_candidate)

            if left is True:
                connected_edge = next((edge for edge in self.added_edge if left_candidate in edge), None)
                if connected_edge not in edge_map:
                    edge_map[connected_edge] = [-1, -1]
                edge_map[connected_edge][1]=edge_map[current_edge][0]
            elif left is False:
                # cut_edge=self.find_correct_cut_edge(left_candidate, v_id)
                cut_edge = next((edge for edge in self.cutted_edges if left_candidate in edge), None)
                self.vertex2joint[left_candidate].discard(region)
                if tuple(cut_edge) not in edge_map:
                    edge_map[tuple(cut_edge)] = [-1, -1]  
                edge_map[tuple(cut_edge)][1] = edge_map[current_edge][0]

            return region_count




        if any(left_candidate in edge for edge in self.added_edge):
            connected_edge = next((edge for edge in self.added_edge if left_candidate in edge), None)
            if connected_edge not in edge_map:
                edge_map[connected_edge] = [-1, -1]
            edge_map[connected_edge][1]=edge_map[current_edge][0]
            
        if any(left_candidate in edge for edge in self.cutted_edges):
            # cut_edge=self.find_correct_cut_edge(left_candidate, v_id)
            cut_edge = next((edge for edge in self.cutted_edges if left_candidate in edge), None)
            self.vertex2joint[left_candidate].discard(region)
            if tuple(cut_edge) not in edge_map:
                edge_map[tuple(cut_edge)] = [-1, -1]  
            edge_map[tuple(cut_edge)][1] = edge_map[current_edge][0]
        return region_count


    def find_correct_cut_edge(self, candidate, v_id):

        best_edge = None
        best_distance = float('inf')
        v_coord = self.vertices[v_id]
        
        for edge in self.cutted_edges:
            if candidate in edge:
                # Find the connected vertex (the one not equal to left_candidate)
                other_vertex = edge[0] if edge[1] == candidate else edge[1]
                other_coord = self.vertices[other_vertex]
                # Compute Euclidean distance between v_coord and other_coord.
                distance = math.hypot(v_coord[0] - other_coord[0], v_coord[1] - other_coord[1])
                if distance < best_distance:
                    best_distance = distance
                    best_edge = edge
        print("EDGE",best_edge)
        return best_edge
    
    def process_right(self, edge_map, current_edge, region_count, region, sorted_candidates, connected_vertex, v_id ):
        idx = sorted_candidates.index(connected_vertex)
        n=len(sorted_candidates)

        edge_map[current_edge] = [edge_map[current_edge][0], region_count + 1]
        region_count += 1

        right_candidate=sorted_candidates[(idx+1)%n]
        while (all(right_candidate not in edge for edge in self.cutted_edges) and 
        all(right_candidate not in edge for edge in self.added_edge)):
            self.vertex2joint[right_candidate].discard(region)
            self.vertex2joint[right_candidate].add(edge_map[current_edge][1])

            idx = sorted_candidates.index(right_candidate)

            right_candidate=sorted_candidates[(idx+1)%n]

        if any(right_candidate in edge for edge in self.added_edge) and any(right_candidate in edge for edge in self.cutted_edges):
            connected_edge = next((edge for edge in self.added_edge if right_candidate in edge), None)
            cut_edge = next((edge for edge in self.cutted_edges if right_candidate in edge), None)

            con_v=connected_edge[0] if connected_edge[1]==right_candidate else connected_edge[1]
            cut_v=cut_edge[0] if cut_edge[1]==right_candidate else cut_edge[1]

            left=self.check_relative(con_v, cut_v, right_candidate)

            if left is False:
                connected_edge = next((edge for edge in self.added_edge if right_candidate in edge), None)
                if connected_edge not in edge_map:
                    edge_map[connected_edge] = [-1, -1]
                edge_map[connected_edge][0]=edge_map[current_edge][1]
            elif left is True:
                # cut_edge=self.find_correct_cut_edge(left_candidate, v_id)
                cut_edge = next((edge for edge in self.cutted_edges if right_candidate in edge), None)
                self.vertex2joint[right_candidate].discard(region)
                if tuple(cut_edge) not in edge_map:
                    edge_map[tuple(cut_edge)] = [-1, -1]  
                edge_map[tuple(cut_edge)][0] = edge_map[current_edge][1]
                
            return region_count

        if any(right_candidate in edge for edge in self.added_edge):
            connected_edge = next((edge for edge in self.added_edge if right_candidate in edge), None)
            if connected_edge not in edge_map:
                edge_map[connected_edge] = [-1, -1]
            edge_map[connected_edge][0]=edge_map[current_edge][1]

        if any(right_candidate in edge for edge in self.cutted_edges):
            # cut_edge=self.find_correct_cut_edge(right_candidate, v_id)
            cut_edge = next((edge for edge in self.cutted_edges if right_candidate in edge), None)
            self.vertex2joint[right_candidate].discard(region)
            if tuple(cut_edge) not in edge_map:
                edge_map[tuple(cut_edge)] = [-1, -1]
            edge_map[tuple(cut_edge)][0]=edge_map[current_edge][1]
        return region_count


    def find_regions_for_point(self, point, new_region, tol=1e-6):
        x, y = point
        inside_regions=0
        for region_id, coords in self.region_coors.items():

            num_vertices=len(coords)
            inside=False
            if num_vertices < 3 or region_id == new_region:
                # Skip degenerate polygons
                continue
            
            x1, y1 =coords[0]
            for i in range(1, num_vertices+1):
                x2, y2 =coords[i % num_vertices]
                if y>min(y1,y2):
                    if y<=max(y1,y2):
                        if x<=max(x1,x2):
                            x_intersection=(y-y1)*(x2-x1)/(y2-y1)+x1
                            if x1==x2 or x<=x_intersection:
                                inside =not inside
                x1, y1 = x2, y2
            
            if inside==True:
                inside_regions=region_id
                return inside_regions
        
        return inside_regions



    def find_vertices_covered_by_polygon(self, coords, new_region, tol=1e-16):
        """
        Given a polygon (list of (x,y) points) in polygon_coords,
        returns a list of vertex IDs from self.vertices that lie inside
        (or on the boundary of) that polygon.
        
        - Uses matplotlib.path.Path for the point-in-polygon test.
        - 'tol' helps if a point is exactly on the boundary.
        """
        # Build a Path object from the polygon’s coordinates
        covered_vertices = []
        for v_id ,(x, y) in self.vertices.items():
            num_vertices=len(coords)
            inside=False

            x1, y1 =coords[0]
            for i in range(1, num_vertices+1):
                x2, y2 =coords[i % num_vertices]
                if y>min(y1,y2):
                    if y<=max(y1,y2):
                        if x<=max(x1,x2):
                            x_intersection=(y-y1)*(x2-x1)/(y2-y1)+x1
                            if x1==x2 or x<=x_intersection:
                                inside =not inside
                x1, y1 = x2, y2

            if inside==True and v_id not in self.regions[new_region]:
                covered_vertices.append(v_id)

        return covered_vertices

        # polygon_path = Path(polygon_coords, closed=True)
        
        # covered_vertices = []
        # for v_id, (vx, vy) in self.vertices.items():
        #     # 'radius=-tol' treats points within 'tol' of the boundary as inside
        #     if polygon_path.contains_point((vx, vy), radius=-tol):
        #         covered_vertices.append(v_id)
        
        # return covered_vertices

    def find_polygon_edge_intersections(self, polygon_points):
        """
        Given a list of vertices (polygon_points) for the new sampled polygon,
        find and return a list of intersection points between its edges and
        the existing edges (stored in self.edges and self.vertices).
        
        Parameters:
        - polygon_points: a list of (x,y) tuples representing the new polygon's vertices.
        
        Returns:
        A list of Shapely Point objects representing intersections.
        """
        intersections = []
        scanned=[]
        n = len(polygon_points)

        # Build the polygon's edges as LineString objects.
        poly_edges = [LineString([polygon_points[i], polygon_points[(i + 1) % n]])
                    for i in range(n)]
        
        intersection_vertex_map = {}
        for rgs in self.nucle_fall_region:
            print(rgs)
            for edge in self.regions_edges[rgs]:
                scanned.append(edge)
        flat_scanned = [edge for sublist in scanned for edge in sublist] 


        def get_new_vertex_id(self):
            return max(self.vertices.keys()) + 1 if self.vertices else 0


        def choose_cutted_edge(self, v1, v2, Double_inter):
            v1x, v1y =self.vertices[v1] 
            cand1_x, cand1_y=Double_inter[0].x, Double_inter[0].y
            cand2_x, cand2_y=Double_inter[1].x, Double_inter[1].y

            dist1=(v1x-cand1_x)**2+(v1y-cand1_y)**2
            dist2=(v1x-cand2_x)**2+(v1y-cand2_y)**2

            idex1=intersection_vertex_map[(cand1_x, cand1_y)]
            idex2=intersection_vertex_map[(cand2_x, cand2_y)]
            if(dist1<dist2):
                self.cutted_edges.append([v1, idex1])
                self.cutted_edges.append([v2, idex2])
            else:
                self.cutted_edges.append([v1, idex2])
                self.cutted_edges.append([v2, idex1])


        for edge in flat_scanned:

            Multi_point=False

            v1, v2=edge
            if v1 not in self.covered and v2 not in self.covered:
                Multi_point=True
            elif v1 not in self.covered:
                surviving = v1
            elif v2 not in self.covered:
                surviving = v2
            else:
                # If both endpoints are covered, skip this edge.
                continue
            existing_line = LineString([self.vertices[v1], self.vertices[v2]])
            
            if Multi_point==True:

                Double_inter=[]
                for poly_edge in poly_edges:
                    inter=poly_edge.intersection(existing_line)
                    if not inter.is_empty:
                        if inter.geom_type == "Point" and (inter.x, inter.y) not in intersection_vertex_map:
                            intersections.append(inter) # store the intersection point
                            Double_inter.append(inter)
                            new_vid = get_new_vertex_id(self)
                            self.inter_point.append(new_vid )  # store the intersection point idx globally
                            intersection_vertex_map[(inter.x, inter.y)] = new_vid  # create map ( coordinate -> intersection point's idx)
                            self.vertices[new_vid] = (inter.x, inter.y) # store the intersection point into vertex dic
                if len(Double_inter)==2: 
                    choose_cutted_edge(self, v1, v2, Double_inter) # store two cutted edges
            else:
            # Check each edge of the new polygon for intersection.
                for poly_edge in poly_edges:
                    inter = poly_edge.intersection(existing_line)
                    if not inter.is_empty:
                        # If the intersection is a point (or multipoint), add them.
                        if inter.geom_type == "Point" and (inter.x, inter.y) not in intersection_vertex_map:
                            intersections.append(inter)
                            new_vid = get_new_vertex_id(self)
                            self.inter_point.append(new_vid )
                            intersection_vertex_map[(inter.x, inter.y)] = new_vid
                            self.vertices[new_vid] = (inter.x, inter.y)
                            self.cutted_edges.append([surviving, new_vid]) # store the one cutted edges

                    # elif inter.geom_type == "MultiPoint":
                    #     intersections.extend(list(inter.geoms))
                    # You might also get LineString if edges overlap exactly.
                    # In that case, you could choose to sample a point along the overlap.
        return intersections



    def connect(self):
        import copy
        def sort_indices_clockwise(self, indices):
            coords = [self.vertices[idx] for idx in indices]

            cx = sum(x for x, y in coords) / len(coords)
            cy = sum(y for x, y in coords) / len(coords)

            def angle(idx):
                x, y = self.vertices[idx]
                return math.atan2(y - cy, x - cx)

            return sorted(indices, key=angle, reverse=True)

        print(self.nucle_map)
        for region, new_vertices in self.nucle_map.items(): # pick the new vertex and its fall-in region

            print(region, "REGION")
            cand_vertex = [vid for vid, joint in self.vertex2joint.items() if region in joint] # find all possible vertice of this region
            filtered_vertex = [v for v in cand_vertex if v not in self.covered] # remove those being covered
            sorted_vertices=sort_indices_clockwise(self,filtered_vertex) # sort the remaining in clockwise
            cp_sorted=copy.copy(sorted_vertices)

            CW=sort_indices_clockwise(self, new_vertices) # sort the candidate vertices in clockwise
            CW_edge=[]
            CCW=list(reversed(new_vertices)) 
            CCW_edge=[]
            #do a two-direction scan, first is sort the vertices in clockwise order and second is counterclockwise.

            #for each scan, we count the sum of dist (the way to choose_hex_connetion_for_pent_vertex remains), we take the permutation who has the lowest dist sum and append these new edges.  
            sum1=0
            sum2=0
            project_list1=[]
            dict1=defaultdict(list)
            project_list2=[]
            dict2=defaultdict(list)

            index=max(self.vertices.keys()) + 1
            for idx in CW:
                polygon_ids=self.vertex2joint[idx]
                chose_idx, dist= self.choose_hex_connection_for_pent_vertex( idx, polygon_ids, sorted_vertices)
                if dist is None:
                    dist = 0
                sum1 += dist
                if chose_idx==None:
                    location=self.projection(idx, cp_sorted)
                    dict1[index]=location
                    project_list1.append(index)
                    chose_idx=index
                    index+=1
                    # self.vertices[new_vid] = best_projection
                    # self.projected_vertices.append(new_vid)
                else:
                    sorted_vertices.remove(chose_idx)
                # print(idx, "choose" , chose_idx)
                CW_edge.append((idx, chose_idx))
            index=max(self.vertices.keys()) + 1
            sorted_vertices=sort_indices_clockwise(self,filtered_vertex)
            for idx in CCW:
                polygon_ids=self.vertex2joint[idx]
                chose_idx, dist= self.choose_hex_connection_for_pent_vertex( idx, polygon_ids, sorted_vertices) # return the candidates in hexagon, and its distance
                if dist is None:
                    dist = 0
                sum2+=dist
                if chose_idx==None:
                    location=self.projection(idx, cp_sorted)
                    dict2[index]=location
                    project_list2.append(index)
                    chose_idx=index
                    index+=1
                    
                else:
                    sorted_vertices.remove(chose_idx) 
                # print(idx, "choose" , chose_idx)
                CCW_edge.append((idx, chose_idx))

            if sum1<sum2:
                self.added_edge.extend(CW_edge) 
                self.edges.extend(CW_edge)
                self.vertices.update(dict1)
                self.projected_vertices=project_list1
                print("CW")
            else:
                self.added_edge.extend(CCW_edge)
                self.edges.extend(CCW_edge)
                self.vertices.update(dict2)
                self.projected_vertices=project_list2


                

    def projection(self, pent_vertex_id, hex_candidate_ids):
        """
        Compute the projection of the pentagon vertex P onto an edge formed by consecutive vertices
        from hex_candidate_ids. Only candidate pairs that actually form an edge in self.edges (either 
        as [v1, v2] or [v2, v1]) are considered.

        Parameters:
        pent_vertex_id: the ID of the pentagon vertex P.
        hex_candidate_ids: a list of vertex IDs (ordered) that are candidates from the hexagon.
        
        Returns:
        best_projection: (x, y) coordinates of the projection point on the best edge.
        best_edge: a tuple (v1, v2) representing the edge used.
        best_distance: the minimal distance from P to its projection.
        """

        # Get coordinate for the pentagon vertex P.
        P = np.array(self.vertices[pent_vertex_id])
        
        best_distance = float('inf')
        best_projection = None
        best_edge = None

        # Loop through consecutive candidate pairs.
        # (Assume the candidate list is ordered; if not, you might need additional sorting.)
        n = len(hex_candidate_ids)
        print("number",n)
        for i in range(n):
            v1 = hex_candidate_ids[i]
            v2 = hex_candidate_ids[(i + 1) % n]
            
            # Check if the candidate edge actually exists in self.edges.
            if ([v1, v2] not in self.edges) and ([v2, v1] not in self.edges):
                continue  # Skip this pair if the edge is not present.
            if pent_vertex_id==113:
                print("Check",v1,v2)
            # Get the coordinates for endpoints A and B.
            A = np.array(self.vertices[v1])
            B = np.array(self.vertices[v2])
            
            # Compute the projection of P onto line AB.
            AB = B - A
            AB_norm_sq = np.dot(AB, AB)
            if AB_norm_sq == 0:
                # Degenerate edge: treat A as the projection.
                proj = A
            else:
                # Parameter t for projection, then clamp to [0, 1] to remain on the segment.
                t = np.dot(P - A, AB) / AB_norm_sq
                t = max(0, min(1, t))
                proj = A + t * AB
            
            # Compute the distance from P to the projection.
            dist = np.linalg.norm(P - proj)
            print(v1,v2, dist)
            if dist < best_distance:
                best_distance = dist
                best_projection = proj
                best_edge = (v1, v2)
        
        # Convert best_projection from numpy array to tuple (if found).
        if best_projection is not None:
            # new_vid = max(self.vertices.keys()) + 1 
            best_projection = tuple(best_projection)
            # self.vertices[new_vid] = best_projection
            # self.projected_vertices.append(new_vid)
        return best_projection



    def choose_hex_connection_for_pent_vertex(self, pent_vertex_id, polygon_ids, hex_candidate_ids, tol=1e-6):
        """
        For a given pentagon vertex (pent_vertex_id), choose a candidate hexagon vertex from
        hex_candidate_ids such that when the new edge (P -> H) is added, the three edges
        meeting at P (the two adjacent pentagon edges and the new edge) all form angles
        less than 180°. Among those candidates, the one with smallest distance is chosen.
        
        Assumes:
        - self.vertices: dict mapping vertex_id -> (x, y)
        - self.regions[self.pentagon_region_id]: ordered list of vertex IDs for the pentagon.
        
        Returns (chosen_hex_vertex_id, distance) or (None, None) if no candidate qualifies.
        """
        if isinstance(hex_candidate_ids, int):
            hex_candidate_ids = [hex_candidate_ids]
        # Get P coordinate.
        P = self.vertices[pent_vertex_id] # P is the coordinate of the vertex of pentagon to be connected 
        
        # Retrieve the ordered pentagon vertex list. (Assume we have stored the pentagon's region ID.)
        pentagon_ids = self.regions[next(iter(polygon_ids))] # find all vertex of this pentagon

        try:
            idx = pentagon_ids.index(pent_vertex_id)
        except ValueError:
            print(f"Pentagon vertex {pent_vertex_id} not found in pentagon region!")
            return (None, None)
        
        # Get the adjacent pentagon vertices (cyclically).
        P_prev = self.vertices[pentagon_ids[idx - 1]]   
        P_next = self.vertices[pentagon_ids[(idx + 1) % len(pentagon_ids)]]
        
        # Initialize candidate list.
        valid_candidates = []
        
        # Loop through the hexagon candidate vertex IDs.
        for h_id in hex_candidate_ids:
            H = self.vertices[h_id]
            # Check if the new configuration at P is convex:
            if all_angles_convex(P, P_prev, P_next, H, tol=tol): # verify it is convex
                # Compute Euclidean distance from P to H.
                distance =np.linalg.norm(np.array(P) - np.array(H))
                valid_candidates.append((h_id, distance))
        print("Valid",valid_candidates)
        if valid_candidates:
            # Choose the candidate with the smallest distance.
            valid_candidates.sort(key=lambda x: x[1])
            chosen_id, min_distance = valid_candidates[0]  # find the smallest distance
            return chosen_id, min_distance
        else:
            return (None, None)

    def connect_pentagon_vertices(self, pentagon_vertex_ids, hex_region_mapping, tol=1e-6):
        """
        For each pentagon vertex in pentagon_vertex_ids, connect it to one candidate hexagon vertex.
        
        Parameters:
        - pentagon_vertex_ids: list of pentagon vertex IDs.
        - hex_region_mapping: dict mapping pentagon vertex ID -> list of candidate hexagon vertex IDs
            (obtained by your earlier point-in-polygon tests).
        
        For each pentagon vertex, choose the hexagon candidate that, when connected, 
        results in all three incident angles (with the two adjacent pentagon edges) being < 180°.
        The connection with the smallest distance among valid ones is chosen.
        The new connection edge is added to self.edges.
        """
        new_edges = []
        for p_id in pentagon_vertex_ids:
            candidates = hex_region_mapping.get(p_id, [])
            chosen, dist_val = self.choose_hex_connection_for_pent_vertex(p_id, candidates, tol=tol)
            if chosen is not None:
                new_edges.append((p_id, chosen))
                self.edges.append([p_id, chosen])
                print(f"Pentagon vertex {p_id} connected to hexagon vertex {chosen} (distance {dist_val:.4f})")
            else:
                print(f"No valid hexagon candidate found for pentagon vertex {p_id}")
        return new_edges
    def construct_reordered_regions_from_joint2vertex(self):
        """
        从joint2vertex重构reordered_regions
        """
        # 找出最大的晶粒ID
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())
        max_grain_id = 0
        for joint in self.joint2vertex.keys():
            max_grain_id = max(max_grain_id, max(joint))
        
        # 初始化reordered_regions
        reordered_regions = [set() for _ in range(max_grain_id)]
        
        # 遍历所有的joint，将顶点分配给对应的晶粒
        for joint, vertex_id in self.joint2vertex.items():
            for grain_id in joint:
                reordered_regions[grain_id - 1].add(vertex_id)
        
        # 转换为排序的元组格式
        reordered_regions = [tuple(sorted(grain_vertices)) 
                            for grain_vertices in reordered_regions]
        
        return reordered_regions

    def deal_with_quadruple(self):
        """
        使用重构的reordered_regions处理四重点
        """
        # 重构reordered_regions
        reordered_regions = self.construct_reordered_regions_from_joint2vertex()
        
        # 然后使用与原始函数几乎相同的逻辑
        self.quadruples = {}

        for k, v in self.vertex2joint.copy().items():
            if len(v)>3:
                
                grains = list(v)
               # print('quadruple', k, grains)
                num_vertices = max(self.vertices.keys()) + 1 if self.vertices else 0
            
                first = grains[0]
                v.remove(first)
                self.vertex2joint[num_vertices] = v.copy()
                v.add(first)
                
                self.vertices[num_vertices] = self.vertices[k]

                
                n1 = reordered_regions[first-1]
                print("n1",n1, first)
                print(grains)
                for test_grains in grains[1:]:
                    print("test",test_grains)
                    print("check",reordered_regions[test_grains-1])
                    print("check2",reordered_regions[122])
                    if len(set(n1).intersection(set(reordered_regions[test_grains-1])))==1:
                        remove_grain = test_grains

                        break
                print("grains set:", v, "remove grain", remove_grain)
                v.remove(remove_grain)

                self.vertex2joint[k] = v.copy()
                v.remove(first)
                v = list(v)
                
                self.quadruples.update({v[0]:(k,num_vertices), v[1]:(k,num_vertices)})     
                
        for k, v in self.vertex2joint.items():
            if len(v)!=3:
                print("error",k, v)  

        print("WHY", self.vertex2joint[211]) 


    def find_grains_containing_vertex(self, vertex_id):
        """查找包含指定顶点的所有晶粒"""
        
        grains_list = []
        
        # 方法1：从vertex2joint直接查找
        if vertex_id in self.vertex2joint:
            grains_list = list(self.vertex2joint[vertex_id])
            print(f"Method 1 - vertex2joint[{vertex_id}]: {grains_list}")
        else:
            print(f"Vertex {vertex_id} not found in vertex2joint")
        
        # 方法2：从joint2vertex反向查找（验证）
        grains_from_joint2vertex = []
        for joint, vertex in self.joint2vertex.items():
            if vertex == vertex_id:
                grains_from_joint2vertex.extend(list(joint))
        
        if grains_from_joint2vertex:
            print(f"Method 2 - from joint2vertex: {sorted(set(grains_from_joint2vertex))}")
        else:
            print(f"Vertex {vertex_id} not found in joint2vertex values")
        
        # 方法3：检查vertex是否存在
        if vertex_id in self.vertices:
            print(f"Vertex {vertex_id} coordinates: {self.vertices[vertex_id]}")
        else:
            print(f"Vertex {vertex_id} not found in vertices")
        
        return grains_list


    def add_nucleation_to_x(self, x_dict, edge_index_dict, mask):
        import random
        def shoelace(pts):
            a = 0.0
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i+1) % n]
                a += x1*y2 - x2*y1
            return abs(a) * 0.5

        M=max(self.region_coors.keys())+1-x_dict["grain"].size(0)
        print(max(self.region_coors.keys()))
        print(M)
        N= max(self.vertices.keys())+1-x_dict["joint"].size(0)
        print(N)
        print(x_dict["grain"].size(0))

        old_grain=x_dict["grain"]
        K,F=old_grain.shape
        new_grain_feats=torch.full((M,F),0.0, dtype=old_grain.dtype, device=old_grain.device)
        for rid in range(M):
            print(rid)
            if x_dict["grain"].size(0)+rid+1 not in self.regions:
                continue 
            vert_ids = self.regions[x_dict["grain"].size(0)+rid+1]
            print(vert_ids)

            coords = [self.vertices[v] for v in vert_ids]
            xs, ys = zip(*coords)

            # compute centroid & area
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            area   = shoelace(coords)

            new_grain_feats[rid,0]=x_mean
            new_grain_feats[rid,1]=y_mean
            new_grain_feats[rid,2]=old_grain[0,2]
            new_grain_feats[rid,3]=area

            theta_x_deg = random.uniform(0, 90)
            theta_z_deg = random.uniform(0, 90)

            # 2) Convert to radians
            theta_x = math.radians(theta_x_deg)
            theta_z = math.radians(theta_z_deg)

            # 3) Compute cos and sin
            x_dict["grain"][rid-1,5] = math.cos(theta_x)
            x_dict["grain"][rid-1,6] = math.sin(theta_x)
            x_dict["grain"][rid-1,7] = math.cos(theta_z)
            x_dict["grain"][rid-1,8] = math.sin(theta_z)

            x_dict["grain"][rid-1,10]=old_grain[0,-1]

        x_dict["grain"] = torch.cat([old_grain, new_grain_feats], dim=0)

        print("what", self.region2newidx)


        for rid, _ in self.region2newidx.items():
            vert_ids = self.regions[rid]
            coords = [self.vertices[v] for v in vert_ids]
            print("region id", rid)
            xs, ys = zip(*coords)

            # compute centroid & area
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            area   = shoelace(coords)
            print("Position", x_mean)
            print("Position", x_dict["grain"][rid-1,0])
            x_dict["grain"][rid-1,0]=x_mean
            x_dict["grain"][rid-1,1]=y_mean
            change=area-x_dict["grain"][rid-1,3]
            x_dict["grain"][rid-1,3]=area

            theta_x_deg = random.uniform(0, 90)
            theta_z_deg = random.uniform(0, 90)

            # 2) Convert to radians
            theta_x = math.radians(theta_x_deg)
            theta_z = math.radians(theta_z_deg)

            x_dict["grain"][rid-1,5] = math.cos(theta_x)
            x_dict["grain"][rid-1,6] = math.sin(theta_x)
            x_dict["grain"][rid-1,7] = math.cos(theta_z)
            x_dict["grain"][rid-1,8] = math.sin(theta_z)
            x_dict["grain"][rid-1,9] = change
            x_dict["grain"][rid-1,10] = 0
            print("AREA", change)
            print(x_dict["grain"][rid-1,:])

        old_joint=x_dict["joint"]
        print(old_joint[0,:])
        K,F=old_joint.shape
        new_joint_feats=torch.full((N,F),0.0,dtype=old_joint.dtype, device=old_joint.device)
        print("size",x_dict["grain"].size(0))

        for vid in range(0, N):
            x, y = self.vertices[x_dict["joint"].size(0)+vid]
            new_joint_feats[vid, 0] = float(x)   
            new_joint_feats[vid, 1] = float(y)    
            new_joint_feats[vid, 2] = old_joint[0,2]
            new_joint_feats[vid, 3]=old_joint[0,3]
            new_joint_feats[vid, 4]=old_joint[0,4]


        x_dict["joint"] = torch.cat([old_joint, new_joint_feats], dim=0)
        print("size",x_dict["joint"].size(0))

        # print(self.vertex2joint)

        joint_ids = []
        grain_ids = []
        for joint, grains in self.vertex2joint.items():
            for g in grains:
                joint_ids.append(joint)
                grain_ids.append(g-1)
        edge_index = torch.tensor([joint_ids, grain_ids], dtype=torch.long)
        print(edge_index)
        edge_index_dict[('joint', 'pull', 'grain')] = edge_index

        srcs = []
        dsts = []
        for j, neighs in self.vertex_neighbor.items():
            for k in neighs:
                # if you want undirected, and avoid doubles, do:
                #     if k <= j: 
                #         continue
                srcs.append(j)
                dsts.append(k)

        # 2) Stack into a 2×E LongTensor
        edge_jj = torch.tensor([srcs, dsts], dtype=torch.long)
        print(edge_jj)
        # 3) Assign into your edge_index_dict
        edge_index_dict[('joint', 'connect', 'joint')] = edge_jj

        # print(mask['joint'][:,0])
        new_junctions = list(range(x_dict["joint"].size(0)-N, max(self.vertices.keys())+1))

        mj = mask['joint']  
        D = mj.size(1)                  
        pad = torch.zeros(len(new_junctions), D,
                        dtype=mj.dtype, device=mj.device)
        mj = torch.cat([mj, pad], dim=0)   # now shape [236+6, D]
        mj[new_junctions, 0] = 1
        for i in self.covered:
            mj[i, 0] = 0
        mask['joint'] = mj
        # print(mask['joint'][:,0])

        print(x_dict["joint"].size(0)-N, max(self.vertices.keys())+1)

        new_grains= list(range(x_dict["grain"].size(0)-M, max(self.region_coors.keys())+1))
        print(x_dict["grain"].size(0)-M, max(self.region_coors.keys()))
        print(new_grains)
        # print(mask['grain'][:,0])
        mg=mask['grain']
        D=mg.size(1)
        pad2=torch.zeros(len(new_grains),D,dtype=mg.dtype, device=mg.device)
        mg=torch.cat([mg,pad2], dim=0)
        mg[new_grains,0]=1
        print(self.replaced)
        for i in self.replaced:
            mg[i-1,0]=0
        mask['grain']=mg
        # print(mask['grain'][:,0])
        print(x_dict["grain"].size(0)-M, max(self.region_coors.keys()))

        return x_dict, edge_index_dict

    def update(self, init = False):
        
        """
        Input: joint2vertex, vertices, edges, 
        Output: region_coors, vertex_neighbor
        """
      #  self.edge_len.clear()
        
      #  self.vertex2joint = dict((v, k) for k, v in self.joint2vertex.items())
        self.vertex_neighbor.clear()                    
   
        # form region
        self.regions.clear()
        self.region_coors.clear()
        self.region_center.clear()
        self.region_edge.clear()
        self.vertex_coord_to_index.clear()
        self.index_to_vertex_coord.clear()
        region_bound = defaultdict(list)
        
        for k, v in self.joint2vertex.items():
            for region in set(k):

                self.regions[region].append(v)
                
                self.region_coors[region].append(self.vertices[v])

                coord = self.vertices[v]
                if isinstance(coord, np.ndarray):
                    coord = tuple(coord.tolist())
                elif isinstance(coord, list):
                    coord = tuple(coord)
                # Store the mapping: key = coordinate, value = vertex index
                self.vertex_coord_to_index[coord] = v
                self.index_to_vertex_coord[v]=coord
        cnt = 0
        edge_count = 0

        for region, verts in self.region_coors.items():

            if len(verts)<=1: continue
        #    assert len(verts)>1, ('one vertex is not a grain ', region, self.regions[region])
            
            moved_region = []

            vert_in_region = self.regions[region]
                
            if self.BC == 'periodic':
                for i in range(1, len(vert_in_region)):
                    verts[i] = periodic_move(verts[i], verts[i-1]) 
            if self.BC == 'noflux' and region>1:
                verts_array = np.array(verts)
                region_bound[region] = [np.min(verts_array[:,0]), np.max(verts_array[:,0]),
                                             np.min(verts_array[:,1]), np.max(verts_array[:,1])]
                
            inbound = [True, True]
            
            for vert in verts:
                inbound = [i and (j>-eps) for i, j in zip(inbound, vert)]  
            for vert in verts:
                vert = [i + 1*(not j) for i, j in zip(vert, inbound)]
                moved_region.append(vert)
        
            x, y = zip(*moved_region)
         
            self.region_center[region] = [np.mean(x), np.mean(y)]
            
            
            
            sort_index = sorted(range(len(moved_region)), \
                key = lambda x: counterclock(moved_region[x], self.region_center[region]))  
            
            if self.BC == 'noflux' and region == 1:
                sort_index.reverse()
                
            self.region_coors[region] = [moved_region[i] for i in sort_index]
            
            sorted_vert = [vert_in_region[i] for i in sort_index]
            self.regions[region] = sorted_vert


            cnt += len(vert_in_region) 

            
            if init:
                
                grain_edge = []
                save_edge = True
                
                for i in range(len(sorted_vert)):
                    cur = sorted_vert[i]
                    nxt = sorted_vert[i+1] if i<len(sorted_vert)-1 else sorted_vert[0]
                    
                    if region in self.quadruples:
                        if cur in self.quadruples[region] or nxt in self.quadruples[region]:
                            if not linked_edge_by_junction(self.vertex2joint[cur], self.vertex2joint[nxt]):
                                save_edge = False
                           
                    grain_edge.append([cur, nxt])
                
                if save_edge:
                    self.edges.extend(grain_edge)
                else:
                   # print('before',grain_edge)
                    v1, v2 = self.quadruples[region]
                    for e in grain_edge:
                        if e[0] == v1: e[0] = v2
                        elif e[0] == v2: e[0] = v1   
                        if e[1] == v1: e[1] = v2
                        elif e[1] == v2: e[1] = v1 
                   # print('after', grain_edge)
                    self.edges.extend(grain_edge)

      #  print('num vertices of grains', cnt)
        print('num edges, junctions', len([i for i in self.edges if i[0]>-1 ]), len(self.joint2vertex))        
        # form edge             

        for src, dst in self.edges:
            if src>-1:
                self.vertex_neighbor[src].add(dst)
                if src not in self.vertices:
                    print('in', self.vertex2joint[src])
                    print(src, 'not available')
                if dst not in self.vertices:
                    print(dst, 'not available',src) 
                    print('in', self.vertex2joint[dst])
                                       
               # self.edge_len.append(periodic_dist_(self.vertices[src], self.vertices[dst]))   
       # print('edge vertices', len(self.vertex_neighbor))  
        self.abnormal_points = []
        for v, n in self.vertex_neighbor.items():
            if len(n)!=3:
                print((v,n), self.vertices[v])
                self.abnormal_points.append(self.vertices[v])
               # raise ValueError((v,n))


       # self.compute_error_layer()
        if self.BC == 'noflux':
            
            remain_keys = np.array(list(region_bound.keys()))
            grain_bound = np.array(list(region_bound.values()))
            max_y = 1
            cone_ratio = 0
            
            if hasattr(self, 'max_y'):
                max_y = self.max_y
            if hasattr(self, 'cone_ratio'):
                cone_ratio = self.cone_ratio
            self.corner_grains[0] = remain_keys[(np.absolute(grain_bound[:,0])<1e-6) & (np.absolute(grain_bound[:,2] - cone_ratio)<1e-6)][0]  
            self.corner_grains[1] = remain_keys[(np.absolute(1-grain_bound[:,1])<1e-6) & (np.absolute(grain_bound[:,2])<1e-6)][0]
            self.corner_grains[2] = remain_keys[(np.absolute(grain_bound[:,0])<1e-6) & (np.absolute(max_y-cone_ratio-grain_bound[:,3])<1e-6)][0]  
            self.corner_grains[3] = remain_keys[(np.absolute(1-grain_bound[:,1])<1e-6) & (np.absolute(max_y-grain_bound[:,3])<1e-6)][0]  
           # print(self.corner_grains)
            
        if init:  
            self.plot_polygons()
            
            
    def find_boundary_vertex(self, alpha, cur_joint):
        
        """
        alpha shape is [nx, ny] here
        
        """
        m, n = alpha.shape
        s = self.imagesize[0]
        for i in range(m-1):
            if alpha[i, 0] != alpha[i+1, 0]:
                cur_joint.update({tuple(sorted([1, alpha[i, 0], alpha[i+1, 0]])): [i/s, 0,  3]})
            if alpha[i, -1] != alpha[i+1, -1]:
                cur_joint.update({tuple(sorted([1, alpha[i, -1], alpha[i+1, -1]])): [i/s, n/s, 3]}) 
                
        for i in range(n-1):
            if alpha[0, i] != alpha[0, i+1]:
                cur_joint.update({tuple(sorted([1, alpha[0, i], alpha[0, i+1]])): [0, i/s, 3]})
            if alpha[-1, i] != alpha[-1, i+1]:
                cur_joint.update({tuple(sorted([1, alpha[-1, i], alpha[-1, i+1]])): [m/s, i/s, 3]})                 
                
        
        return
                
class GrainHeterograph:
    def __init__(self):
        self.features = {'grain':['x', 'y', 'z', 'area', 'extraV', 'cosx', 'sinx', 'cosz', 'sinz', 'span'],
                         'joint':['x', 'y', 'z', 'G', 'R', 'span']}
        self.mask = {}
        
        self.features_grad = {'grain':['darea'], 'joint':['dx', 'dy']}
        
    
        self.targets = {'grain':['darea', 'extraV'], 'joint':['dx', 'dy']}
        self.events = {'grain_event':'elimination', 'edge_event':'rotation'}    
        
        self.edge_type = [('grain', 'push', 'joint'), \
                          ('joint', 'pull', 'grain'), \
                          ('joint', 'connect', 'joint')]
        
        self.targets_scaling = {'grain':20, 'joint':5}    
            
        self.feature_dicts = {}
        self.target_dicts = {}
        self.edge_index_dicts = {}
        self.edge_weight_dicts = {}
        self.additional_features = {}
        
        self.physical_params = {}

    def form_gradient(self, prev, nxt, event_list, elim_list):
        
        self.event_list = event_list
        
        """
            
        Gradients for next prediction
            
        """        
        
        if nxt is not None:

        
            darea = nxt.feature_dicts['grain'][:,3:4] - self.feature_dicts['grain'][:,3:4]

           # for grain, scaleup in elim_list:
           #     if darea[grain]<=0:
           #         darea[grain] *= scaleup

            self.target_dicts['grain'] = self.targets_scaling['grain']*\
                np.hstack((darea, nxt.feature_dicts['grain'][:,4:5]))
                                         
            self.target_dicts['joint'] = self.targets_scaling['joint']*\
               self.subtract(nxt.feature_dicts['joint'][:,:2], self.feature_dicts['joint'][:,:2], 'next')

            
           # self.additional_features['nxt'] = nxt.edge_index_dicts
            
            

            ''' gradients '''
            
            # check if the grain neighbor of the junction is the same
            for i in range(len(self.mask['joint'])):
                if self.mask['joint'][i,0] == 1:
                    if i in nxt.vertex2joint and set(self.vertex2joint[i]) == set(nxt.vertex2joint[i]):
                        pass
                    else:
                        self.mask['joint'][i,0] = 0
                      #  print('not matched', i, self.vertex2joint[i])
                      


            
            '''edge'''

            self.edges = [[src, dst] for src, dst in self.edges if src>-1 and dst>-1]
            self.target_dicts['edge_event'] = -100*np.ones(len(self.edges), dtype=int)
 
            for i, pair in enumerate(self.edges):
                if pair in nxt.edges:
                    if tuple(pair) in event_list:
                        self.target_dicts['edge_event'][i] = 1
                    else:
                        self.target_dicts['edge_event'][i] = 0
                    
            print('number of positive/negative events', \
                  sum(self.target_dicts['edge_event']>0), sum(self.target_dicts['edge_event']==0))
            
            
            edge_pair = []    
            for i, el in enumerate(self.edge_weight_dicts[self.edge_type[2]][:,0]):
                if el > -1:
                    edge_pair.append([el, nxt.edge_weight_dicts[self.edge_type[2]][i,0]])
            
            assert len(self.edges) == len(edge_pair)
            
            self.mask['edge'] = np.ones(len(self.edges), dtype=int)
            self.target_dicts['edge'] = np.zeros(len(self.edges))
            
            for i, (el, el_n) in enumerate(edge_pair):
                
                if self.target_dicts['edge_event'][i]>0:
                    self.target_dicts['edge'][i] = 0.5*self.targets_scaling['joint']*(-el_n-el)
            
                else:
                    self.target_dicts['edge'][i] = 0.5*self.targets_scaling['joint']*(el_n-el)
                
                if self.target_dicts['edge_event'][i]<0 or el_n<-1:
                    self.mask['edge'][i] = 0
            

            
            
                
            '''grain'''    
                
                
            self.target_dicts['grain_event'] = np.zeros(len(self.mask['grain']), dtype=int)    
            for i in range(len(self.mask['grain'])):
                if self.mask['grain'][i] == 1 and nxt.mask['grain'][i] == 0:
                    self.target_dicts['grain_event'][i] = 1
                
            print('number of grain events', np.sum(self.target_dicts['grain_event']))



            self.gradient_max = {'joint':np.max(np.absolute(self.mask['joint']*self.target_dicts['joint'])),
                                 'grain':np.max(np.absolute(self.target_dicts['grain'])),
                                 'edge':np.max(np.absolute(self.mask['edge']*self.target_dicts['edge']))}   
            
            gradscale = np.absolute(self.mask['joint']*self.target_dicts['joint'])
            gradscale = gradscale[gradscale>0]
            
            self.gradient_scale = {'joint':np.mean(gradscale),\
                                   'grain':np.mean(np.absolute(self.target_dicts['grain']))}     
                
            print('maximum gradient', self.gradient_max)
            print('average gradient', self.gradient_scale)
            
            assert np.all(self.mask['joint']*self.target_dicts['joint']>-1) \
               and np.all(self.mask['joint']*self.target_dicts['joint']<1)
            assert np.all(self.target_dicts['grain']>-1) and (np.all(self.target_dicts['grain']<1))
            assert np.all(self.mask['edge']*self.target_dicts['edge']>-1) \
               and np.all(self.mask['edge']*self.target_dicts['edge']<1) 
            
           # del self.edges
           # del self.vertex2joint
                       
        """
            
        Gradients of history
            
        """
        
        
                                     
        if prev is None:
            self.prev_grad_grain = 0*self.feature_dicts['grain'][:,:1]
            self.prev_grad_joint = 0*self.feature_dicts['joint'][:,:2]
            self.prev_grad_edge  = 0*self.edge_weight_dicts[self.edge_type[2]][:,:1]
                    
        else:
            self.prev_grad_grain = self.targets_scaling['grain']*\
                (self.feature_dicts['grain'][:,3:4] - prev.feature_dicts['grain'][:,3:4]) 
            self.prev_grad_joint = self.targets_scaling['joint']*\
                self.subtract(self.feature_dicts['joint'][:,:2], prev.feature_dicts['joint'][:,:2], 'prev')
               # (self.feature_dicts['joint'][:,:2] - prev.feature_dicts['joint'][:,:2])             
            self.prev_grad_edge  = 0.5*self.targets_scaling['joint']*\
                self.subtract(self.edge_weight_dicts[self.edge_type[2]][:,:1], prev.edge_weight_dicts[self.edge_type[2]][:,:1], 'prev')
        
        self.feature_dicts['grain'][:,4] *= self.targets_scaling['grain']
        
        
        
        
        self.feature_dicts['grain'][:, len(self.features['grain'])-1] = self.span/120 
        self.feature_dicts['joint'][:, len(self.features['joint'])-1] = self.span/120
                                                 
        self.feature_dicts['grain'] = np.hstack((self.feature_dicts['grain'], self.prev_grad_grain))

        self.feature_dicts['joint'] = np.hstack((self.feature_dicts['joint'], self.prev_grad_joint)) 
                
        
      #  self.edge_weight_dicts[self.edge_type[2]] = np.hstack((self.edge_weight_dicts[self.edge_type[2]], 
      #                                                         self.prev_grad_edge)) 

        
        for nodes, features in self.features.items():
            self.features[nodes] = self.features[nodes] + self.features_grad[nodes]  
            assert len(self.features[nodes]) == self.feature_dicts[nodes].shape[1]


    @staticmethod
    def subtract(b, a, loc):
        
        short_len = len(a)
        
        if loc == 'prev':
            return np.concatenate((b[:short_len,:]-a, 0*b[short_len:,:]), axis=0)
            
        if loc == 'next':
            return b[:short_len,:]-a


    @staticmethod
    def fillup(b, a):

        short_len = len(a)
        
        return np.concatenate((a, 0*b[short_len:,:]), axis=0)
        
    def append_history(self, prev_list):
        
        exist = np.where(self.edge_weight_dicts[self.edge_type[2]][:,0]>-1)[0]
        self.edge_weight_dicts[self.edge_type[2]] = self.edge_weight_dicts[self.edge_type[2]][exist,:]
        
        
        for prev in prev_list:
            
            if prev is None:           
                prev_grad_grain = 0*self.feature_dicts['grain'][:,:1]
                prev_grad_joint = 0*self.feature_dicts['joint'][:,:2]  
            #    prev_edge_len = 0*self.edge_weight_dicts[self.edge_type[2]][:,:1]
            else:
                prev_grad_grain = self.fillup(self.prev_grad_grain, prev.prev_grad_grain)
                prev_grad_joint = self.fillup(self.prev_grad_joint, prev.prev_grad_joint)
            #    prev_edge_len = self.fillup(self.edge_weight_dicts[self.edge_type[2]][:,:1],
            #                                prev.edge_weight_dicts[self.edge_type[2]][:,:1])
            
            self.feature_dicts['grain'] = np.hstack((self.feature_dicts['grain'], prev_grad_grain))
            self.feature_dicts['joint'] = np.hstack((self.feature_dicts['joint'], prev_grad_joint))                                       
            
         #   self.edge_weight_dicts[self.edge_type[2]] = np.hstack((self.edge_weight_dicts[self.edge_type[2]], 
         #                                                          prev_edge_len))   
            
        return
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Generate heterograph data")
    parser.add_argument("--mode", type=str, default = 'check')
    parser.add_argument("--seed", type=int, default = 1)

    args = parser.parse_args()        
        
    if args.mode == 'check':

        seed = 10020
       # g1 = graph(lxd = 40, seed = seed, BC = 'noflux') 
        
        g1 = graph(user_defined_config=user_defined_config())

        g1.show_data_struct()

       # g1.plot_grain_distribution()
    
    if args.mode == 'instance':
        
        for seed in range(args.seed*12, (args.seed+1)*12):
            print('\n')
            print('test seed', seed)

            g1 = graph(lxd = 40, seed=seed) 


          #  g1.show_data_struct() 

