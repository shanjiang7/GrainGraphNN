#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:27:17 2022

@author: yigongqin
"""


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
from user_generate import user_defined_config

def angle_norm(angular):

    return - ( 2*(angular + pi/2)/(pi/2) - 1 )

eps = 1e-12
def in_bound(x, y, max_y=1, cone_ratio = 0):
    
    if x>=-eps and x<=1+eps and y>=-eps + cone_ratio*(1-x) and y<=max_y - cone_ratio*(1-x) +eps:
        return True
    else:
        return False
    


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
        self.edges = []  ## index linkage
       # self.edge_len = []
        self.regions = defaultdict(list) ## index group
        self.region_coors = defaultdict(list)
        self.region_edge = defaultdict(set)
        self.region_center = defaultdict(list)
        self.vertex_coord_to_index = {}
        self.index_to_vertex_coord = {}
        self.elim_grain_to_vertices={}
        self.edge_prob_dict = {}
        self.junct_gradient_dict={}

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
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True)
            plt.savefig('planar_graph.png',dpi=600)


        # if not is_planar:
        #     print("counterexample:",counterexample.edges())
        #     pos = nx.spring_layout(counterexample)
        #     nx.draw(counterexample, pos, with_labels=True)
        #     plt.savefig('planar_graph.png',dpi=600)

        print("Graph is planar:", is_planar)
    
    def snap_classifer(self,fname='snap_classifer'):
        fig, ax = plt.subplots(figsize=(8, 6))
        def sort_polygon(coords):
            cx = sum([p[0] for p in coords]) / len(coords)
            cy = sum([p[1] for p in coords]) / len(coords)
            
            # Sort the points by the angle they make with the centroid
            sorted_coords = sorted(coords, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            return sorted_coords
        color = 'blue'
        alpha = 0.5
        print('elimgrains',self.elim_grain_to_vertices.items())
        for grain, vertex_indices in self.elim_grain_to_vertices.items():
            # Get coordinates for all vertices of this grain.
            coords = [self.index_to_vertex_coord[v] for v in vertex_indices]
            
            # Check we have enough points
            if len(coords) < 3:
                continue
            
            ref=coords[0]
            # ref = None
            # for coord in coords:
            #     if coord[0] > 0.5 and coord[1] > 0.5:
            #         ref = coord
            #         break
            # if ref is None:
            #     ref = coords[0] 

            # Adjust every coordinate so that it is in the same periodic cell as the reference.
            adjusted_coords = [periodic_move(coord, ref) for coord in coords]
            
            # Now sort the adjusted coordinates into a proper polygon order.
            sorted_coords = sort_polygon(adjusted_coords)  # Assume sort_polygon is defined elsewhere
            
            # Create a polygon patch from the sorted coordinates
            polygon = patches.Polygon(sorted_coords, closed=True,
                                    facecolor=color, edgecolor='black', alpha=alpha)
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
        plt.savefig(fname, dpi=300)

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
            if (movement[0]**2 + movement[1]**2) < (0.04**2):
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

