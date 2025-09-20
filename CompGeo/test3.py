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
import random
import argparse
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from matplotlib.path import Path

eps = 1e-12

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

def in_bound(x, y, max_y=1, cone_ratio = 0):
    
    if x>=-eps and x<=1+eps and y>=-eps + cone_ratio*(1-x) and y<=max_y - cone_ratio*(1-x) +eps:
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

def in_wedge(P, P_prev, P_next, H, tol=1e-6):
    """
    Check if the candidate point H (as (x,y)) lies inside the wedge at P defined by rays to P_prev and P_next.
    Assumes the pentagon is convex so that the interior angle at P is less than 180°.
    One way: if the sum of angles between (P_prev->P, H->P) and (H->P, P_next->P)
    equals the interior angle (within tolerance), then H is inside the wedge.
    """
    def angle_between(v1, v2):
        # returns the angle in radians between vectors v1 and v2
        dot = np.dot(v1, v2)
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        # Clamp for numerical issues
        cos_angle = max(min(dot/norm_prod, 1), -1)
        return math.acos(cos_angle)
    
    # Create vectors from P
    v_prev = np.array(P_prev) - np.array(P)
    v_next = np.array(P_next) - np.array(P)
    v_H = np.array(H) - np.array(P)
    
    angle_total = angle_between(v_prev, v_next)
    angle_candidate = angle_between(v_prev, v_H) + angle_between(v_H, v_next)
    
    # If the candidate is inside the wedge, these angles should nearly match.
    return abs(angle_candidate - angle_total) < tol

class graph:
    def __init__(self, randInit: bool = True, noise: float =0.0001, lxd: float = 40, BC: str= 'periodic', seed: int=10020):
        self.BC = BC
        self.noise=noise
        self.lxd=lxd
        self.lyd=self.lxd
        self.seed=seed

        self.vertices = defaultdict(list)
        self.edges = []
        self.pentagon_region_id=defaultdict(list)
        self.vertex2joint = defaultdict(set)
        self.quadruples = {}
        self.ini_grain_size =  6
        self.density = self.ini_grain_size/self.lxd

        self.regions = defaultdict(list) ## index group
        self.region_coors = defaultdict(list)
        self.added_region=defaultdict(list)
        self.region_edge = defaultdict(set)
        self.region_center = defaultdict(list)
        self.vertex_neighbor = defaultdict(set)
        self.nucle_fall_region=[]
        self.intersection_points=[]
        self.regions_edges=defaultdict(list)
        self.covered=[]
        self.cutted_edges=[]
        self.nucle_map =defaultdict(list)
        self.added_edge=[]
        self.projected_vertices=[]
        self.inter_point=[]
        self.region2newidx=defaultdict(set)

        if randInit:
            np.random.seed(seed)
            if self.BC=='periodic':
                self.random_voronoi_periodic()
                self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())

    def snap(self, fname='snap'):

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



        if hasattr(self, 'added_edge'):
            for edge in self.added_edge:
                v1, v2 = edge  # each edge is (vertex_id1, vertex_id2)
                pt1 = self.vertices[v1]
                pt2 = self.vertices[v2]
                print("Point", v1, v2)
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='green', linestyle='--', linewidth=1)

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
        # for vid, coord in self.vertices.items():
        #     ax.text(coord[0], coord[1], str(vid), color='purple', fontsize=5,
        #             ha='center', va='center')
        # Draw reference lines and labels.
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        ax.axvline(0, color='gray', linewidth=1, linestyle='--')
        ax.text(1, 0, '1', va='center', ha='left', color='gray')
        ax.text(0, 1, '1', va='bottom', ha='center', color='gray')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(fname, dpi=300)

    
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
        self.nucle_fall_region.clear()
        self.nucle_map.clear()
        # shape_type = random.choice(["pentagon", "quadrilateral"])
        shape_type="pentagon"
        if shape_type == "pentagon":
            n_vertices = 5
        else:
            n_vertices = 4

        print(f"Adding a {shape_type} region.")

        polygon_points = []
        for i in range(n_vertices):
            angle = 2 * math.pi * i / n_vertices - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            # Add small random perturbation:
            perturb = perturb_ratio * radius 
            x += np.random.uniform(-perturb, perturb)
            y += np.random.uniform(-perturb, perturb)
            polygon_points.append((x, y))
        
        # Assign new vertex IDs for the polygon vertices.
        next_vertex_id = max(self.vertices.keys()) + 1 if self.vertices else 0
        new_vertex_ids = []
        for pt in polygon_points:
            self.vertices[next_vertex_id] = pt
            self.vertex2joint[next_vertex_id] = set() 
            new_vertex_ids.append(next_vertex_id)
            next_vertex_id += 1
        
        for i in range(n_vertices):
            v1 = new_vertex_ids[i]
            v2 = new_vertex_ids[(i + 1) % n_vertices]
            self.edges.append([v1, v2])

        
        new_region = max(self.region_coors.keys()) + 1 if self.region_coors else 0
        self.regions[new_region] = new_vertex_ids
        print("new idx", new_region,  new_vertex_ids)
        self.region_coors[new_region] = polygon_points
        self.added_region[new_region] = polygon_points
        xs, ys = zip(*polygon_points)
        self.region_center[new_region] = [np.mean(xs), np.mean(ys)]
        
        for v in new_vertex_ids:
            self.vertex2joint[v].add(new_region)
        
        # for v in new_vertex_ids:
        #     # self.vertex2joint[v] now contains the set of regions (e.g. {new_region})
        #     self.joint2vertex[tuple(sorted(self.vertex2joint[v]))] = v
        for v in new_vertex_ids:
            self.joint2vertex[(new_region, v)] = v

        # # Save the region id with an attribute depending on the shape.
        # if shape_type == "pentagon":
        #     self.pentagon_region_id[new_vertex_ids] = new_region
        # else:
        #     self.quadrilateral_region_id = new_region


        # Optionally, find which existing Voronoi vertices are covered by this new polygon.
        covered = self.find_vertices_covered_by_polygon(polygon_points,new_region)
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
            found_regions = self.find_regions_for_point((vx, vy), new_region)
            # self.joint2vertex[(new_region, found_regions)] = v_id
            self.nucle_fall_region.append(found_regions)
            self.nucle_map[found_regions].append(v_id)
            print(f"{shape_type.capitalize()} vertex {v_id} at ({vx:.3f}, {vy:.3f}) lies in region(s): {found_regions}")
        # print(self.nucle_fall_region)
        # print(self.nucle_map)



        self.intersection_points+=self.find_polygon_edge_intersections(polygon_points)

        self.connect()


        self.reconstruct(new_vertex_ids, new_region)
        print("MAP",self.region2newidx)

        self.vertex2joint = {k: v for k, v in self.vertex2joint.items() if k not in self.covered}
        self.joint2vertex = dict((tuple(sorted(v)), k) for k, v in self.vertex2joint.items())

        # print(self.vertex2joint)
        self.nucle_fall_region=[]
        self.intersection_points=[]
        self.covered=[]
        self.cutted_edges=[]
        self.nucle_map =defaultdict(list)
        self.added_edge=[]
        self.projected_vertices=[]
        self.inter_point=[]



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


    def remove_vertex(self):

        for cv in self.covered:
            if cv in self.vertices:
                del self.vertices[cv]
            if cv in self.vertex2joint:
                del self.vertex2joint[cv]
        for k in [key for key, val in self.joint2vertex.items() if val in self.covered]:
            del self.joint2vertex[k]

        new_edge_list=[]
        for edge in self.edges:
            v1,v2=edge
            if v1 in self.covered or v2 in self.covered:
                continue
            else:
                new_edge_list.append(edge)
        self.edges=new_edge_list



    def random_voronoi_periodic(self):

        mirrored_seeds, seeds = hexagonal_lattice(dx=self.density, noise = self.noise, BC = self.BC)
        vor = Voronoi(mirrored_seeds)     
    
        reordered_regions = []
        vert_map = {}
        vert_count = 0
        alpha = 0
        
        for region in vor.regions:
            flag = True
            inboundpoints = 0
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

                reordered_region = []
                
                for index in region:

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

        # for k, v in self.vertex2joint.items():
        #     if len(v)!=3:
        #         print(k, v) 


    def update(self, init = False):
        
        """
        Input: joint2vertex, vertices, edges, 
        Output: region_coors, vertex_neighbor
        """

        self.vertex_neighbor.clear()                    
   
        # form region
        self.regions.clear()
        self.region_coors.clear()
        self.region_center.clear()
        self.region_edge.clear()
        region_bound = defaultdict(list)

        for k, v in self.joint2vertex.items():
            for region in set(k):

                self.regions[region].append(v)
                
                self.region_coors[region].append(self.vertices[v])
        # print(self.regions[37])
        # print(self.regions[4])
        # print(self.regions[35])

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
                    self.regions_edges[region].append(grain_edge)
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
                    self.regions_edges[region].append(grain_edge)

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
        # if init:  
        #     self.plot_polygons()


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

            print("CHECK",)
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
                            intersections.append(inter)
                            Double_inter.append(inter)
                            new_vid = get_new_vertex_id(self)
                            self.inter_point.append(new_vid )
                            intersection_vertex_map[(inter.x, inter.y)] = new_vid
                            self.vertices[new_vid] = (inter.x, inter.y)
                if len(Double_inter)==2: 
                    choose_cutted_edge(self, v1, v2, Double_inter)
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
                            self.cutted_edges.append([surviving, new_vid])

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
        for region, new_vertices in self.nucle_map.items():

            print(region, "REGION")
            cand_vertex = [vid for vid, joint in self.vertex2joint.items() if region in joint]
            filtered_vertex = [v for v in cand_vertex if v not in self.covered]
            sorted_vertices=sort_indices_clockwise(self,filtered_vertex)
            cp_sorted=copy.copy(sorted_vertices)

            CW=sort_indices_clockwise(self, new_vertices)
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
                chose_idx, dist= self.choose_hex_connection_for_pent_vertex( idx, polygon_ids, sorted_vertices)
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
                self.vertices.update(dict1)
                self.projected_vertices=project_list1
                print("CW")
            else:
                self.added_edge.extend(CCW_edge)
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
        P = self.vertices[pent_vertex_id]
        
        # Retrieve the ordered pentagon vertex list. (Assume we have stored the pentagon's region ID.)
        pentagon_ids = self.regions[next(iter(polygon_ids))]

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
            if all_angles_convex(P, P_prev, P_next, H, tol=tol):
                # Compute Euclidean distance from P to H.
                distance =np.linalg.norm(np.array(P) - np.array(H))
                valid_candidates.append((h_id, distance))
        print("Valid",valid_candidates)
        if valid_candidates:
            # Choose the candidate with the smallest distance.
            valid_candidates.sort(key=lambda x: x[1])
            chosen_id, min_distance = valid_candidates[0]
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

if __name__ == '__main__':
    g1 = graph(lxd = 40)
    g1.update(init=True)
    # g1.add_polygon(center=(0.4105, 0.52), radius=0.09, perturb_ratio=0.1)

    # g1.add_polygon(center=(0.31, 0.32), radius=0.02, perturb_ratio=0.2)
    # g1.update()
    # g1.add_polygon(center=(0.65, 0.3), radius=0.1, perturb_ratio=0.2)

    g1.add_polygon(center=(0.32, 0.55), radius=0.06, perturb_ratio=0.2)
    g1.update()
    g1.consolidate_and_remove_best()


    # g1.add_polygon(center=(0.5, 0.6), radius=0.06, perturb_ratio=0.2)
    # g1.update()

    # g1.add_polygon(center=(0.6, 0.65), radius=0.04, perturb_ratio=0.2)
    # g1.update()


    # g1.add_polygon(center=(0.73, 0.78), radius=0.04, perturb_ratio=0.2)
    # g1.update()

    g1.snap()
