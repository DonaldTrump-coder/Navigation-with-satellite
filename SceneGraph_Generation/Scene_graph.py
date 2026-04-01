import math
from typing import Dict, List
from collections import deque
import numpy as np
import cv2

class SceneGraph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.next_id = 0
        
    def add_node(self, label, center, pix_center, description, mask):
        node = Node(label, center, pix_center, description, mask)
        self.nodes[self.next_id] = node
        self.next_id += 1
        
    def add_edge(self, source_id, target_id):
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        d_lon = source_node.center[0] - target_node.center[0]
        d_lat = source_node.center[1] - target_node.center[1]
        angle = math.atan2(d_lat, d_lon)
        angle = math.degrees(angle)
        if -20 <= angle < 20:
            direction = "on the east side of"
        elif 20 <= angle < 70:
            direction = "on the north-east side of"
        elif 70 <= angle < 110:
            direction = "on the north side of"
        elif 110 <= angle < 160:
            direction = "on the north-west side of"
        elif 160 <= angle or angle < -160:
            direction = "on the west side of"
        elif -160 <= angle < -110:
            direction = "on the south-west side of"
        elif -110 <= angle < -70:
            direction = "on the south side of"
        else:
            direction = "on the south-east side of"
        edge = Edge(source_id, target_id, direction)
        self.edges.append(edge)
    
    def add_edges(self, id1, id2):
        self.add_edge(id1, id2)
        self.add_edge(id2, id1)
    
    def bfs_navigation(self, start_id, target_id):
        visited = set()
        queue = deque([start_id])
        parent = {start_id: None}
        while queue:
            current_node = queue.popleft()
            if current_node == target_id:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = parent[current_node]
                return path[::-1] # Reverse the path to get from start to target
            
            visited.add(current_node)
            
            # Check all neighbors (edges)
            for edge in self.edges:
                if edge.source == current_node and edge.target not in visited:
                    queue.append(edge.target)
                    parent[edge.target] = current_node
                elif edge.target == current_node and edge.source not in visited:
                    queue.append(edge.source)
                    parent[edge.source] = current_node
        
        return None
        
    def get_text(self):
        """Text:
        Node 0: Label: A, GeoCoordinates: (lon, lat), Description: A;
        Node 1: Label: B, GeoCoordinates: (lon, lat), Description: B;
        ...
        Edges:
         - <Node source_id>(source Label) <on the relation of> <Node target_id>(target Label)
         ...
        """
        text = ""
        for idx, node in enumerate(self.nodes.values()):
            text += f"Node {idx}: Label: {node.label}, GeoCoordinates: (lon: {node.center[0]}, lat: {node.center[1]}), Description: {node.description}\n"
        text += "\nEdges:\n"
        for edge in self.edges:
            source_node = edge.source
            target_node = edge.target
            source_label = self.nodes[source_node].label
            target_label = self.nodes[target_node].label
            relation = edge.direction
            text += f" - <Node {source_node}>({source_label}) <{relation}> <Node {target_node}>({target_label})\n"
        return text
    
    def get_flight_points(self,
                          start, # (x, y)
                          front_id = None,
                          object_ids = None,
                          next_id = None,
                          max_interval = None,
                          expand_rate = None # >1
                          ):
        if front_id is None:
            front_point = start
        else:
            front_point = self.nodes[front_id].pix_center
        if next_id is None:
            next_point = start
        else:
            next_point = self.nodes[next_id].pix_center
        
        object_masks = [self.nodes[obj_id].mask for obj_id in object_ids]
        combined_mask = np.zeros_like(object_masks[0], dtype=np.uint8)
        for mask in object_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours))
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_box = cv2.transform(np.array([box]), rot_mat)[0]
        
        expanded_box = []
        for point in rotated_box:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            expanded_point = (center[0] + dx * expand_rate, center[1] + dy * expand_rate)
            expanded_box.append(expanded_point)
        
        expanded_box = np.array(expanded_box, dtype=np.float32)
        inv_rot_mat = cv2.getRotationMatrix2D(center, -angle, 1)
        expanded_box = cv2.transform(np.array([expanded_box]), inv_rot_mat)[0]
        expanded_box = np.int32(expanded_box)
        
        min_distance = float('inf')
        start_point = None
        for point in expanded_box:
            distance = np.linalg.norm(np.array(front_point) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                start_point = point
        remaining_points = [point for point in expanded_box if not np.array_equal(point, start_point)]
        
        min_distance = float('inf')
        final_point = None
        for point in remaining_points:
            distance = np.linalg.norm(np.array(next_point) - np.array(point))
            if distance < min_distance:
                min_distance = distance
                final_point = point
        
        diag1_start, diag1_end = expanded_box[0], expanded_box[2]
        diag2_start, diag2_end = expanded_box[1], expanded_box[3]
        
        diagonal = False
        if (np.array_equal(start_point, diag1_start) and np.array_equal(final_point, diag1_end)) or (np.array_equal(start_point, diag1_end) and np.array_equal(final_point, diag1_start)):
            diagonal = True
        if (np.array_equal(start_point, diag2_start) and np.array_equal(final_point, diag2_end)) or (np.array_equal(start_point, diag2_end) and np.array_equal(final_point, diag2_start)):
            diagonal = True
        
        target_point = None
        times = 0
        interval = 0
        vertical_direction = 0
        if diagonal is True:
            dist = 0
            p1, p2, p3, p4 = expanded_box
            edges = [
                (p1, p3, p2, p4),
                (p2, p4, p1, p3)
            ]
            for edge in edges:
                if (np.array_equal(start_point, edge[0]) and np.array_equal(final_point, edge[1])) or (np.array_equal(start_point, edge[1]) and np.array_equal(final_point, edge[0])):
                    dist1 = np.linalg.norm(start_point - edge[2])
                    dist2 = np.linalg.norm(start_point - edge[3])
                    if dist1 < dist2:
                        target_point = edge[3]
                        dist = dist1
                        vertical_direction = (edge[2] - start_point) / dist
                    else:
                        target_point = edge[2]
                        dist = dist2
                        vertical_direction = (edge[3] - start_point) / dist
                    break
            times = math.ceil(dist / max_interval)
            if times % 2 != 0:
                times += 1
            interval = dist / times
            
        else:
            dist = np.linalg.norm(start_point - final_point)
            p1, p2, p3, p4 = expanded_box
            edges = (
                (p1, p2, p3, p4),
                (p2, p3, p1, p4),
                (p3, p4, p1, p2),
                (p4, p1, p2, p3)
            )
            for edge in edges:
                if (np.array_equal(start_point, edge[0]) and np.array_equal(final_point, edge[1])) or (np.array_equal(start_point, edge[1]) and np.array_equal(final_point, edge[0])):
                    dist1 = np.linalg.norm(start_point - edge[2])
                    dist2 = np.linalg.norm(start_point - edge[3])
                    if dist1 < dist2:
                        target_point = edge[2]
                    else:
                        target_point = edge[3]
                    break
            times = math.ceil(dist / max_interval)
            if times % 2 == 0:
                times += 1
            interval = dist / times
            vertical_direction = (final_point - start_point) / dist
            
        start_point = np.float64(start_point)
        target_point = np.float64(target_point)
        final_point = np.float64(final_point)
        direction = target_point - start_point
        flight_points = [start_point]
        flying_point = start_point.copy()
        for _ in range(times):
            flying_point += direction
            flight_points.append(flying_point.copy())
            flying_point += vertical_direction * interval
            flight_points.append(flying_point.copy())
            direction = -direction
        flight_points.append(final_point)
        
        return flight_points
            
def get_edge_length(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
            
class Node:
    def __init__(self,
                 label,
                 center = None,
                 pix_center = None,
                 description = None,
                 mask = None):
        self.label = label
        self.center = center # (lon, lat)
        self.pix_center = pix_center # (x, y)
        self.description = description
        self.mask = mask
        
class Edge:
    def __init__(self,
                 source, # idx
                 target, # idx
                 direction # from traget -> source
                 ):
        self.source = source
        self.target = target
        self.direction = direction
        
def pix2geo(x, y, min_lon, max_lon, min_lat, max_lat, width, height):
    lon = min_lon + (x / width) * (max_lon - min_lon)
    lat = max_lat - (y / height) * (max_lat - min_lat)
    return (lon, lat)