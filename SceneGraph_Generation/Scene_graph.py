import math
from typing import Dict, List
from collections import deque

class SceneGraph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.next_id = 0
        
    def add_node(self, label, center, description, mask):
        node = Node(label, center, description, mask)
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
        for idx, node in enumerate(self.nodes.items()):
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
        
class Node:
    def __init__(self,
                 label,
                 center = None,
                 description = None,
                 mask = None):
        self.label = label
        self.center = center # (lon, lat)
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