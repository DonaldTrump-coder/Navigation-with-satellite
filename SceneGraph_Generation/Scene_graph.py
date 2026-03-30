import math

class SceneGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
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
                 source,
                 traget,
                 direction # from traget -> source
                 ):
        self.source = source
        self.traget = traget
        self.direction = direction