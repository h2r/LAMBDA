import networkx as nx
import matplotlib.pyplot as plt
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2
import numpy as np
import transformations as tf
import pdb
import os 

graph_fpath = "Trajectories/graphs/upstairs"
SAVE_NAME = 'upstairs.png'

with open(graph_fpath + '/graph', 'rb') as graph_file:
    # Load the graph from disk.
    data = graph_file.read()
    graph = map_pb2.Graph()
    graph.ParseFromString(data)
    print(
        f'Loaded graph has {len(graph.waypoints)} waypoints and {len(graph.edges)} edges'
    )

# pdb.set_trace()
objs = [waypoint for waypoint in graph.waypoints]
id2loc = {obj.id: [obj.seed_tform_waypoint.position.x, obj.seed_tform_waypoint.position.y, obj.seed_tform_waypoint.position.z] for obj in graph.anchoring.anchors}
obj2id = {obj.annotations.name: obj.id for obj in objs}
id2obj = {obj.id: obj.annotations.name for obj in objs}


max_z = float('-inf'); min_z = float('inf')
for obj in graph.anchoring.anchors:

    max_z = max(obj.seed_tform_waypoint.position.z, max_z)
    min_z = min(obj.seed_tform_waypoint.position.z, min_z)

print('MAX: ', max_z, 'MIN: ', min_z)
map_2d = False

if map_2d:
    # convert spot nva graph to nx graph for planning
    connect_graph = nx.Graph()
    
    nodes = {id:  (id2loc[id][0], id2loc[id][1]) for id in id2loc.keys()}
    connect_graph.add_nodes_from(nodes)

    connect_graph.add_edges_from([(e.id.from_waypoint, e.id.to_waypoint) for e in graph.edges])
    # node_positions = {node: nodes[node]['pos'] for node, data in connect_graph.nodes(data=True)}
    node_positions = nodes
    # pdb.set_trace()
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(connect_graph, node_positions, ax=ax, node_size=200, node_color='skyblue')

    # Draw edges
    nx.draw_networkx_edges(connect_graph, node_positions, ax=ax, width=2, edge_color='gray')
else:
    nodes = {id: [id2loc[id][0], id2loc[id][1], id2loc[id][2]] for id in id2loc.keys()}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(xs= [nodes[x][0] for x in nodes.keys()], ys = [nodes[x][1] for x in nodes.keys()], zs = [nodes[x][2] for x in nodes.keys()], s=20, ec="w", c='skyblue')

    edges = np.array([(nodes[e.id.from_waypoint], nodes[e.id.to_waypoint]) for e in graph.edges])

    # Plot the edges
    for e in edges:
        ax.plot(*e.T, color="tab:gray")

# Add labels to nodes
# traj = ['pebbly-moa-Lk1eKDsWctdahyFUYmuOwQ==', 'apodal-guppy-GiKnSqr5tjLg0mxJrl3D3A==', 'spayed-chunga-mzIz4YM6uDcOTQs4BE6Hog==', 'tensed-boa-xYAA6hlK03sNoyCFv9myLg==', 'barky-lion-CnIv1DhdcBLYaykAMePqYQ==', 'yucky-biped-P4EZeKZ83gNVAM6NVUilwQ==', 'chafed-goose-9EUBS5JEIqzvEADiAaBruA==', 'moldy-cicada-vvOonrkUh3zeP.FvnbpGMQ==', 'fogged-chimp-SciMoeEkEKkwzELllK1GVw==', 'bulgy-ermine-eevg2YudQjiYiB4j4PmfnA==', 'meager-sponge-QNVwDAR+yBl6kFr0xbVQaA==', 'surly-eel-L3WHxzLsncHzXd.VbkuLdw==', 'finer-weasel-1beejRvUz0yBJASf7zbPaQ==', 'gummed-oryx-guW37qvkWHzT9aXma7CxZw==', 'peeved-embryo-Umsv5ZvvrktexhGh7RJpsw==', 'fly-jackal-YH7k7fOllCW3qBmzZLIJmQ==']# labels = {node: id2obj[node] for node in connect_graph.nodes() if node in obj2id.values()}
# pdb.set_trace()
# labels = {node: id2obj[node] for node in connect_graph.nodes() if node in obj2id.values()}
# labels.update({node: '0' for node in connect_graph.nodes() if node in traj})
# nx.draw_networkx_labels(connect_graph, node_positions, labels=labels, font_size=10, font_color='black')

# Set plot limits to ensure all nodes are visible
# ax.set_xlim(0, 8)
# ax.set_ylim(-5, 14)
# breakpoint()
# Show the plot
plt.axis('off')  # Turn off axis labels and ticks
plt.savefig(os.path.join('.',SAVE_NAME))
plt.show()




