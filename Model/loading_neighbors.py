import pandas as pd

# load connections from csv
df = pd.read_csv(os.path.join(args.data, 'neighbors_manual_v7_rename.csv'))
connections = {}
node_ids = set()  # Set to store unique node IDs
for index, row in df.iterrows():
    node = int(row['road_segment'])
    neighbors = [int(x) for x in row[1:].dropna().tolist()]
    connections[node] = neighbors
    node_ids.add(node)
    node_ids.update(neighbors)

# Create a mapping from node IDs to indices
mapping_dict = {node_id: index for index, node_id in enumerate(sorted(node_ids))}
print(mapping_dict)


import pandas as pd
import numpy as np

# Apply the mapping to the 'road_segment' and neighbor columns
for col in df.columns:
    df[col] = df[col].map(mapping_dict)



# Get the list of IDs
ids = df['road_segment'].tolist()
#print(len(ids))
#sys.exit()

# Initialize an adjacency matrix with zeros
adj_edges = np.zeros((len(ids), len(ids)))

# Populate the adjacency matrix based on the dataframe
for index, row in df.iterrows():
    node = row['road_segment']

    for neighbor in row[1:]:
        if np.isnan(neighbor):
            continue
        adj_edges[int(node), int(neighbor)] = 1

for adj in adj_edges:
    print(adj)


#-----------------#

#a = torch.rand((64,32,44,4))

def get_outgoing_nodes(node, adj_matrix):
    """
    Get the nodes that have a direct connection from the given node.

    Parameters:
    node (int): The index of the node.
    adj_matrix (np.array): The adjacency matrix.

    Returns:
    list: A list of nodes that connect from the given node.
    """
    # Get the row corresponding to the node
    row = adj_matrix[node, :]

    # Get the indices of the elements that are 1
    outgoing_nodes = np.where(row == 1)[0]

    return outgoing_nodes.tolist()


def get_index(idx, adj_matrix):
    #target = a[:,:,2].unsqueeze(2)
    #print(target[0,0])

    # Get the nodes that connect from node 2
    outgoing_nodes = get_outgoing_nodes(idx, adj_matrix)
    '''
    outgoing_nodes_representations = a[:,:,outgoing_nodes]
    print('outgoing_nodes',outgoing_nodes)
    print('outgoing_nodes_representations',outgoing_nodes_representations[0,0])
    outgoing_mobility = outgoing_nodes_representations-target
    print('outgoing_mobility',outgoing_mobility[0,0])
    '''
    # Get the nodes that connect to node 2
    ingoing_nodes = get_outgoing_nodes(idx, adj_matrix.transpose(1,0))
    '''
    ingoing_nodes_representations = a[:,:,ingoing_nodes]
    print('ingoing_nodes',ingoing_nodes)
    print('ingoing_nodes_representations',ingoing_nodes_representations[0,0])
    ingoing_mobility = target-ingoing_nodes_representations
    print('ingoing_mobility',ingoing_mobility[0,0])
    '''
    return outgoing_nodes,ingoing_nodes

outgoing = []
ingoing = []
for idx in range(len(adj_edges)):
  out_nodes, in_nodes = get_index(idx, adj_edges)
  outgoing.append(out_nodes)
  ingoing.append(in_nodes)

print('outgoing', outgoing)
print('ingoing', ingoing)


def get_node_name(index, node_dict):
    # Create a reverse mapping from indices to node names
    index_to_node = {v: k for k, v in node_dict.items()}
    return index_to_node.get(index, "Node index not found")

#node_mapping = {'1_to_3': 0, '2_to_3': 1, '3_to_1': 2, '3_to_2': 3, '3_to_4': 4, '4_to_3': 5, '4_to_5': 6, '5_to_4': 7, '5_to_6': 8, '6_to_5': 9, '6_to_7': 10, '7_to_6': 11, '7_to_8': 12, '8_to_7': 13, '8_to_13': 14, '8_to_14': 15, '9_to_10': 16, '9_to_12': 17, '10_to_9': 18, '10_to_41': 19, '12_to_9': 20, '12_to_13': 21, '12_to_37': 22, '13_to_12': 23, '13_to_14': 24, '13_to_21': 25, '14_to_8': 26, '14_to_13': 27, '14_to_20': 28, '16_to_17': 29, '16_to_18': 30, '16_to_19': 31, '17_to_12': 32, '17_to_16': 33, '17_to_25': 34, '17_to_37': 35, '18_to_16': 36, '18_to_19': 37, '19_to_16': 38, '19_to_18': 39, '19_to_20': 40, '20_to_14': 41, '20_to_19': 42, '20_to_23': 43, '21_to_13': 44, '21_to_17': 45, '21_to_20': 46, '21_to_23': 47, '23_to_20': 48, '23_to_21': 49, '23_to_25': 50, '23_to_26': 51, '24_to_26': 52, '25_to_17': 53, '25_to_23': 54, '25_to_27': 55, '26_to_23': 56, '26_to_24': 57, '26_to_28': 58, '27_to_25': 59, '27_to_28': 60, '27_to_29': 61, '28_to_26': 62, '28_to_27': 63, '28_to_30': 64, '29_to_27': 65, '29_to_30': 66, '29_to_36': 67, '30_to_28': 68, '30_to_29': 69, '30_to_31': 70, '30_to_35': 71, '31_to_30': 72, '31_to_32': 73, '31_to_34': 74, '32_to_31': 75, '32_to_40': 76, '34_to_31': 77, '34_to_35': 78, '34_to_41': 79, '35_to_30': 80, '35_to_34': 81, '35_to_38': 82, '36_to_29': 83, '36_to_37': 84, '37_to_12': 85, '37_to_17': 86, '37_to_36': 87, '37_to_38': 88, '38_to_35': 89, '38_to_37': 90, '38_to_41': 91, '40_to_32': 92, '40_to_41': 93, '40_to_42': 94, '41_to_10': 95, '41_to_34': 96, '41_to_38': 97, '41_to_40': 98, '41_to_45': 99, '42_to_40': 100, '42_to_43': 101, '43_to_42': 102, '43_to_44': 103, '43_to_46': 104, '44_to_43': 105, '45_to_37': 106, '45_to_41': 107, '45_to_46': 108, '46_to_43': 109, '46_to_45': 110, '46_to_47': 111, '47_to_46': 112, '47_to_48': 113, '47_to_49': 114, '48_to_47': 115, '48_to_49': 116, '49_to_47': 117, '49_to_48': 118}
import pandas as pd

# Read the CSV file
df = pd.read_csv(os.path.join(args.data, 'edges_id.csv'))

# Extract the 'road_segment' column as a list
edges = df['road_segment'].tolist()

# Split the IDs by '_', convert to integers, and sort
edges_sorted = sorted(edges, key=lambda x: [int(i) for i in x.split('_to_')])

# Create a dictionary with IDs as keys and indices as values
node_mapping = {edge: i for i, edge in enumerate(edges_sorted)}

print(node_mapping, len(node_mapping.keys()))