import numpy as np
import pickle as pkl
import pdb

out_edge = np.expand_dims(np.arange(0, 62), 0)
in_edge = np.expand_dims(np.arange(1, 63), 0)
edge1 = np.concatenate([np.concatenate([out_edge, in_edge], axis=0), np.concatenate([in_edge, out_edge], axis=0)], axis=1)

out_edge = np.expand_dims(np.arange(0, 61), 0)
in_edge = np.expand_dims(np.arange(2, 63), 0)
edge2 = np.concatenate([np.concatenate([out_edge, in_edge], axis=0), np.concatenate([in_edge, out_edge], axis=0)], axis=1)

out_edge = np.expand_dims(np.arange(0, 62), 0)
in_edge = np.expand_dims(np.arange(61, -1, -1), 0)
out_edge[0, 31:] += 1
in_edge[0, :31] += 1
edge3 = np.concatenate([np.concatenate([out_edge, in_edge], axis=0), np.concatenate([in_edge, out_edge], axis=0)], axis=1)

out_edge = np.expand_dims(np.arange(0, 63), 0)
in_edge = np.expand_dims(np.arange(0, 63), 0)
edge4 = np.concatenate([out_edge, in_edge], axis=0)

# pdb.set_trace()
edge_list = [edge1, edge2, edge3, edge4]

edge = np.concatenate(edge_list, axis=1)

pkl.dump(edge, open('edge.pkl', 'wb'))
