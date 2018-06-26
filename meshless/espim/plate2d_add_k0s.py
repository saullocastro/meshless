from meshless.espim.plate2d_add_k0s_cell_based import add_k0s as add_k0s_cell
from meshless.espim.plate2d_add_k0s_cell_based_no_smoothing import add_k0s as add_k0s_cell_no_smoothing
from meshless.espim.plate2d_add_k0s_edge_based import add_k0s as add_k0s_edge

def add_k0s(k0, mesh, prop_from_node, method='cell-based', alpha=0.2):
    #alpha between 0. and 0.6, according to studies of Lyly et al.
    if method == 'cell-based':
        return add_k0s_cell(k0, mesh, prop_from_node, alpha=alpha)
    elif method == 'cell-based-no-smoothing':
        return add_k0s_cell_no_smoothing(k0, mesh, prop_from_node, alpha=alpha)
    elif method == 'edge-based':
        return add_k0s_edge(k0, mesh, prop_from_node, alpha=alpha)
    else:
        raise ValueError('Invalid method')



