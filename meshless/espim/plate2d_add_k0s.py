from meshless.espim.plate2d_add_k0s_cell_based import add_k0s as add_k0s_cell
from meshless.espim.plate2d_add_k0s_cell_based_no_smoothing import add_k0s as add_k0s_cell_no_smoothing
from meshless.espim.plate2d_add_k0s_edge_based import add_k0s as add_k0s_edge

def add_k0s(k0, mesh, prop_from_node, method='cell-based'):
    if method == 'cell-based':
        return add_k0s_cell(k0, mesh, prop_from_node)
    elif method == 'cell-based-no-smoothing':
        return add_k0s_cell_no_smoothing(k0, mesh, prop_from_node)
    elif method == 'edge-based':
        return add_k0s_edge(k0, mesh, prop_from_node)
    else:
        raise ValueError('Invalid method')



