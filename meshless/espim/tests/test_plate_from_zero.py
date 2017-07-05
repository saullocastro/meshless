def test_plate_from_zero():
    # Plate geometry and laminate data

    a = 0.406
    b = 0.254
    E1 = 1.295e11
    E2 = 9.37e9
    nu12 = 0.38
    G12 = 5.24e9
    G13 = 5.24e9
    G23 = 5.24e9
    plyt = 1.9e-4
    laminaprop = (E1, E2, nu12, G12, G13, G23)

    angles = [0, 45, -45, 90, 90, -45, 45, 0]

    # Generating Mesh
    # ---

    import numpy as np
    from scipy.spatial import Delaunay

    xs = np.linspace(0, a, 8)
    ys = np.linspace(0, b, 8)
    points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    tri = Delaunay(points)

    # Using Meshless Package
    # ---

    from scipy.sparse import coo_matrix

    from meshless.composite.laminate import read_stack
    from meshless.sparse import solve
    from meshless.linear_buckling import lb
    from meshless.espim.read_mesh import read_delaunay
    from meshless.espim.plate2d_calc_k0 import calc_k0
    from meshless.espim.plate2d_calc_kG import calc_kG
    from meshless.espim.plate2d_add_k0s import add_k0s

    mesh = read_delaunay(points, tri)
    nodes = np.array(list(mesh.nodes.values()))
    prop_from_nodes = True

    nodes_xyz = np.array([n.xyz for n in nodes])

    # **Applying properties

    # applying heterogeneous properties
    for node in nodes:
        lam = read_stack(angles, plyt=plyt, laminaprop=laminaprop)
        node.prop = lam

    # **Defining Boundary Conditions**
    #

    DOF = 5
    def bc(K, mesh):
        for node in nodes[nodes_xyz[:, 0] == xs.min()]:
            for dof in [1, 3]:
                j = dof-1
                K[node.index*DOF+j, :] = 0
                K[:, node.index*DOF+j] = 0
        for node in nodes[(nodes_xyz[:, 1] == ys.min()) |
                          (nodes_xyz[:, 1] == ys.max())]:
            for dof in [2, 3]:
                j = dof-1
                K[node.index*DOF+j, :] = 0
                K[:, node.index*DOF+j] = 0
        for node in nodes[nodes_xyz[:, 0] == xs.max()]:
            for dof in [3]:
                j = dof-1
                K[node.index*DOF+j, :] = 0
                K[:, node.index*DOF+j] = 0

    # **Calculating Constitutive Stiffness Matrix**

    k0s_method = 'cell-based'
    k0 = calc_k0(mesh, prop_from_nodes)
    add_k0s(k0, mesh, prop_from_nodes, k0s_method)
    bc(k0, mesh)
    k0 = coo_matrix(k0)

    # **Defining Load and External Force Vector**

    def define_loads(mesh):
        loads = []
        load_nodes = nodes[(nodes_xyz[:, 0] == xs.max()) &
                           (nodes_xyz[:, 1] != ys.min()) &
                           (nodes_xyz[:, 1] != ys.max())]
        fx = -1. / (nodes[nodes_xyz[:, 0] == xs.max()].shape[0] - 1)
        for node in load_nodes:
            loads.append([node, (fx, 0, 0)])
        load_nodes = nodes[(nodes_xyz[:, 0] == xs.max()) &
                           ((nodes_xyz[:, 1] == ys.min()) |
                            (nodes_xyz[:, 1] == ys.max()))]
        fx = -1. / (nodes[nodes_xyz[:, 0] == xs.max()].shape[0] - 1) / 2
        for node in load_nodes:
            loads.append([node, (fx, 0, 0)])
        return loads

    n = k0.shape[0] // DOF
    fext = np.zeros(n*DOF, dtype=np.float64)
    loads = define_loads(mesh)
    for node, force_xyz in loads:
        fext[node.index*DOF + 0] = force_xyz[0]
    print('Checking sum of forces: %s' % str(fext.reshape(-1, DOF).sum(axis=0)))

    # **Running Static Analysis**

    d = solve(k0, fext, silent=True)
    total_trans = (d[0::DOF]**2 + d[1::DOF]**2)**0.5
    print('Max total translation', total_trans.max())

    # **Calculating Geometric Stiffness Matrix**

    kG = calc_kG(d, mesh, prop_from_nodes)
    bc(kG, mesh)
    kG = coo_matrix(kG)

    # **Running Linear Buckling Analysis**

    eigvals, eigvecs = lb(k0, kG, silent=True)
    print('First 5 eigenvalues')
    print('\n'.join(map(str, eigvals[:5])))

    assert np.allclose(eigvals[:5], [
        1357.88861842,
        3024.52229631,
        4831.59778558,
        6687.33417597,
        7885.32591511,
        ]
            )


if __name__ == '__main__':
    test_plate_from_zero()
