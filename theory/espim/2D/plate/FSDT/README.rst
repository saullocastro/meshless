ES-PIM with Discrete Shear Gap (DSG), Implementation #1
=======================================================

References
----------

Membrane and Bending strains::

    ES-PIM as per G. R. Liu. Meshfree Methods, 2nd Edition. CRC Press, 2009

Transverse shear strains::

    Discrete Shear Gap (DSG) using three smoothing triangles within each tria,
    as per P. Phung-Van et. al. Static and free vibration analyses and dynamic
    control of composite plates integrated with piezoelectric sensors and
    actuators by the cell-based smoothed discrete shear gap method. Smart
    Materials and Structures, Vol. 22, 2013. 


Theories implemented
--------------------

Strain components::

    exx = u,x              # membrane
    eyy = v,y              # membrane
    gxy = u,y + v,x        # membrane
    kxx = phix,x           # bending
    kyy = phiy,y           # bending
    kxy = phix,y + phiy,x  # bending
    gxz = w,x + phix       # transverse shear
    gyz = w,y + phiy       # transverse shear

DOFs per node::

    u, v, w, phix, phiy

For strain smoothing, after applying divergence theorem::
    cexx = u               # membrane
    ceyy = v               # membrane
    cgxy = u + v           # membrane
    ckxx = phix            # bending
    ckyy = phiy            # bending
    ckxy = phix + phiy     # bending

Transverse shear is treated differently, by discrete shear gap (DSG)::
    cgxz = w,x + phix      # membrane transverse shear
    cgyz = w,y + phiy      # membrane transverse shear


Within an edge domain there are four nodes, the displacement field for a given
integration boundary ``i = [1, 2, 3, 4]``::

    ui = u1*fi1 + u2*fi2 + u3*fi3 + u4*fi4
    vi = v1*fi1 + v2*fi2 + v3*fi3 + v4*fi4
    wi = w1*fi1 + w2*fi2 + w3*fi3 + w4*fi4
    phixi = phix1*fi1 + phix2*fi2 + phix3*fi3 + phix4*fi4
    phiyi = phiy1*fi1 + phiy2*fi2 + phiy3*fi3 + phiy4*fi4


