r"""
============================================
Composite Module (:mod:`meshfree.composite`)
============================================

.. currentmodule:: meshfree.composite

The ``meshfree.composite`` module includes functions used to calculate
laminate properties based on input data of stacking sequence and lamina
properties.

The most convenient usage is probably using the
:func:`meshfree.composite.laminate.read_stack()` function::

    from meshfree.composite.laminate import read_stack

    laminaprop = (E11, E22, nu12, G12, G13, G23)
    plyt = ply_thickness
    stack = [0, 90, +45, -45]
    lam = read_stack(stack, plyt=plyt, laminaprop=laminaprop)

Where the laminate stiffness matrix, the often called ``ABD`` matrix, with
``shape=(6, 6)``, can be accessed using::

    >>> lam.ABD

and when shear stiffnesses are required, the ``ABDE`` matrix, with
``shape=(8, 8)``::

    >>> lam.ABDE

.. automodule:: meshfree.composite.laminate
    :members:

.. automodule:: meshfree.composite.lamina
    :members:

.. automodule:: meshfree.composite.matlamina
    :members:

"""
