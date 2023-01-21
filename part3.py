# -*- coding: utf-8 -*-
"""
..warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.
"""

# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
from numpy.lib.function_base import percentile
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys


# MRG packages
import zsolutions4students as solutions
import part1 as generation


# ..todo: Uncomment for displaying limited digits
# numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def run_exercise_solution_helmholtz_dddd():

    # -- set equation parameters
    wavenumber = numpy.pi*1.7

    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 20, 20

    # -- generate mesh

    p_elem2nodes, elem2nodes, node_coords = numpy.loadtxt(
        "p_elem2nodes.dat", dtype="int"), numpy.loadtxt("elem2nodes.dat", dtype="int"), numpy.loadtxt("node_coords.dat")

    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')

    # -- set boundary geometry

    nodes_on_boundary = numpy.array([], dtype=int)
    for i in range(len(node_coords)):
        # On ne prend que les noeuds situés sur le bord de gauche, où il n'y a pas de fractale.
        if node_coords[i][0] == 0. and node_coords[i][1] <= 1. and node_coords[i][1] >= 0.:
            nodes_on_boundary = numpy.append(nodes_on_boundary, i)

    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0., 1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(
            0., 1.)*wavenumber*complex(0., 1.)*wavenumber*solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros(
        (nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = 1.

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - \
            (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(
        p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(
        nodes_on_boundary, values_at_nodes_on_boundary, A, B)

    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
    _ = solutions._plot_contourf(
        nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))

    # ..warning: for teaching purpose only
    # -- plot exact solution
    # solexactreal = solexact.reshape((solexact.shape[0], ))
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
    # _ = solutions._plot_contourf(
    #     nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
    # ..warning: end

    return


if __name__ == '__main__':

    run_exercise_solution_helmholtz_dddd()

    print('End.')
