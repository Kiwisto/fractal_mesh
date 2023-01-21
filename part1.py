# -*- coding: utf-8 -*-
"""
..warning:: Explanation of the code has been given during the lectures.
"""

# Python packages
import matplotlib.pyplot
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import random
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys


# MRG packages
import solutions


# Définition des fonctions

def add_node_to_mesh(node_coords, new_node_coords):

    # # On met à jour node_coords
    node_coords = numpy.append(node_coords, [new_node_coords], axis=0)
    return node_coords


def remove_node_to_mesh(p_elem2nodes, elem2nodes, node_coords, nodeid):

    # On retire le noeud de node_coords
    node_coords = numpy.delete(node_coords, nodeid, 0)

    copy_elem2nodes = numpy.copy(elem2nodes)
    copy_p_elem2nodes = numpy.copy(p_elem2nodes)
    # # On itère sur p_elem2nodes à la recherche des éléments qui contiennent le noeud à supprimer
    for k in range(len(p_elem2nodes)-1):
        a, b = p_elem2nodes[k], p_elem2nodes[k+1]
        for i in range(a, b):
            if elem2nodes[i] == nodeid:     # # On marque les éléments contenant les noeuds
                copy_elem2nodes[a: b] = -1.
                copy_p_elem2nodes[k] = -1
                sub = b-a
                copy_p_elem2nodes[k+1:] -= sub
                break

    L = []
    for el in copy_elem2nodes:     # # On met à jour elem2nodes en supprimant les éléments marqués
        if el != -1:
            if el > nodeid:
                el -= 1
            L.append(el)
    elem2nodes = numpy.array(L)

    L = []
    for el in copy_p_elem2nodes:    # # On met à jour p_elem2nodes en supprimant les éléments marqués
        if el >= 0:
            L.append(el)
    p_elem2nodes = numpy.array(L)

    return p_elem2nodes, elem2nodes, node_coords


def add_elem_to_mesh(p_elem2nodes, elem2nodes, elemid2nodes):

    for el in elemid2nodes:         # # On met à jour elem2nodes et p_elem2nodes
        elem2nodes = numpy.append(elem2nodes, el)
    p_elem2nodes = numpy.append(p_elem2nodes, len(elem2nodes))

    return p_elem2nodes, elem2nodes


def remove_elem_to_mesh(p_elem2nodes, elem2nodes, elemid):

    # On parcourt elem2nodes
    a, b = p_elem2nodes[elemid], p_elem2nodes[elemid+1]
    for loop in range(a, b):
        # On supprime les noeuds de elem2nodes
        elem2nodes = numpy.delete(elem2nodes, a, 0)

    sub = b-a
    p_elem2nodes[elemid:] -= sub
    # On décrémente p_elem2nodes
    p_elem2nodes = numpy.delete(p_elem2nodes, elemid, 0)

    return p_elem2nodes, elem2nodes


def compute_barycenter_of_element(p_elem2nodes, elem2nodes, node_coords, elemid, dim):

    # # On site les noeuds de l'élément
    nodes_to_compute = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid+1]]
    nodes_number = len(nodes_to_compute)
    nodes_to_compute_coords = numpy.zeros((nodes_number, dim))
    for i in range(nodes_number):
        nodes_to_compute_coords[i] = node_coords[nodes_to_compute[i]]

    barycenter_coords = []
    for i in range(dim):        # # On calcule les coordonnées du barycentre
        sum = 0
        for el in nodes_to_compute_coords:
            sum += el[i]
        barycenter_coords.append(sum*1./nodes_number)

    return barycenter_coords


def shift_quad_to_tri_all(p_elem2nodes, elem2nodes):

    def shift_quad_to_tri(p_elem2nodes, elem2nodes, elemid):
        # On situe les noeuds à traiter
        nodes_to_compute = elem2nodes[p_elem2nodes[elemid]
            : p_elem2nodes[elemid+1]]
        bottom_triangle = [nodes_to_compute[0],
                           nodes_to_compute[1], nodes_to_compute[2]]  # On forme le triangle du bas
        top_triangle = [nodes_to_compute[0],
                        nodes_to_compute[2], nodes_to_compute[3]]  # Idem en haut
        p_elem2nodes, elem2nodes = add_elem_to_mesh(
            p_elem2nodes, elem2nodes, bottom_triangle)
        p_elem2nodes, elem2nodes = add_elem_to_mesh(
            p_elem2nodes, elem2nodes, top_triangle)

        return p_elem2nodes, elem2nodes

    N = len(p_elem2nodes)       # On note le nombre d'élément carré à l'origine
    for i in range(N-1):       # Pour chaque élément, on crée les éléments triangle
        p_elem2nodes, elem2nodes = shift_quad_to_tri(
            p_elem2nodes, elem2nodes, i)

    # On supprime tous les éléments carrés (c'est à dire, les N premiers éléments)
    for loop in range(N-1):
        p_elem2nodes, elem2nodes = remove_elem_to_mesh(
            p_elem2nodes, elem2nodes, 0)

    return p_elem2nodes, elem2nodes


def aspect_ratio_tri(p_elem2nodes, elem2nodes, node_coords, elemid, dim):

    alpha = 3**(1./2)/6

    # On situe les noeuds de l'élément
    nodes_to_compute = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid+1]]
    nodes_number = len(nodes_to_compute)
    nodes_to_compute_coords = numpy.zeros((nodes_number, dim))
    for i in range(nodes_number):
        nodes_to_compute_coords[i] = node_coords[nodes_to_compute[i]]

    h = []
    hmax = 0
    for i in range(-1, 2):      # On calcule les longueurs du triangle
        h.append(((nodes_to_compute_coords[i][0]-nodes_to_compute_coords[i+1][0])**2+(
            nodes_to_compute_coords[i][1]-nodes_to_compute_coords[i+1][1])**2)**(1./2))
        if h[-1] > hmax:
            hmax = h[-1]

    # On calcule l'aspect ratio avec la formule du cours
    p = (h[0]+h[1]+h[2])*1./2
    S = (p*(p-h[0])*(p-h[1])*(p-h[2]))**(1./2)
    rho = S/p

    return 1./(alpha*hmax/rho)


def edge_length_factor_tri(p_elem2nodes, elem2nodes, node_coords, elemid, dim):

    # On situe les éléments à calculer
    nodes_to_compute = elem2nodes[p_elem2nodes[elemid]: p_elem2nodes[elemid+1]]
    nodes_number = len(nodes_to_compute)
    nodes_to_compute_coords = numpy.zeros((nodes_number, dim))
    for i in range(nodes_number):
        nodes_to_compute_coords[i] = node_coords[nodes_to_compute[i]]

    h = []      # On calcule les longueurs
    hmax = 0
    hmin = ((nodes_to_compute_coords[0][0]-nodes_to_compute_coords[1][0])**2+(
            nodes_to_compute_coords[0][1]-nodes_to_compute_coords[1][1])**2)**(1./2)
    hmed = 0
    for i in range(-1, 2):      # On situe hmax et hmin
        h.append(((nodes_to_compute_coords[i][0]-nodes_to_compute_coords[i+1][0])**2+(
            nodes_to_compute_coords[i][1]-nodes_to_compute_coords[i+1][1])**2)**(1./2))
        if h[-1] >= hmax:
            hmax = h[-1]
        if h[-1] <= hmin:
            hmin = h[-1]

    h.remove(hmax)  # On retire hmax et hmin pour n'avoir que hmed
    h.remove(hmin)
    hmed = h[0]

    return hmin*1./hmed     # On applique la formule


def shift_internal_nodes_coords(node_coords, xmin, xmax, ymin, ymax, nelemsx, nelemsy):

    # Cette fonction, simple, ne marche que sur la maille carré d'origine. Pour pouvoir le faire sur sur la maille fractale, il faut trouver
    # les bords, ce qui est bien plus compliqué

    for i in range(len(node_coords)):

        # On ne traite que les noeuds dont les coordonnées se trouvent entre 0. et 1., strictement
        for j in range(2):
            if node_coords[i][j] == 0. or node_coords[i][j] == 1.:
                break
        else:
            # On les translate aléatoirement, entre -1/3 et 1/3 de la longueur d'un élément carré
            for j in range(2):
                node_coords[i][j] += random.uniform(-(xmax-xmin)*1./(
                    3*(nelemsx-1)), (xmax-xmin)*1./(3*(nelemsx-1)))

    return node_coords


def remove_all_inside_rectangle(p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner):

    done = False

    while done == False:
        for i in range(len(p_elem2nodes)-1):
            barycenter = compute_barycenter_of_element(
                p_elem2nodes, elem2nodes, node_coords, i, 3)
            if barycenter[0] >= bottom_left_corner[0] and barycenter[0] <= top_right_corner[0] and barycenter[1] >= bottom_left_corner[1] and barycenter[1] <= top_right_corner[1]:
                p_elem2nodes, elem2nodes = remove_elem_to_mesh(
                    p_elem2nodes, elem2nodes, i)
                break
        else:
            done = True

    return p_elem2nodes, elem2nodes


def remove_orphan_nodes(p_elem2nodes, elem2nodes, node_coords):

    done = False
    while done == False:
        for node_id in range(len(node_coords)):
            for el in elem2nodes:
                if el == node_id:
                    break
            else:
                p_elem2nodes, elem2nodes, node_coords = remove_node_to_mesh(
                    p_elem2nodes, elem2nodes, node_coords, node_id)
                break
        else:
            done = True

    return p_elem2nodes, elem2nodes, node_coords


def generate_fractal_mesh(p_elem2nodes, elem2nodes, node_coords, xmin, xmax, ymin, ymax, nelemsx, nelemsy, ordre):

    # Il y a 4 "patterns" possibles, lorsqu'il faut bouger un carré et augmenter de 1 l'ordre sur un bord. J'ai donc codé 4 fonctions différentes, pour
    # chacun des cas. Elles sont très similaire, c'est seulement les coordonnées des noeuds à traiter et du prochain carré à déplacer qui changent entre
    # ces fonctions.
    # Je vais donc commenter qu'une seule d'entre elle, celle ci-dessous, qui fait la transformation du premier ordre sur le bord du haut.

    def move_square_b2r(p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner, lengthX, lengthY, NbrElem):

        p_elem2nodes, elem2nodes = remove_all_inside_rectangle(                               # # On retire le carré situé entre 1/4 et 1/2 du bord à traiter
            p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner)
        # # On rentre dans last_row les noeuds du bord situé entre 1/2 et 3/4 du bord à traiter, ce sont ceux sur lesquels le nouveau carré va être collé
        last_row = numpy.array([], dtype=int)
        for i in range(len(node_coords)):
            if node_coords[i][0] >= top_right_corner[0] and node_coords[i][0] <= top_right_corner[0]+lengthX and node_coords[i][1] == top_right_corner[1]:
                last_row = numpy.append(last_row, i)

        current_row = []
        # # On itère sur le nombre de lignes du carré à créer
        for i in range(NbrElem):
            last_node = last_row[0]
            x = node_coords[last_node][0]
            y = node_coords[last_node][1] + lengthY*1./NbrElem
            z = 0.0
            # # On crée le premier noeud, juste au dessus du premier noeud de last_row
            node_coords = add_node_to_mesh(node_coords, [x, y, z])
            current_row.append(len(node_coords)-1)
            # # On itère sur chaque colonne, à partir du second noeud
            for j in range(1, NbrElem+1):
                last_node = last_row[j]
                x = node_coords[last_node][0]
                y = node_coords[last_node][1] + lengthY*1./NbrElem
                z = 0.0
                # # On crée chaque noeud au dessus du noeud de last_row de même rang (situé juste en dessous)
                node_coords = add_node_to_mesh(node_coords, [x, y, z])
                current_row.append(len(node_coords)-1)
                p_elem2nodes, elem2nodes = add_elem_to_mesh(p_elem2nodes, elem2nodes, [            # # On crée un élément carré avec les deux points précédents
                                                            last_row[j-1], last_row[j], current_row[j], current_row[j-1]])
            # #last_row devient current_row
            last_row = list.copy(current_row)
            current_row = []

        next_bottom_left_corner = [         # # On définit les coordonnées du prochain carré à retirer à partir de celui qu'on vient de retirer
            bottom_left_corner[0]+lengthX, bottom_left_corner[1]+lengthY*1./2]
        next_top_right_corner = [
            top_right_corner[0]+lengthX*1./2, top_right_corner[1]]

        return p_elem2nodes, elem2nodes, node_coords, next_bottom_left_corner, next_top_right_corner

    def move_square_r2t(p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner, lengthX, lengthY, NbrElem):

        p_elem2nodes, elem2nodes = remove_all_inside_rectangle(
            p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner)

        last_row = numpy.array([], dtype=int)
        for i in range(len(node_coords)):
            if node_coords[i][1] >= top_right_corner[1] and node_coords[i][1] <= top_right_corner[1]+lengthY and node_coords[i][0] == bottom_left_corner[0]:
                last_row = numpy.append(last_row, i)

        current_row = []
        for i in range(NbrElem):
            last_node = last_row[0]
            x = node_coords[last_node][0] - lengthX*1./NbrElem
            y = node_coords[last_node][1]
            z = 0.0
            node_coords = add_node_to_mesh(node_coords, [x, y, z])
            current_row.append(len(node_coords)-1)
            for j in range(1, NbrElem+1):
                last_node = last_row[j]
                x = node_coords[last_node][0] - lengthX*1./NbrElem
                y = node_coords[last_node][1]
                z = 0.0
                node_coords = add_node_to_mesh(node_coords, [x, y, z])
                current_row.append(len(node_coords)-1)
                p_elem2nodes, elem2nodes = add_elem_to_mesh(p_elem2nodes, elem2nodes, [
                                                            current_row[j-1], last_row[j-1], last_row[j], current_row[j]])
            last_row = list.copy(current_row)
            current_row = []

        next_bottom_left_corner = [
            bottom_left_corner[0], bottom_left_corner[1]+lengthY]
        next_top_right_corner = [
            top_right_corner[0]-lengthX*1./2, top_right_corner[1]+lengthY*1./2]

        return p_elem2nodes, elem2nodes, node_coords, next_bottom_left_corner, next_top_right_corner

    def move_square_t2l(p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner, lengthX, lengthY, NbrElem):

        p_elem2nodes, elem2nodes = remove_all_inside_rectangle(
            p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner)

        last_row = numpy.array([], dtype=int)
        for i in range(len(node_coords)):
            if node_coords[i][0] >= bottom_left_corner[0]-lengthX and node_coords[i][0] <= bottom_left_corner[0] and node_coords[i][1] == bottom_left_corner[1]:
                last_row = numpy.append(last_row, i)
        last_row = last_row[::-1]

        current_row = []
        for i in range(NbrElem):
            last_node = last_row[0]
            x = node_coords[last_node][0]
            y = node_coords[last_node][1] - lengthY*1./NbrElem
            z = 0.0
            node_coords = add_node_to_mesh(node_coords, [x, y, z])
            current_row.append(len(node_coords)-1)
            for j in range(1, NbrElem+1):
                last_node = last_row[j]
                x = node_coords[last_node][0]
                y = node_coords[last_node][1] - lengthY*1./NbrElem
                z = 0.0
                node_coords = add_node_to_mesh(node_coords, [x, y, z])
                current_row.append(len(node_coords)-1)
                p_elem2nodes, elem2nodes = add_elem_to_mesh(p_elem2nodes, elem2nodes, [
                                                            last_row[j-1], current_row[j-1], current_row[j], last_row[j]])
            last_row = list.copy(current_row)
            current_row = []

        next_bottom_left_corner = [
            bottom_left_corner[0]-lengthX*1./2, bottom_left_corner[1]]
        next_top_right_corner = [
            bottom_left_corner[0], bottom_left_corner[1]+lengthY*1./2]

        return p_elem2nodes, elem2nodes, node_coords, next_bottom_left_corner, next_top_right_corner

    def move_square_l2b(p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner, lengthX, lengthY, NbrElem):

        p_elem2nodes, elem2nodes = remove_all_inside_rectangle(
            p_elem2nodes, elem2nodes, node_coords, bottom_left_corner, top_right_corner)

        last_row = numpy.array([], dtype=int)
        for i in range(len(node_coords)):
            if node_coords[i][1] >= bottom_left_corner[1]-lengthY and node_coords[i][1] <= bottom_left_corner[1] and node_coords[i][0] == top_right_corner[0]:
                last_row = numpy.append(last_row, i)
        last_row = last_row[::-1]

        current_row = []
        for i in range(NbrElem):
            last_node = last_row[0]
            x = node_coords[last_node][0] + lengthX*1./NbrElem
            y = node_coords[last_node][1]
            z = 0.0
            node_coords = add_node_to_mesh(node_coords, [x, y, z])
            current_row.append(len(node_coords)-1)
            for j in range(1, NbrElem+1):
                last_node = last_row[j]
                x = node_coords[last_node][0] + lengthX*1./NbrElem
                y = node_coords[last_node][1]
                z = 0.0
                node_coords = add_node_to_mesh(node_coords, [x, y, z])
                current_row.append(len(node_coords)-1)
                p_elem2nodes, elem2nodes = add_elem_to_mesh(p_elem2nodes, elem2nodes, [
                                                            last_row[j-1], current_row[j-1], current_row[j], last_row[j]])
            last_row = list.copy(current_row)
            current_row = []

        next_bottom_left_corner = [
            bottom_left_corner[0]+lengthX*1./2, bottom_left_corner[1]-lengthY*1./2]
        next_top_right_corner = [
            bottom_left_corner[0]+lengthX, bottom_left_corner[1]]

        return p_elem2nodes, elem2nodes, node_coords, next_bottom_left_corner, next_top_right_corner

    NbrElem = nelemsx//4
    lengthX = (xmax-xmin)*1./4
    lengthY = (ymax-ymin)*1./4

    # # On suit l'état de chaque bord du carré en mettant à jour la prochaine opération à faire, et les coordonnées du prochain carré à retirer

    top_side = {
        "next_operation": 0,
        "bottom_left_corner": [1./4, 3./4],
        "top_right_corner": [1./2, 1.0]
    }

    left_side = {
        "next_operation": 1,
        "bottom_left_corner": [0., 1./4],
        "top_right_corner": [1./4, 1./2]
    }

    bottom_side = {
        "next_operation": 2,
        "bottom_left_corner": [1./2, 0.],
        "top_right_corner": [3./4, 1./4]
    }

    right_side = {
        "next_operation": 3,
        "bottom_left_corner": [3./4, 1./2],
        "top_right_corner": [1., 3./4]
    }

    sides = [top_side, left_side, right_side, bottom_side]

    # On boucle sur chaque bord en incrémentant à chaque fois la prochaine opération à faire, et en mettant à jour le prochain carré à traiter

    for loop in range(ordre):
        for side in sides:
            if side["next_operation"] == 0:
                p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"] = move_square_b2r(
                    p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"], lengthX, lengthY, NbrElem)
                side["next_operation"] += 1
            elif side["next_operation"] == 1:
                p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"] = move_square_r2t(
                    p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"], lengthX, lengthY, NbrElem)
                side["next_operation"] += 1
            elif side["next_operation"] == 2:
                p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"] = move_square_t2l(
                    p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"], lengthX, lengthY, NbrElem)
                side["next_operation"] += 1
            elif side["next_operation"] == 3:
                p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"] = move_square_l2b(
                    p_elem2nodes, elem2nodes, node_coords, side["bottom_left_corner"], side["top_right_corner"], lengthX, lengthY, NbrElem)
                side["next_operation"] = 0

        NbrElem = NbrElem//2    # On met à jour la taille du carré à retirer
        lengthX, lengthY = right_side["top_right_corner"][0] - \
            right_side["bottom_left_corner"][0], right_side["top_right_corner"][1] - \
            right_side["bottom_left_corner"][1]

    return p_elem2nodes, elem2nodes, node_coords


def run_mesh_generation():

    # # paramètres de maillage
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 64, 64
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)

    # # génération maille carrée
    p_elem2nodes, elem2nodes, node_coords, node_l2g = solutions._set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # # zone d'essai

    p_elem2nodes, elem2nodes, node_coords = generate_fractal_mesh(
        p_elem2nodes, elem2nodes, node_coords, xmin, xmax, ymin, ymax, nelemsx, nelemsy, 5)

    # p_elem2nodes, elem2nodes, node_coords = remove_orphan_nodes(
    #     p_elem2nodes, elem2nodes, node_coords)

    p_elem2nodes, elem2nodes = shift_quad_to_tri_all(p_elem2nodes, elem2nodes)

    # node_coords = shift_internal_nodes_coords(
    #     node_coords, xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # # plot mesh
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes,
                         node_coords, color='yellow')

    # node = 128
    # solutions._plot_node(p_elem2nodes, elem2nodes,
    #                      node_coords, node, color='red', marker='o')

    # elem = 45
    # solutions._plot_elem(p_elem2nodes, elem2nodes,
    #                      node_coords, elem, color='orange')

    matplotlib.pyplot.show()

    return p_elem2nodes, elem2nodes, node_coords


if __name__ == '__main__':

    p_elem2nodes, elem2nodes, node_coords = run_mesh_generation()

    print('End.')
