#!/usr/bin/env python
import openmesh as om
import numpy as np
import glob
import os
from numpy.linalg import inv
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('directory', metavar='Directory',
                type=str, help='Put the directory of the dataset not optimized')

args = vars(ap.parse_args())

Directory = args['directory']
DirectoryOptimized = Directory + 'Optimized'

Transformations = []
TransformationsOptimized = []

# Read initial data
textfilenames = sorted(glob.glob((os.path.join(Directory, '00*.txt'))))
for w, cltx in enumerate(textfilenames):
    nt = len(cltx)
    if cltx[nt-6] == "-" or cltx[nt-7] == "-" or cltx[nt-8] == "-":
        del textfilenames[w]

for j, namefile in enumerate(textfilenames):
    T = np.zeros((4, 4))
    txtfile = open(namefile)
    for i, line in enumerate(txtfile):
        if 0 < i < 5:
            paramVect = []
            for param in line.strip().split(' '):
                paramVect.append(param)

            T[i-1][0:] = np.array(paramVect)

    Transformations.append(T.transpose())
    txtfile.close()

# Read optimized data
textfilenames = sorted(
    glob.glob((os.path.join(DirectoryOptimized, '00*.txt'))))
for w, cltx in enumerate(textfilenames):
    nt = len(cltx)
    if cltx[nt-6] == "-" or cltx[nt-7] == "-" or cltx[nt-8] == "-":
        del textfilenames[w]

for j, namefile in enumerate(textfilenames):
    T = np.zeros((4, 4))
    txtfile = open(namefile)
    for i, line in enumerate(txtfile):
        if 0 < i < 5:
            paramVect = []
            for param in line.strip().split(' '):
                paramVect.append(param)

            T[i-1][0:] = np.array(paramVect)

    TransformationsOptimized.append(T.transpose())
    txtfile.close()

for text, Ti, To in zip(textfilenames, Transformations, TransformationsOptimized):

    filename = text[:len(text)-3] + 'ply'

    mesh = om.read_trimesh(filename=filename,
                           binary=True,
                           msb=True,
                           lsb=True,
                           swap=True,
                           vertex_normal=True,
                           vertex_color=False,
                           vertex_tex_coord=False,
                           halfedge_tex_coord=False,
                           edge_color=False,
                           face_normal=False,
                           face_color=False,
                           color_alpha=True,
                           color_float=True)

    print('mesh has_vertex_normals ' + str(mesh.has_vertex_normals()))
    # exit(0)
    print '-----------------------------------'
    print 'Solve for file' + filename + ' ...'

    print('\nRead a point cloud with ' + str(mesh.n_vertices()) + ' points\n')

    point_array = np.transpose(mesh.points())
    # print(point_array.shape)
    # print(point_array)

    # TODO check time this loope takes. Use matricial multiplication to compute very fast
    for p, vh in zip(mesh.points(), mesh.vertices()):

        newPt = np.zeros((3))

        # Move point to map
        T = inv(Ti)
        rot = np.matrix(T[0: 3, 0: 3])
        Pt = rot.dot(p) + T[0: 3, 3]
        Pt = [Pt[0, 0], Pt[0, 1], Pt[0, 2]]
        # Move point to camera
        T = To
        rot = np.matrix(T[0: 3, 0: 3])
        Pto = rot.dot(Pt) + T[0: 3, 3]
        newPt = [Pto[0, 0], Pto[0, 1], Pto[0, 2]]

        mesh.set_point(vh, newPt)

    # compute normals
    for vh in mesh.vertices():
        normal = mesh.calc_vertex_normal(vh)
        print(normal)
        mesh.set_normal(vh, normal)

    # mesh.request_vertex_normals()
    # mesh.update_vertex_normals()
    # mesh.update_normals()
    om.write_mesh(filename=filename,
                  mesh=mesh,
                  binary=False,
                  msb=False,
                  lsb=False,
                  swap=False,
                  vertex_normal=True,
                  vertex_color=False,
                  vertex_tex_coord=False,
                  halfedge_tex_coord=False,
                  edge_color=False,
                  face_normal=False,
                  face_color=False,
                  color_alpha=False,
                  color_float=False
                  )
    break
print('Finish')
