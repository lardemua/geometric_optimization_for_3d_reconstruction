#!/usr/bin/env python

# ----
# Apply the transformations computed in the optimization process on the point clouds of the dataset optimized.
# Compute the normals of the point clouds of the two datasets.

import argparse
import subprocess
import numpy as np
import glob
import os
from numpy.linalg import inv

##
# @brief Executes the command in the shell in a blocking manner
#
# @param cmd a string with teh command to execute
#
# @return


def bash(cmd):
    print "Executing command: " + cmd
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print line,
        p.wait()


#---------------------------------------
#--- Argument parser
#---------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('directory', metavar='Directory',
                type=str, help='Directory of the dataset not optimized')

ap.add_argument("-skip", action='store_true',
                help="Pass the computation of the normals of the dataset not optimized", required=False)

args = vars(ap.parse_args())


# Work directory
Directory = args['directory']
DirectoryOptimized = os.path.dirname(Directory) + "/dataset_optimized"

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

if not args['skip']:
    for text in textfilenames:

        filename = text[:-3] + 'ply'

        print '\n---------------------------------------------\n'
        print 'The file ' + filename + ' is been processed...'
        print '\n---------------------------------------------\n'

        bash('./compNormalPly.py' + ' ' + filename + ' ' + filename)

        print '\n---------------------------------------------\n'
        print '------> The normals of the dataset not optimized were computed'
        print '\n---------------------------------------------\n'

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

    filename = text[:-3] + 'ply'

    print '\n---------------------------------------------\n'
    print 'The file ' + filename + ' is been processed...'
    print '\n---------------------------------------------\n'

    T = inv(Ti)

    Tstr = str(T[0, 0]) + ',' + str(T[0, 1]) + ',' + str(T[0, 2]) + ',' + str(T[0, 3]) + ',' + str(T[1, 0]) + ',' + str(T[1, 1]) + ',' + str(T[1, 2]) + ',' + str(T[1, 3]) + \
        ',' + str(T[2, 0]) + ',' + str(T[2, 1]) + ',' + str(T[2, 2]) + ',' + str(T[2, 3]) + \
        ',' + str(T[3, 0]) + ',' + str(T[3, 1]) + ',' + \
        str(T[3, 2]) + ',' + str(T[3, 3])

    bash('./compTransPly.py' + ' ' + filename +
         ' ' + filename + ' ' + '-tm ' + Tstr)

    T = To

    Tstr = str(T[0, 0]) + ',' + str(T[0, 1]) + ',' + str(T[0, 2]) + ',' + str(T[0, 3]) + ',' + str(T[1, 0]) + ',' + str(T[1, 1]) + ',' + str(T[1, 2]) + ',' + str(T[1, 3]) + \
        ',' + str(T[2, 0]) + ',' + str(T[2, 1]) + ',' + str(T[2, 2]) + ',' + str(T[2, 3]) + \
        ',' + str(T[3, 0]) + ',' + str(T[3, 1]) + ',' + \
        str(T[3, 2]) + ',' + str(T[3, 3])

    bash('./compTransPly.py' + ' ' + filename +
         ' ' + filename + ' ' + '-tm ' + Tstr)

    print '\n---------------------------------------------\n'
    print '------> The new transformation was applied'
    print '\n---------------------------------------------\n'

    bash('./compNormalPly.py' + ' ' + filename + ' ' + filename)

    print '\n---------------------------------------------\n'
    print '------> The normals of the dataset optimized were computed'
    print '\n---------------------------------------------\n'

print('---> Finish <---')
