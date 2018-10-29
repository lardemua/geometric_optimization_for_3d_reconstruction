#!/usr/bin/env python

# ----
# Change the position of the points of the point cloud basing on a new transformation.

import argparse
import subprocess

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
ap.add_argument('arg1', metavar='InFilename',
                type=str, help='Name of the ply file to compute the transformations')

ap.add_argument('arg2', metavar='OutFilename',
                type=str, help='Return name of the ply file')

ap.add_argument("-tm", metavar='Matrix',
                help="4x4 tranformation matrix: n1,n2,n3,t1,n4,n5,n6,t2,n7,n8,n9,t3,0,0,0,1", required=False)

args = vars(ap.parse_args())

# args input
filename = args['arg1']
outFilename = args['arg2']

if args['tm']:
    tranformationMatrix = args['tm']
else:
    tranformationMatrix = '1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1'

#---------------------------------------
#--- Transform PointClouds
#---------------------------------------

pcd_1 = '/tmp/pc.pcd'
pcd_2 = '/tmp/pc_trans.pcd'

bash('pcl_ply2pcd ' + filename + ' ' + pcd_1)
bash('pcl_transform_point_cloud ' + pcd_1 +
     ' ' + pcd_2 + ' ' + tranformationMatrix)

bash('pcl_pcd2ply ' + pcd_2 + ' ' + outFilename)
