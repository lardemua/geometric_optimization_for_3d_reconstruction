#!/usr/bin/env python

# ----
# Compute the normals of the vertices of the point cloud.

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
ap.add_argument('arg1', metavar='In Filename',
                type=str, help='Name of the ply file to compute the normals of the vertices')

ap.add_argument('arg2', metavar='Out Filename',
                type=str, help='Return name of the ply file')

ap.add_argument("-r", metavar="Radius",
                help="Radius to compute de normals", required=False)

args = vars(ap.parse_args())

# args input
filename = args['arg1']
outFilename = args['arg2']

if args['r']:
    radius = args['r']
else:
    radius = '20'

#---------------------------------------
#--- Compute normals
#---------------------------------------

pcd_1 = '/tmp/pc.pcd'
pcd_2 = '/tmp/pc_normals.pcd'

bash('pcl_ply2pcd ' + filename + ' ' + pcd_1)
bash('pcl_normal_estimation ' + pcd_1 + ' ' + pcd_2 + ' -k ' + radius)
bash('pcl_pcd2ply ' + pcd_2 + ' ' + outFilename)
