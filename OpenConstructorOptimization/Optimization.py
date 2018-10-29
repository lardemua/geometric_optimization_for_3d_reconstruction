#!/usr/bin/env python
"""Calibration Optimization Camera-Camera

Program whose function is to optimize the calibration of n cameras at the same time.
"""

from myClasses import *
from costFunctions import *

import os  # Using operating system dependent functionality
import cv2  # OpenCV library
import cv2.aruco as aruco  # Aruco Markers
import subprocess
import glob  # Finds all the pathnames
import random
import time  # Estimate time of process
import argparse  # Read command line arguments
import numpy as np  # Arrays and opencv images
import matplotlib.pyplot as plt  # Library to do plots 2D
import networkx as nx
import matplotlib.animation as animation  # Animate plots
import pickle
import sys


from tqdm import tqdm  # Show a smart progress meter
from scipy.sparse import lil_matrix  # Lib Sparse bundle adjustment
from scipy.optimize import least_squares  # Lib optimization
from numpy.linalg import inv
from transformations import quaternion_from_matrix
from transformations import quaternion_matrix
from transformations import random_quaternion
from transformations import quaternion_slerp
from matplotlib import gridspec  # Make subplott
from mpl_toolkits.mplot3d import Axes3D  # Library to do plots 3D
from itertools import combinations


##
# @brief Executes the command in the shell in a blocking manner
#
# @param cmd a string with teh command to execute
#
# @return
def bash(cmd):

    print("Executing command: " + cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print
        line,
        p.wait()


def create_dir(inputDir, newDir):

    if not os.path.exists(newDir):
        # Create new directory
        os.makedirs(newDir)

    bash('cp ' + inputDir + '/* ' + newDir + '/')

    print("----------------------------\nNew directory was created\n----------------------------")


def get_cam_trans(X, textFileNames):  # Get camera transformations by OpenConstructor

    # HACK???
    for w, cltx in enumerate(textFileNames):
        nt = len(cltx)
        if cltx[nt - 6] == "-" or cltx[nt - 7] == "-" or cltx[nt - 8] == "-":
            del textFileNames[w]

    print("\n" + "--------------------------------------------------------")
    print("Get initial extrinsic parameters of the cameras...\n")

    for i, textFileName in enumerate(textFileNames):

        print textFileName
        print "\n"

        Tot = np.zeros((4, 4))
        textFile = open(textFileName)

        for j, line in enumerate(textFile):
            if 0 < j < 5:
                paramVect = []
                for param in line.strip().split(' '):
                    paramVect.append(param)

                Tot[j - 1][0:] = np.array(paramVect)

        Tt = Tot.transpose()

        # T cameras from OpenContructor
        camera = MyCamera(T=Tt, id=str(i))

        X.cameras.append(camera)
        # --------
        textFile.close()
        print("Initial Extrinsic matrix of camera " + str(i) + "...\n")
        print("T = \n" + str(Tt))
        print("\n--------------------------------------------------------")

    return X


def request_map_node_input(GA):

    map_node = 'A0'  # to be defined by hand

    while map_node not in GA.nodes:
        print('Must define a map that exists in the graph. \nShould be one of ' + str(GA.nodes))
        name = raw_input("Insert a valid map node: ")
        map_node = str(name)
        print("\n")

    print("-> Map node is " + map_node)
    print("-----------------------\n")

    return map_node


def draw(raw, corner, size_square):

    xcm = (corner[0][0, 0] + corner[0][1, 0] + corner[0][2, 0] + corner[0][3, 0]) / 4
    ycm = (corner[0][0, 1] + corner[0][1, 1] + corner[0][2, 1] + corner[0][3, 1]) / 4
    x1 = int(xcm - size_square)
    y1 = int(ycm - size_square)
    x2 = int(xcm + size_square)
    y2 = int(ycm + size_square)
    cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 0, 255), 2)


def draw_initial_guess(X, Pc, detections, height, width):

    # Size of the square drawn in the image
    size_square = 6

    for detection in detections:

        # draw(raw, corner, size_square)
        print(detection.corner)

        # fetch the camera for this detection
        camera = [camera for camera in X.cameras if camera.id == detection.camera[1:]][0]

        # fetch the ArUco for this detection
        aruco = [aruco for aruco in X.arucos if aruco.id == detection.aruco[1:]][0]

        Ta = aruco.getT()
        Tc = camera.getT()
        T = np.matmul(inv(Tc), Ta)

        ##
        print(Pc)
        xypix = points2imageFromT(T, X.intrinsics, Pc, X.distortion)

        k = int(camera.id)

        x = int(xypix[0][0])
        y = int(xypix[0][1])

        arucoCenterX = int(detection.corner[0][0][0] + (detection.corner[0][1][0] - detection.corner[0][0][0]) / 2)
        arucoCenterY = int(detection.corner[0][0][1] + (detection.corner[0][2][1] - detection.corner[0][0][1]) / 2)

        cv2.line(s[k].raw, (x,y), (arucoCenterX, arucoCenterY), (255, 255, 0), 2)

        # Draw initial projections
        if 0 < xypix[0][0] < height and 0 < xypix[0][1] < width:
            x1 = int(xypix[0][0] - size_square)
            y1 = int(xypix[0][1] - size_square)
            x2 = int(xypix[0][0] + size_square)
            y2 = int(xypix[0][1] + size_square)
            cv2.rectangle(s[k].raw, (x1, y1), (x2, y2), (255, 128, 0), -1)

        # print()

        cv2.namedWindow('camera' + str(k), flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow('camera' + str(k), width=1200, height=700)
        cv2.moveWindow('camera' + str(k), 0, 0)
        cv2.imshow('camera' + str(k), s[k].raw)

    # Draw projection 3D
    plt.ion()
    fig3 = plt.figure()
    ax3D = fig3.add_subplot(111, projection='3d')
    plt.hold(True)
    # plt.title("3D projection of ArUco markers")
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    ax3D.set_aspect('equal')
    ax3D.set_xticklabels([])
    ax3D.set_yticklabels([])
    ax3D.set_zticklabels([])

    X.plot3D(ax3D, 'k.', Pc)

    fig3.show()
    plt.waitforbuttonpress(0.1)


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------------------
    # --- Argument parser
    # ---------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument('dir', metavar='Directory',
                    help='Directory of the dataset to optimize')

    ap.add_argument('option', choices=['fromAruco', 'fromFile'],
                    help="Get initial estimation from the markers detection or from a file.")

    ap.add_argument('option2', choices=['all', 'translation'],
                    help="Choose translation and rotation or translation only in the vector x to optimize.")

    ap.add_argument("-d", action='store_true',
                    help="Draw initial estimation", required=False)

    ap.add_argument("-do", action='store_true',
                    help="Draw during optimization", required=False)

    ap.add_argument("-saveResults", action='store_true',
                    help="Save the results of the optimization", required=False)

    ap.add_argument("-processDataset", action='store_true',
                    help="Process the point clouds with the results obtained from the optimization process", required=False)

    ap.add_argument("-ms", metavar="markerSize",
                    help="Size of the AruCo markers (m)", required=False)

    args = vars(ap.parse_args())

    if args['ms']:
        markerSize = float(args['ms'])
    else:
        markerSize = 0.082

    # ---------------------------------------
    # --- Initialization
    # ---------------------------------------

    # Initialize X structure
    X = MyX()

    # Initialize the ArUco nodes graph
    GA = nx.Graph()
    GA.add_edge('Map', 'C0', weight=1)

    newDir = os.path.dirname(args['dir']) + "/dataset_optimized"

    # Get all image file names (each image corresponds to a camera)
    # TODO image format .jpg vs .png
    fileNames = sorted(glob.glob((os.path.join(args['dir'], '*.jpg'))))
    if fileNames == []:
        fileNames = sorted(glob.glob((os.path.join(args['dir'], '*.png'))))

    if args['option'] == 'fromFile':

        if args['saveResults']:
            create_dir(args['dir'], newDir)

        # Read data calibration camera (Dictionary elements -> "mtx", "dist")
        tmpParamList = []
        text_file = open("CameraParameters/Mycalibration.txt", "r")

        for value in text_file.read().split(', '):
            value = value.replace("\n", "")
            tmpParamList.append(value)
        paramsNP = np.array(tmpParamList)
        text_file.close()

        # Intrinsic matrix and distortion vector
        mtx = np.zeros((3, 3))
        mtx[0][0:3] = paramsNP[0:3]
        mtx[1][0:3] = paramsNP[3:6]
        mtx[2][0:3] = paramsNP[6:9]
        dist = np.zeros((1, 5))
        dist[:] = paramsNP[9:14]

        # Read data extrinsic calibration camera (OpenConstructor)
        textFileNames = sorted(glob.glob((os.path.join(args['dir'], '00*.txt'))))

        X = get_cam_trans(X, textFileNames)

        # add nodes to graph (images)
        for i in range(1, len(fileNames)):
            GA.add_edge('Map', 'C'+str(i), weight=1)

    elif args['option'] == 'fromAruco':

        # Read data calibration camera (Dictionary elements -> "mtx", "dist")
        d = np.load("CameraParameters/cameraParameters.npy")

        # Load initial intrinsic matrix and distortion vector
        mtx = d.item().get('mtx')
        dist = d.item().get('dist')

    intrinsics = np.zeros((3, 4))
    intrinsics[:, :3] = mtx

    # Define ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()

    s = [stru() for i in range(len(fileNames))]

    # Detect ArUco Markers
    detections = []

    k = 0

    for fileName in fileNames:

        # load image
        raw = cv2.imread(fileName)
        s[k].raw = raw
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # lists of ids and the corners belonging to each id
        corners, ids, _ = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text

        if not (ids is None):
            if len(ids) > 0:
                # print "----------------------------"
                # print("> Camera " + str(k))

                # Estimate pose of each marker
                rotationVecs, translationVecs, _ = aruco.estimatePoseSingleMarkers(corners, markerSize, mtx, dist)

                for rotationVec, translationVec, idd, corner in zip(rotationVecs, translationVecs, ids, corners):

                    detection = MyDetection(rotationVec[0], translationVec[0], 'C' + str(k), 'A' + str(idd[0]), corner)

                    # print(detection)
                    #
                    # print(detection.printValues())
                    # exit(0)
                    detections.append(detection)

                    if args['d'] or args['do']:
                        aruco.drawAxis(raw, mtx, dist, rotationVec, translationVec, 0.05)  # Draw Axis
                        cv2.putText(raw, "Id:" + str(idd[0]), (corner[0][0,0], corner[0][0,1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                aruco.drawDetectedMarkers(raw, corners)

        if args['d'] or args['do']:

            size_square = 15

            # print corners
            for corner in corners:
                draw(raw, corner, size_square)

        k = k+1

    # get shape of the images
    width, height, _ = cv2.imread(fileNames[0]).shape

    # ---------------------------------------
    # --- Set values for intrinsic and distortion parameters in MyX class
    # ---------------------------------------
    X.width = width
    X.height = height
    X.intrinsics = intrinsics
    X.distortion = dist

    # ---------------------------------------
    # --- Initial guess for parameters (create x0).
    # ---------------------------------------

    for detection in detections:
        GA.add_edge(detection.aruco, detection.camera, weight=1)

    if args['d'] or args['do']:
        # Draw graph
        fig2 = plt.figure()

    # pos = nx.random_layout(GA)
    pos = nx.kamada_kawai_layout(GA)
    # pos = nx.spring_layout(GA)

    colors = range(4)
    edges, weights = zip(*nx.get_edge_attributes(GA, 'weight').items())

    if args['d'] or args['do']:
        # edge_labels = nx.draw_networkx_edge_labels(GA, pos)

        nx.draw(GA, pos, node_color='#A0CBE2', edgelist=edges, edge_color=weights, width=6, edge_cmap=plt.cm.Greys_r,
                with_labels=True, alpha=1, node_size=2500, font_color='k')  # , edge_labels=edge_labels)

        plt.waitforbuttonpress(0.1)

    print("----------------------------\n\n" + "Created Nodes:")
    print(GA.nodes)
    print("\n" + "----------------------------")
    print('-> GA is connected ' + str(nx.is_connected(GA)))
    print("----------------------------")

    if not nx.is_connected(GA):
        exit()

    # Manual entering of world reference node by user
    # map_node = request_map_node_input(GA)
    map_node = 'C0' # for debugging purposes

    # Iterate all nodes in graph
    for node in tqdm(GA.nodes):

        # print "--------------------------------------------------------"
        # print('Solving for ' + node + "...")

        paths = list(nx.all_shortest_paths(GA, node, map_node))

        if paths == []:
            paths = [[node]]

        # print(paths)

        transformations_for_path = []

        for path in paths:
            T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)

            for i in range(1, len(path)):

                start = path[i - 1]
                end = path[i]
                start_end = [start, end]

                # print("Start: " + start)
                # print("End: " + end)

                if start == 'Map':
                    if args['option'] == 'fromAruco':
                        Ti = inv(np.identity(4))
                    elif args['option'] == 'fromFile':
                        camera = [camera for camera in X.cameras if camera.id == end[1:]][0]
                        Ti = inv(camera.getT())

                elif end == 'Map':
                    if args['option'] == 'fromAruco':
                        Ti = np.identity(4)
                    elif args['option'] == 'fromFile':
                        camera = [camera for camera in X.cameras if camera.id == start[1:]][0]
                        Ti = camera.getT()

                else:

                    det = [x for x in detections if x.aruco in start_end and x.camera in start_end][0]

                    # print(det)

                    Ti = det.getT()

                    if start[0] == 'C':     # Must invert transformation given by the AruCo detection
                                            # to go from AruCo to camera
                        # print('Will invert...')
                        Ti = inv(Ti)

                T = np.matmul(Ti, T)

                # print("Ti = \n" + str(Ti))
                # print("T = \n" + str(T))

            q = quaternion_from_matrix(T, isprecise=False)
            t = (tuple(T[0:3, 3]), tuple(q))
            # print t
            # exit()
            transformations_for_path.append(t)

        qm = averageTransforms(transformations_for_path)
        # print qm

        T = quaternion_matrix(qm[1])
        T[0, 3] = qm[0][0]
        T[1, 3] = qm[0][1]
        T[2, 3] = qm[0][2]
        # print T
        # exit()

        # print("Transformation from " + node +
        #       " to " + map_node + " is: \n" + str(T))

        # node is a camera
        if node[0] == 'C' and args['option'] == 'fromAruco':
            camera = MyCamera(T=T, id=node[1:])
            X.cameras.append(camera)
        elif node == 'Map':
            vvv = 0
            # camera = MyCamera(T=T, id='Map')
            # X.cameras.append(camera)
        else:
            aruco = MyAruco(T=T, id=node[1:])
            X.arucos.append(aruco)

    X.arucos.sort(key=lambda x: float(x.id), reverse=False)

    # Get vector x0
    X.toVector(args)
    print("X.v= ")
    print(X.v)

    x0 = np.array(X.v, dtype=np.float)
    # print len(X.cameras)
    # print len(X.arucos)
    # print len(x0)
    # print x0
    # exit()

    #Add noise to X0
    # x_random = x0 * np.array([random.uniform(0.99, 1.01)
    #                           for _ in xrange(len(x0))], dtype=np.float)

    # x0 = x_random
    # X.fromVector(x0, args)

    # print x0

    l = markerSize
    Pc = np.array([[0, 0, 0]])

    handles = []

    if args['d'] or args['do']:
        draw_initial_guess(X, Pc, detections, height, width)

    key = ord('w')
    while key != ord('o'):
        key = cv2.waitKey(20)
        plt.waitforbuttonpress(0.01)

    # ---------------------------------------
    # --- Compute ground truth of initial estimate
    # ---------------------------------------

    RealPts = []
    Aid = 0
    factor = markerSize + 0.02

    for y in range(6):
        yi = -(y * factor)
        for x in range(9):
            xi = x * factor
            Pt = Point3DtoComputeError(xi, yi, 0, str(Aid))
            RealPts.append(Pt)
            Aid = Aid + 1

    # for RealPt in RealPts:
    #     print "A" + RealPt.id + ": \n" + \
    #         str(RealPt.x) + ", " + str(RealPt.y) + ", " + str(RealPt.z)
    # exit()

    # ---------------------------------------
    # --- Test call of objective function
    # ---------------------------------------

    costFunction.counter = 0

    # Call objective function with initial guess (just for testing)
    initial_residuals = costFunction(x0, X, Pc, detections, args, handles, None, s)

    handle_fun = None
    if args['d'] or args['do']:
        # Draw graph
        fig4 = plt.figure()
        axcost = fig4.add_subplot(111)
        plt.plot(initial_residuals, 'b',
                 label="Initial residuals")

        handle_fun, = plt.plot(initial_residuals, 'r--',
                               label="Final residuals")

        plt.legend(loc='best', prop={'size': 22})
        axcost.set_xlabel('Detections', size=22)
        axcost.set_ylabel('Cost (pixel)', size=22)
        plt.ylim(ymin=-0.05)
        plt.waitforbuttonpress(0.1)

        y1 = range(0, 13, 2)
        squady = list(['0', '2', '4', '6', '8', '10', '12'])

        # Put detections on x-axis
        squad = []
        number_of_detections = len(detections)
        x1 = range(number_of_detections)
        for detection in detections:
            xstring = detection.camera + '/' + detection.aruco
            squad.append(xstring)

        axcost.set_xticks(x1)
        axcost.set_xticklabels(squad, minor=False, rotation=25, size=17)
        axcost.set_yticks(y1)
        axcost.set_yticklabels(squady, minor=False, rotation=0, size=17)

    # print("\n-> Initial cost = " + str(initial_residuals)) + "\n"

    # -------------------------------------------------------------------------------
    # --- TODO: Create the sparse matrix

    M = len(detections)
    N = len(x0)
    # 21 x number of detections
    # 6 cam param, 6 ArUco param, 4 intrinsics, 5 distortion = 21
    print("Sparse matrix is " + str(M) + " x " + str(N))

    A = lil_matrix((M, N), dtype=int)

    id_detection = 0

    for detection in detections:

        camera = [camera for camera in X.cameras if camera.id == detection.camera[1:]][0]
        idxs_camera = X.idxsFromCamera(camera.id, args)
        print("CAMERA= ")
        print(idxs_camera)

        aruco = [aruco for aruco in X.arucos if aruco.id == detection.aruco[1:]][0]
        idxs_aruco = X.idxsFromAruco(aruco.id, args)
        print("ARUCO= ")
        print(idxs_aruco)

        idxs_intrinsic_and_distortion = X.idxsFromIntrinsicAndDistortion(camera.id, args)
        print("Intrinsics and distortion= ")
        print(idxs_intrinsic_and_distortion)

        idxs = np.append(idxs_camera, idxs_aruco)
        idxs = np.append(idxs, idxs_intrinsic_and_distortion)
        print("IDXS= ")
        print(idxs)
        print("-------------------------------")

        A[id_detection, idxs] = 1

        id_detection = id_detection + 1

    print('A shape = ' + str(A.shape))
    print('A =\n' + str(A.toarray()))

    # ---------------------------------------
    # --- Set the bounds for the parameters

    # bounds : 2-tuple of array_like, optional
    # Lower and upper bounds on independent variables. Defaults to no bounds.
    # Each array must match the size of x0 or be a scalar,
    # in the latter case a bound will be the same for all variables.
    # Use np.inf with an appropriate sign to disable bounds on all or some variables.

    # camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    # points_3d = params[n_cameras * 9:].reshape((n_points, 3))

    if args['option2'] == 'all':
        n_values_per_aruco = 6
    else:
        n_values_per_aruco = 3

    # Cameras
    Bmin_cameras = []
    Bmax_cameras = []
    for i in range(0, len(X.cameras)):
        for i in range(0, 6):
            Bmin_cameras.append(-np.inf)
            Bmax_cameras.append(np.inf)

    # ArUcos
    Bmin_aruco = []
    Bmax_aruco = []
    idx_aruco = 6 * len(X.cameras)
    for i in range(0, len(X.arucos)):
        for i in range(0, n_values_per_aruco):
            # delta = abs(x0[idx_aruco] * 0.01)
            # Bmin_aruco.append(x0[idx_aruco] - delta)
            # Bmax_aruco.append(x0[idx_aruco] + delta)
            Bmin_aruco.append(-np.inf)
            Bmax_aruco.append(np.inf)
            idx_aruco = idx_aruco + 1

    # Intrinsics
    Bmin_intrinsics = []
    Bmax_intrinsics = []
    idx_intrinsics = len(X.cameras) * 6 + len(X.arucos) * n_values_per_aruco
    for i in range(0, 4):
        # delta = abs(x0[idx_intrinsics+i] * 0.1)
        # Bmin_intrinsics.append(x0[idx_intrinsics+i] - delta)
        # Bmax_intrinsics.append(x0[idx_intrinsics+i] + delta)
        Bmin_intrinsics.append(-np.inf)
        Bmax_intrinsics.append(np.inf)

    print("Bmin_intrinsics= ")
    print(Bmin_intrinsics)
    print("Bmax_intrinsics= ")
    print(Bmax_intrinsics)

    # Distortion
    Bmin_distortion = []
    Bmax_distortion = []
    idx_distortion = idx_intrinsics + 4
    for i in range(0, 5):
        # delta = abs(x0[idx_distortion+i] * 0.1)
        # Bmin_distortion.append(x0[idx_distortion+i] - delta)
        # Bmax_distortion.append(x0[idx_distortion+i] + delta)
        Bmin_distortion.append(-np.inf)
        Bmax_distortion.append(np.inf)

    print("Bmin_distortion= ")
    print(Bmin_distortion)
    print("Bmax_distortion= ")
    print(Bmax_distortion)

    # Merge min vectors
    Bmin = Bmin_cameras
    Bmin.extend(Bmin_aruco)
    Bmin.extend(Bmin_intrinsics)
    Bmin.extend(Bmin_distortion)

    # Merge max vectors
    Bmax = Bmax_cameras
    Bmax.extend(Bmax_aruco)
    Bmax.extend(Bmax_intrinsics)
    Bmax.extend(Bmax_distortion)

    # Convert into tuple
    bounds = (Bmin, Bmax)

    # Debugging
    print("BOUNDS = ")
    print(bounds)

    # ---------------------------------------
    # --- Optimization (minimization)

    print("\n\nStarting minimization")

    # cost_random = costFunction(x_random, dist, intrinsics, s, X, Pc)
    # print(x_random)

    # exit(0)

    # --- Without sparsity matrix
    t0 = time.time()

    # Method TODO: changes here in args?
    res = least_squares(costFunction, x0, verbose=2, jac_sparsity=A, x_scale='jac', ftol=1e-4, xtol=1e-4, bounds=bounds,
                     method='trf', args=(X, Pc, detections, args, handles, handle_fun, s))

    # print(res.x)
    t1 = time.time()

    print("\nOptimization took {0:.0f} seconds".format(t1 - t0))

    X.fromVector(list(res.x), args)

    # ---------------------------------------
    # --- Present the results
    # ---------------------------------------
    solution_residuals = costFunction(res.x, X, Pc, detections, args, handles, None, s)

    print("\nOPTIMIZATON FINISHED")
    print("Initial (x0) average error = " +
          str(np.average(initial_residuals)))
    print("Solution average error = " +
          str(np.average(solution_residuals))) + "\n"


    # TODO: change to func?
    # ---------------------------------------
    # --- Save results
    # ---------------------------------------

    if args['saveResults']:
        print("Saving results...")

        textFileNames = sorted(glob.glob((os.path.join(newDir, '00*.txt'))))

        for w, cltx in enumerate(textFileNames):
            nt = len(cltx)
            if cltx[nt-6] == "-" or cltx[nt-7] == "-" or cltx[nt-8] == "-":
                del textFileNames[w]

        for j, nameFile in enumerate(textFileNames):

            camera = [camera for camera in X.cameras if camera.id ==
                      str(j)][0]
            T = camera.getT()
            Tp = T.transpose()

            lines = open(nameFile).read().splitlines()
            for i in range(4):
                lines[i+1] = str(Tp[i][0]) + ' ' + str(Tp[i][1]) + ' ' + \
                    str(Tp[i][2]) + ' ' + str(Tp[i][3])

            open(nameFile, 'w').write('\n'.join(lines))
            print("Save T optimized for camera " + str(j) + "...")

    # TODO: change to func?
    # ---------------------------------------
    # --- Draw final results
    # ---------------------------------------

    if args['d'] or args['do']:
        # fig5 = plt.figure()
        # plt.plot(solution_residuals, 'r--')
        handle_fun.set_ydata(solution_residuals)
        plt.waitforbuttonpress(0.1)

        for detection in detections:
            camera = [camera for camera in X.cameras if camera.id ==
                      detection.camera[1:]][0]

            aruco = [aruco for aruco in X.arucos if aruco.id ==
                     detection.aruco[1:]][0]

            Ta = aruco.getT()
            Tc = camera.getT()
            T = np.matmul(inv(Tc), Ta)

            xypix = points2imageFromT(T, X.intrinsics, Pc, X.distortion)

            k = int(camera.id)

            # Draw initial projections
            if 0 < xypix[0][0] < height and 0 < xypix[0][1] < width:
                cv2.circle(s[k].raw, (int(xypix[0][0]), int(xypix[0][1])),
                           10, (0, 255, 255), -1)

            cv2.imshow('camera'+str(k), s[k].raw)

        X.setPlot3D(Pc)

        # Average Error of 3D projection
        # Error = computeError(RealPts, X.InitPts)

        # print '\nAverage Error of initial estimation = ' + str(Error)

        # Error = computeError(RealPts, X.OptPts)

        # print '\nAverage Error after optimization = ' + str(Error) + '\n'

        while key != ord('q'):
            key = cv2.waitKey(20)
            plt.waitforbuttonpress(0.01)

    # Process dataset
    if args['processDataset'] and args['saveResults']:
        bash('./processDataset.py ' + args['dir'])