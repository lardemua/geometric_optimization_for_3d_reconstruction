# MyClasses

from costFunctions import *
import cv2  # OpenCV library



class stru:
    def __init__(self):
        self.raw = None


class MyPoint3D:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None


class MyRodrigues:
    def __init__(self):
        self.r1 = None
        self.r2 = None
        self.r3 = None


class Point3DtoComputeError(MyPoint3D):
    def __init__(self, x, y, z, id):
        MyPoint3D.__init__(self)
        self.x = x
        self.y = y
        self.z = z
        self.id = id


class MyHandle:
    def __init__(self, handle_scatter, handle_text):
        self.handle_scatter = handle_scatter
        self.handle_text = handle_text


class MyHandle3D:
    def __init__(self, handle_scatter, handle_plotx, handle_ploty, handle_plotz, handle_text, handle_textx, handle_texty, handle_textz):
        self.handle_scatter = handle_scatter
        self.handle_plotx = handle_plotx
        self.handle_ploty = handle_ploty
        self.handle_plotz = handle_plotz
        self.handle_text = handle_text
        self.handle_textx = handle_textx
        self.handle_texty = handle_texty
        self.handle_textz = handle_textz


class MyTransform(MyRodrigues, MyPoint3D):
    def __init__(self, rvec, tvec):
        MyPoint3D.__init__(self)
        self.x = tvec[0]
        self.y = tvec[1]
        self.z = tvec[2]
        MyRodrigues.__init__(self)
        self.r1 = rvec[0]
        self.r2 = rvec[1]
        self.r3 = rvec[2]

    def getT(self):
        """Transformation matrix
            Gets 4x4 Tranform from internal xyz and Rodrigues
        """
        T = np.eye(4)
        dxyz = np.array([self.x, self.y,
                         self.z], dtype=np.float)
        T[0:3, 3] = dxyz.transpose()
        rod = np.array([self.r1, self.r2, self.r3], dtype=np.float)
        DCM = cv2.Rodrigues(rod)
        T[0:3, 0:3] = DCM[0]
        return T

    def setPointRod(self, T):
        """Transformation matrix
            Sets point and rod values from 4x4 Tranform
        """
        dxyz = T[0:3, 3]
        dxyz = dxyz.transpose()
        rods, _ = cv2.Rodrigues(T[0:3, 0:3])
        rods = rods.transpose()
        rod = rods[0]
        self.x = dxyz[0]
        self.y = dxyz[1]
        self.z = dxyz[2]
        self.r1 = rod[0]
        self.r2 = rod[1]
        self.r3 = rod[2]

    def printValues(self):
        return "xyz = " + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "\n" + "rod = " + str(self.r1) + ", " + str(self.r2) + ", " + str(self.r3)


class MyDetection(MyTransform):
    def __init__(self, rvec, tvec, cam, ar, corner):
        MyTransform.__init__(self, rvec, tvec)
        self.camera = cam
        self.aruco = ar
        self.corner = corner

    def __str__(self):
        return "\nDetection from camera " + str(self.camera) + " of aruco " + str(self.aruco) + ":"


class MyCamera(MyTransform):
    def __init__(self, T, id):
        MyTransform.setPointRod(self, T)
        self.id = id


class MyAruco(MyTransform):
    def __init__(self, T, id):
        MyTransform.setPointRod(self, T)
        self.id = id


class MyX:
    def __init__(self):
        self.cameras = []
        self.arucos = []
        self.v = []
        self.ArucoHandles = []
        self.CameraHandles = []
        self.InitPts = []
        self.OptPts = []
        self.distortion = []
        self.intrinsics = []
        self.width = None
        self.height = None

    def plot3D(self, ax3D, symbol, Pc):

        # Referential of arucos
        s = 0.08
        P = np.array([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]])

        for aruco in self.arucos:
            T = aruco.getT()
            rot = T[0:3, 0:3]
            trans = T[0: 3, 3]

            Ptransf = []
            wp = np.zeros((4, 3))

            for p in P:
                Ptransf.append(rot.dot(p) + trans)

            i = 0
            for p in Pc:
                wp[i, :] = rot.dot(p) + trans
                i = i + 1

            # Draw aruco axis
            handle_plotx, = ax3D.plot([Ptransf[0][0], Ptransf[1][0]], [
                Ptransf[0][1], Ptransf[1][1]], [Ptransf[0][2], Ptransf[1][2]], 'r-')
            handle_ploty, = ax3D.plot([Ptransf[0][0], Ptransf[2][0]], [
                Ptransf[0][1], Ptransf[2][1]], [Ptransf[0][2], Ptransf[2][2]], 'g-')
            handle_plotz, = ax3D.plot([Ptransf[0][0], Ptransf[3][0]], [
                Ptransf[0][1], Ptransf[3][1]], [Ptransf[0][2], Ptransf[3][2]], 'b-')

            # Draw text
            handle_text = ax3D.text(Ptransf[0][0], Ptransf[0][1],
                                    Ptransf[0][2], "A" + aruco.id, color='darkorchid')  # , size=20, zorder=1)
            handle_textx = ax3D.text(Ptransf[1][0], Ptransf[1][1],
                                     Ptransf[1][2], "x", color='red')
            handle_texty = ax3D.text(Ptransf[2][0], Ptransf[2][1],
                                     Ptransf[2][2], "y", color='green')
            handle_textz = ax3D.text(Ptransf[3][0], Ptransf[3][1],
                                     Ptransf[3][2], "z", color='blue')

            # Draw aruco point 3D
            wp = wp.transpose()

            # print 'Aruco ' + aruco.id + ':' + '\n       x -> ' + \
            #     str(wp[0, 0]) + '\n       y -> ' + \
            #     str(wp[1, 0]) + '\n       z -> ' + str(wp[2, 0])

            Pta = Point3DtoComputeError(wp[0, 0], wp[1, 0], wp[2, 0], aruco.id)
            self.InitPts.append(Pta)

            handle_scatter = ax3D.scatter(wp[0, :], wp[1, :], wp[2, :], symbol)

            handle = MyHandle3D(handle_scatter, handle_plotx, handle_ploty, handle_plotz,
                                handle_text, handle_textx, handle_texty, handle_textz)
            self.ArucoHandles.append(handle)

        for camera in self.cameras:
            T = camera.getT()
            rot = T[0:3, 0:3]
            trans = T[0: 3, 3]

            Ptransf = []
            wp = np.zeros((4, 3))

            for p in P:
                Ptransf.append(rot.dot(p) + trans)

            i = 0
            for p in Pc:
                wp[i, :] = rot.dot(p) + trans
                i = i + 1

            # Draw camera axis
            handle_plotx, = ax3D.plot([Ptransf[0][0], Ptransf[1][0]], [
                Ptransf[0][1], Ptransf[1][1]], [Ptransf[0][2], Ptransf[1][2]], 'r-')
            handle_ploty, = ax3D.plot([Ptransf[0][0], Ptransf[2][0]], [
                Ptransf[0][1], Ptransf[2][1]], [Ptransf[0][2], Ptransf[2][2]], 'g-')
            handle_plotz, = ax3D.plot([Ptransf[0][0], Ptransf[3][0]], [
                Ptransf[0][1], Ptransf[3][1]], [Ptransf[0][2], Ptransf[3][2]], 'b-')

            # Draw text
            handle_text = ax3D.text(Ptransf[0][0], Ptransf[0][1],
                                    Ptransf[0][2], "C" + camera.id, color='darkorchid')  # , size=15, zorder=1)
            handle_textx = ax3D.text(Ptransf[1][0], Ptransf[1][1],
                                     Ptransf[1][2], "x", color='red')
            handle_texty = ax3D.text(Ptransf[2][0], Ptransf[2][1],
                                     Ptransf[2][2], "y", color='green')
            handle_textz = ax3D.text(Ptransf[3][0], Ptransf[3][1],
                                     Ptransf[3][2], "z", color='blue')

            handle = MyHandle3D(None, handle_plotx, handle_ploty, handle_plotz,
                                handle_text, handle_textx, handle_texty, handle_textz)
            self.CameraHandles.append(handle)

    def setPlot3D(self, Pc):

        # Referential of arucos
        s = 0.080
        P = np.array([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]])
        self.OptPts = []

        for aruco, handle in zip(self.arucos, self.ArucoHandles):

            T = aruco.getT()
            rot = T[0:3, 0:3]
            trans = T[0: 3, 3]

            Ptransf = []
            wp = np.zeros((4, 3))

            for p in P:
                Ptransf.append(rot.dot(p) + trans)

            i = 0
            for p in Pc:
                wp[i, :] = rot.dot(p) + trans
                i = i + 1

            # redraw plot x
            handle.handle_plotx.set_xdata([Ptransf[0][0], Ptransf[1][0]])
            handle.handle_plotx.set_ydata([Ptransf[0][1], Ptransf[1][1]])
            handle.handle_plotx.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[1][2]])

            # redraw plot y
            handle.handle_ploty.set_xdata([Ptransf[0][0], Ptransf[2][0]])
            handle.handle_ploty.set_ydata([Ptransf[0][1], Ptransf[2][1]])
            handle.handle_ploty.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[2][2]])

            # redraw plot z
            handle.handle_plotz.set_xdata([Ptransf[0][0], Ptransf[3][0]])
            handle.handle_plotz.set_ydata([Ptransf[0][1], Ptransf[3][1]])
            handle.handle_plotz.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[3][2]])

            # redraw text
            handle.handle_text.set_position((Ptransf[0][0], Ptransf[0][1]))
            handle.handle_text.set_3d_properties(z=Ptransf[0][2], zdir='y')

            # redraw text x
            handle.handle_textx.set_position((Ptransf[1][0], Ptransf[1][1]))
            handle.handle_textx.set_3d_properties(z=Ptransf[1][2], zdir='y')

            # redraw text y
            handle.handle_texty.set_position((Ptransf[2][0], Ptransf[2][1]))
            handle.handle_texty.set_3d_properties(z=Ptransf[2][2], zdir='y')

            # redraw text z
            handle.handle_textz.set_position((Ptransf[3][0], Ptransf[3][1]))
            handle.handle_textz.set_3d_properties(z=Ptransf[3][2], zdir='y')

            # Draw aruco point 3D
            wp = wp.transpose()

            # print 'Aruco ' + aruco.id + ':' + '\n       x -> ' + \
            #     str(wp[0, 0]) + '\n       y -> ' + \
            #     str(wp[1, 0]) + '\n       z -> ' + str(wp[2, 0])

            Pta = Point3DtoComputeError(wp[0, 0], wp[1, 0], wp[2, 0], aruco.id)
            self.OptPts.append(Pta)

            # redraw scatter
            handle.handle_scatter._offsets3d = (wp[0, :], wp[1, :], wp[2, :])

        for camera, handle in zip(self.cameras, self.CameraHandles):
            T = camera.getT()
            rot = T[0:3, 0:3]
            trans = T[0: 3, 3]

            Ptransf = []
            wp = np.zeros((4, 3))

            for p in P:
                Ptransf.append(rot.dot(p) + trans)

            i = 0
            for p in Pc:
                wp[i, :] = rot.dot(p) + trans
                i = i + 1

            # redraw plot x
            handle.handle_plotx.set_xdata([Ptransf[0][0], Ptransf[1][0]])
            handle.handle_plotx.set_ydata([Ptransf[0][1], Ptransf[1][1]])
            handle.handle_plotx.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[1][2]])

            # redraw plot y
            handle.handle_ploty.set_xdata([Ptransf[0][0], Ptransf[2][0]])
            handle.handle_ploty.set_ydata([Ptransf[0][1], Ptransf[2][1]])
            handle.handle_ploty.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[2][2]])

            # redraw plot z
            handle.handle_plotz.set_xdata([Ptransf[0][0], Ptransf[3][0]])
            handle.handle_plotz.set_ydata([Ptransf[0][1], Ptransf[3][1]])
            handle.handle_plotz.set_3d_properties(
                zs=[Ptransf[0][2], Ptransf[3][2]])

            # redraw text
            handle.handle_text.set_position((Ptransf[0][0], Ptransf[0][1]))
            handle.handle_text.set_3d_properties(z=Ptransf[0][2], zdir='y')

            # redraw text x
            handle.handle_textx.set_position((Ptransf[1][0], Ptransf[1][1]))
            handle.handle_textx.set_3d_properties(z=Ptransf[1][2], zdir='y')

            # redraw text y
            handle.handle_texty.set_position((Ptransf[2][0], Ptransf[2][1]))
            handle.handle_texty.set_3d_properties(z=Ptransf[2][2], zdir='y')

            # redraw text z
            handle.handle_textz.set_position((Ptransf[3][0], Ptransf[3][1]))
            handle.handle_textz.set_3d_properties(z=Ptransf[3][2], zdir='y')

    def toVector(self, args):
        n_cameras = len(self.cameras)
        n_arucos = len(self.arucos)

        for i in range(n_cameras):
            self.v.extend(
                [self.cameras[i].x, self.cameras[i].y, self.cameras[i].z,
                 self.cameras[i].r1, self.cameras[i].r2, self.cameras[i].r3])

        for i in range(n_arucos):
            # print('Aruco ' + str(self.arucos[i].id))
            if args['option2'] == 'all':
                self.v.extend(
                    [self.arucos[i].x, self.arucos[i].y, self.arucos[i].z,
                     self.arucos[i].r1, self.arucos[i].r2, self.arucos[i].r3])
            else:
                self.v.extend(
                    [self.arucos[i].x, self.arucos[i].y, self.arucos[i].z])

        # Use intrinsic and distortion parameters in vector to optimize
        self.v.extend([self.intrinsics[0][0], self.intrinsics[0][2], self.intrinsics[1][1], self.intrinsics[1][2]])
        # print([self.intrinsics[0][0], self.intrinsics[0][2],self.intrinsics[1][1],self.intrinsics[1][2]])

        self.v.extend(self.distortion[0])
        # print(self.distortion[0])

    def fromVector(self, v, args):
        n_cameras = len(self.cameras)
        n_arucos = len(self.arucos)

        if args['option2'] == 'all':
            n_values_per_aruco = 6
        else:
            n_values_per_aruco = 3

        for i in range(n_cameras):
            self.cameras[i].x = v[i*6]
            self.cameras[i].y = v[i*6+1]
            self.cameras[i].z = v[i*6+2]
            self.cameras[i].r1 = v[i*6+3]
            self.cameras[i].r2 = v[i*6+4]
            self.cameras[i].r3 = v[i*6+5]
        for i in range(n_arucos):
            self.arucos[i].x = v[i*n_values_per_aruco+n_cameras*6]
            self.arucos[i].y = v[i*n_values_per_aruco+1+n_cameras*6]
            self.arucos[i].z = v[i*n_values_per_aruco+2+n_cameras*6]
            if args['option2'] == 'all':
                self.arucos[i].r1 = v[i*n_values_per_aruco+3+n_cameras*6]
                self.arucos[i].r2 = v[i*n_values_per_aruco+4+n_cameras*6]
                self.arucos[i].r3 = v[i*n_values_per_aruco+5+n_cameras*6]

        idx_intrinsics = 6 * n_cameras + n_values_per_aruco * n_arucos
        self.intrinsics[0][0] = v[idx_intrinsics]
        self.intrinsics[0][2] = v[idx_intrinsics+1]
        self.intrinsics[1][1] = v[idx_intrinsics+2]
        self.intrinsics[1][2] = v[idx_intrinsics+3]

        idx_distortion = idx_intrinsics + 4
        self.distortion[0][0] = v[idx_distortion]
        self.distortion[0][1] = v[idx_distortion+1]
        self.distortion[0][2] = v[idx_distortion+2]
        self.distortion[0][3] = v[idx_distortion+3]
        self.distortion[0][4] = v[idx_distortion+4]

    def idxsFromCamera(self, camera, args):

        idx_in_cameras = [x.id for x in self.cameras].index(camera)
        # print(idx_in_cameras)
        idxs_in_X = np.array(range(6)) + 6 * idx_in_cameras
        # print(idxs_in_X)
        return idxs_in_X

    def idxsFromAruco(self, aruco, args):

        n_cameras = len(self.cameras)
        if args['option2'] == 'all':
            n_values_per_aruco = 6
        else:
            n_values_per_aruco = 3

        idx_in_arucos = [x.id for x in self.arucos].index(aruco)
        # print("Aruco id = " + aruco + "\n")
        # print("Aruco index = " + str(idx_in_arucos) + "\n")

#                                                          offset
        idxs_in_X = np.array(range(n_values_per_aruco)) + (6 * n_cameras + idx_in_arucos * n_values_per_aruco)
        # print(idxs_in_X)

        return idxs_in_X

    def idxsFromIntrinsicAndDistortion(self, camera, args):
        # if more than one different camera is used? maybe obtain different values for different cams

        n_cameras = len(self.cameras)
        n_arucos = len(self.arucos)
        if args['option2'] == 'all':
            n_values_per_aruco = 6
        else:
            n_values_per_aruco = 3

        starting_idx = int(6 * n_cameras + n_arucos * n_values_per_aruco)

        # For now... 4 intrinsics + 5 distortions = 9
        idxs = np.array(range(9)) + starting_idx

        return idxs

