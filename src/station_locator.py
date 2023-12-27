#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf.transformations as transformations
import tf
import pyrealsense2 as rs
from geometry_msgs.msg import TransformStamped

def rotationMatrixToQuaternion1(rot_matrix):
    trace = np.trace(rot_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
        y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
        z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
    else:
        if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([x, y, z, w])
    return quaternion

c_info = ""
camera_matrix = []
dist_coeffs = np.zeros((4, 1), dtype=np.float32)
sift = cv2.SIFT_create()
calibration = cv2.imread('../resources/qr_code_big.png')
calibGray = cv2.cvtColor(calibration, cv2.COLOR_BGR2GRAY)
h, w = calibGray.shape
kp1, des1 = sift.detectAndCompute(calibGray, None)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

object_points = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 1, 0]], dtype=np.float32)

center = (0,0)
depth_msg = 0
intrinsics = []
tf_pub = None

yaw_increase = np.radians(180)
quaternion_rotated = transformations.quaternion_from_euler(0, 0, yaw_increase)

def image_callback(msg):
    global camera_matrix, dist_coeffs, sift, calibration, calibGray, h, w, tf_pub
    global kp1, des1, colors, object_points, center, depth_msg, intrinsics
    try:
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

        calibrated = True
        wframe = cv_image

        wframeGray = cv2.cvtColor(wframe, cv2.COLOR_BGR2GRAY)
        hc, wc = wframeGray.shape

        bf = cv2.BFMatcher()
        kp2, des2 = sift.detectAndCompute(wframeGray, None)

        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        if len(good) > 10:
            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
            if matrix is not None:
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                if dst.shape == (4, 1, 2):
                    for i in range(4):
                        if 0 < np.int32(dst[i][0][0]) < wc and 0 < np.int32(dst[i][0][1]) < hc:
                            calibrated = calibrated & True
                        else:
                            print("These values are useless")
                            calibrated = False
                    if calibrated is True:
                        image = wframe
                        pts = (np.float32(dst).reshape(4, 2))
                        image_points = pts
                        x = int((pts[2][0] - pts[0][0]) / 2)
                        y = int((pts[2][1] - pts[0][1]) / 2)

                        pts = np.int32(pts).reshape((-1, 1, 2))
                        cv2.polylines(image, [pts], True, (0, 255, 255), 3)

                        success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points,
                                                                                    camera_matrix, dist_coeffs)

                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                        euler_angles_rad = cv2.RQDecomp3x3(rotation_matrix)[0]
                        euler_angles_deg = np.degrees(euler_angles_rad)

                        unitv_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape(
                            (4, 1, 3))
                        points, jacobian = cv2.projectPoints(unitv_points, rotation_vector, translation_vector,
                                                             camera_matrix, dist_coeffs)

                        #quat = rotationMatrixToQuaternion1(rotation_matrix)
                        #print(euler_angles_deg)
                        quat = transformations.quaternion_from_euler(0,0,np.radians(euler_angles_rad[0]))
                        #quat = transformations.quaternion_inverse(quat_temp)
                        #quat = transformations.quaternion_from_matrix(rotation_matrix)

                        quat = transformations.quaternion_multiply(quaternion_rotated, quat)

                        if len(points) > 0:
                            points = points.reshape((4, 2))
                            # print(points)
                            # points = [(points[i][0] + x, points[i][1] + y) for points[i] in points]
                            for i in range(0, len(points)):
                                points[i][0] += x
                                points[i][1] += y
                            origin = (int(points[0][0]), int(points[0][1]))
                            center = origin

                            # print(image.shape)
                            if image.shape[1] >= origin[0] >= 0 and image.shape[0] >= origin[1] >= 0:
                                depth_distance = depth_msg/1000
                                print(depth_distance)
                                dx, dy, dz = rs.rs2_deproject_pixel_to_point(intrinsics, [origin[0], origin[1]],
                                                                             depth_distance)
                                dx = dx * (-1)
                            else:
                                dx, dy, dz = 0, 0, 0

                            #print("x:" + str(dx) + " y:" + str(dy) + " z:" + str(dz) + " angles:" + str(quat))
                            position = {"x": dx,
                                        "y": dy,
                                        "z": dz}
                            orientation = {"x": quat[0],
                                           "y": quat[1],
                                           "z": quat[2],
                                           "w": quat[3]}
                            pose = {"position": position,
                                    "orientation": orientation}
                            print(pose)

                            tf_msg = TransformStamped()
                            tf_msg.header.stamp = rospy.Time.now()
                            tf_msg.header.frame_id = "camera_link"
                            tf_msg.child_frame_id = "qr_frame"
                            tf_msg.transform.translation.x = dz
                            tf_msg.transform.translation.y = dx
                            tf_msg.transform.translation.z = 0
                            tf_msg.transform.rotation.x = quat[0]
                            tf_msg.transform.rotation.y = quat[1]
                            tf_msg.transform.rotation.z = quat[2]
                            tf_msg.transform.rotation.w = quat[3]

                            tf_pub.sendTransformMessage(tf_msg)

                            for p, c in zip(points[1:], colors[:3]):
                                p = (int(p[0]), int(p[1]))

                                cv2.line(image, origin, p, c, 5)

                        cv2.imshow("Image Window", image)
                        cv2.waitKey(1)
            else:
                print("these values are useless")
        else:
            print("Not enough matches are found - {}/{}".format(len(good), 10))
    except Exception as e:
        print(e)

def camera_info_callback(msg):
    global c_info, camera_matrix, intrinsics
    camera_matrix = np.array([msg.K], dtype=np.float32)
    camera_matrix = camera_matrix.reshape((3,3))

    intrinsics = rs.intrinsics()
    intrinsics.width = msg.width
    intrinsics.height = msg.height
    intrinsics.ppx = msg.K[2]
    intrinsics.ppy = msg.K[5]
    intrinsics.fx = msg.K[0]
    intrinsics.fy = msg.K[4]
    if msg.distortion_model == 'plumb_bob':
        intrinsics.model = rs.distortion.brown_conrady
    elif msg.distortion_model == 'equidistant':
        intrinsics.model = rs.distortion.kannala_brandt4
    #intrinsics.coeffs = [i for i in msg.D]
    intrinsics.coeffs = [0,0,0,0,0]

    c_info.unregister()

def depth_callback(msg):
    global center, depth_msg
    cv_image = CvBridge().imgmsg_to_cv2(msg, msg.encoding)
    depth_msg = cv_image[center[1], center[0]]

def image_subscriber():
    global c_info, tf_pub

    rospy.init_node('station_locator_node', anonymous=True)

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    #rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    c_info = rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback)
    #c_info = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, camera_info_callback)
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_callback)
    #rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)
    tf_pub = tf.TransformBroadcaster()

    rospy.spin()

if __name__ == '__main__':
    try:
        image_subscriber()
    except rospy.ROSInterruptException:
        pass