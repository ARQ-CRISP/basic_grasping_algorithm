#!/usr/bin/env python

import pcl
import numpy as np
from sklearn.decomposition import PCA
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def crop_pointcloud(pointcloud, min_x=-0.345, max_x=0.355, min_y=0.2, max_y=0.9, min_z=0.84, max_z=1.35):
    """
        Function taking as input a PCL point cloud and outputs its cropped version according to the input parameters.
        Default values correspond to expected values to crop a 50x50cm area in the simulated environment of the UR5 in ARQ.

        @param pointcloud: PCL PointCloud to crop
        @param min_x: Float specifying the lower bound value of the x values to keep
        @param max_x: Float specifying the upper bound value of the x values to keep
        @param min_y: Float specifying the lower bound value of the y values to keep
        @param max_y: Float specifying the upper bound value of the y values to keep
        @param min_z: Float specifying the lower bound value of the z values to keep
        @param max_z: Float specifying the upper bound value of the z values to keep
        @return: PCL PointCloud corresponding to the configured cropping
    """
    # Create cropping box
    cropper = pointcloud.make_cropbox()
    cropper.set_Translation(0, 0, 0)
    cropper.set_Rotation(0, 0, 0)
    # Set the boundaries
    cropper.set_MinMax(min_x, min_y, min_z, 1, max_x, max_y, max_z, 1)
    # Apply the cropping
    cropped_cloud = cropper.filter()
    return cropped_cloud


def plane_segmentation(pointcloud, ransac_threshold=0.01):
    """
        Function that returns two numpy arrays which correspond to the segmented plane and object found in a PCL
        point cloud.

        @param pointcloud: PCL PointCloud to segment
        @param ransac_threshold: Upper bound of the error (in meters) tolerated to consider a point as part of the plane

        @return: Two numpy arrays, corresponding to the segmented plane and object from the input point cloud
    """
    # Initialize PCL RANSAC to carry out the plane segmentation
    model_plane = pcl.SampleConsensusModelPlane(pointcloud)
    ransac = pcl.RandomSampleConsensus(model_plane)
    # Set the tolerated error
    ransac.set_DistanceThreshold(ransac_threshold)
    # Run the segmentation
    ransac.computeModel()
    # Get the indices of all the points part of the segmented plane
    inliers = ransac.get_Inliers()
    # Use Python sets to get the remaining indices (i.e. corresponding to the object)
    outliers = list(set(range(pointcloud.size)) - set(inliers))
    # Extract the points from the numpy array version of the PCL PointCloud
    all_points = pointcloud.to_array()
    plane_points = all_points[inliers]
    object_points = all_points[outliers]
    # Return the two numpy arraysobject
    return plane_points, object_points


def get_grasp_pose(object_points, plane_points, roll_offset=np.pi, pitch_offset=0, yaw_offset=np.pi / 2,
                   finger_length=0.1157, reference_frame="world"):
    """
        Generate a PoseStamped message from two numpy arrays, corresponding to a segmented plane and objects. Default
        values correspond to how we the EZGripper is mounted on the UR5 at ARQ lab.

        @param object_points: Numpy array corresponding to the points of the object
        @param plane_points: Numpy array corresponding to the points of the segmented plane
        @param roll_offset: Roll offset between the reference frame and the end-effector's frame
        @param pitch_offset: Pitch offset between the reference frame and the end-effector's frame
        @param yaw_offset: Yaw offset between the reference frame and the end-effector's frame
        @param finger_length: Length of the fingers of the gripper
        @param reference_frame: String specifying the reference frame of the output message

        @return: PoseStamped message corresponding to the pose of the end-effector or None if no object is detected
    """
    # If not object is detected, return None
    if not object_points.shape[0]:
        return None
    # Compute the height of the segmented table part
    plane_z = np.mean(plane_points[:, 2])
    # Make sure to only consider xyz and not potential color
    object_points = object_points[:, :3]
    # Get the centroid of the object
    centroid = object_points.mean(axis=0)
    # Position of the grasp is aligned to the centroid, with fingers as close as possible to the table
    grasp_position = [centroid[0], centroid[1], plane_z + finger_length + 0.005]
    # Perform PCA on the detected object in order to get it's orientation
    pca = PCA(n_components=3)
    pca.fit(object_points)
    # Orientation of the main axis of the object
    main_axis = pca.components_[0]
    # Get the orientation between the main axis and the X axis
    cos_angle = main_axis.dot([1, 0, 0])
    yaw_angle = np.arccos(cos_angle)
    yaw_angle = -yaw_angle if main_axis[1] < 0 else yaw_angle
    # Apply the yaw offset to the computed yaw angle
    yaw_angle += yaw_offset
    # Define the PoseStamped message
    grasp_pose = PoseStamped()
    grasp_pose.header.frame_id = reference_frame
    grasp_pose.pose.position.x = grasp_position[0]
    grasp_pose.pose.position.y = grasp_position[1]
    grasp_pose.pose.position.z = grasp_position[2]
    quat = quaternion_from_euler(roll_offset, pitch_offset, yaw_angle)
    grasp_pose.pose.orientation.x = quat[0]
    grasp_pose.pose.orientation.y = quat[1]
    grasp_pose.pose.orientation.z = quat[2]
    grasp_pose.pose.orientation.w = quat[3]
    # Return the message
    return grasp_pose


def generate_grasp(pointcloud):
    """
        Function that generates a grasp from a PCL PointCloud. The signature of the function would need to be modified
        if you want to use it with non-default values

        @param: PCL PointCloud to be used as an input of the grasping algorithm

        @return: PoseStamped message corresponding to the generated grasp
    """
    cropped = crop_pointcloud(pointcloud)
    plane, object = plane_segmentation(cropped)
    generated_grasp_pose = get_grasp_pose(object, plane)
    return PoseStamped() if generated_grasp_pose is None else generated_grasp_pose
