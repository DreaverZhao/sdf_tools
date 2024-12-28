#!/usr/bin/python3
"""
creating a 3d random forest
computing the sdf and sdf gradient
save to file
"""
import ros_numpy
import rospy
import numpy as np

from geometry_msgs.msg import Point
from sdf_tools.utils_3d import compute_sdf_and_gradient
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker


def create_point_cloud():
    rng = np.random.RandomState(0)
    random_tree_num = 10
    points_per_tree = 50
    tree_points = []
    for i in range(random_tree_num):
        # randomly generate root poses on xy plane
        root_pose = rng.uniform([-10.0, -10.0, 0.0], [60.0, 10.0, 0.0], [1, 3])
        # randomly generate tree height
        tree_height = rng.uniform(1.0, 10.0)
        # randomly generate tree radius
        tree_radius = rng.uniform(0.1, 0.5)
        # generate tree points
        for j in range(points_per_tree):
            # randomly generate tree points
            tree_point = rng.uniform([-tree_radius, -tree_radius, -10.0], [tree_radius, tree_radius, tree_height], [1, 3])
            tree_points.append(root_pose + tree_point)
    return np.concatenate(tree_points, axis=0)


def point_cloud_to_voxel_grid(pc: np.ndarray, shape, res, origin_point):
    vg = np.zeros(shape, dtype=np.float32)
    indices = ((pc - origin_point) / res).astype(np.int64)
    rows = indices[:, 0]
    cols = indices[:, 1]
    channels = indices[:, 2]
    vg[rows, cols, channels] = 1.0
    return vg


def compute_param(res, points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])
    shape = [int((max_x - min_x) / res) + 1, int((max_y - min_y) / res) + 1, int((max_z - min_z) / res) + 1]
    return shape, np.array([min_x, min_y, min_z], dtype=np.float32)


def visualize_point_cloud(pub: rospy.Publisher, pc: np.ndarray):
    list_of_tuples = [tuple(point) for point in pc]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id='world', stamp=rospy.Time.now())
    pub.publish(msg)


def visualize_sdf(pub, sdf: np.ndarray, shape, res, origin_point):
    points = get_grid_points(origin_point, res, shape)
    list_of_tuples = []
    for x in range(shape[1]):
        for y in range(shape[0]):
            for z in range(shape[2]):
                list_of_tuples.append((points[x, y, z, 0], points[x, y, z, 1], points[x, y, z, 2], sdf[y, x, z]))
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('distance', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id='world', stamp=rospy.Time.now())
    pub.publish(msg)


def get_grid_points(origin_point, res, shape):
    indices = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.stack(indices, axis=-1)
    points = (indices * res) + origin_point
    return points


def rviz_arrow(position: np.ndarray, target_position: np.ndarray, label: str = 'arrow', **kwargs):
    idx = kwargs.get("idx", 0)

    arrow = Marker()
    arrow.action = Marker.ADD  # create or modify
    arrow.type = Marker.ARROW
    arrow.header.frame_id = "world"
    arrow.header.stamp = rospy.Time.now()
    arrow.ns = label
    arrow.id = idx

    arrow.scale.x = kwargs.get('scale', 1.0) * 0.0025
    arrow.scale.y = kwargs.get('scale', 1.0) * 0.004
    arrow.scale.z = kwargs.get('scale', 1.0) * 0.006

    arrow.pose.orientation.w = 1

    start = Point()
    start.x = position[0]
    start.y = position[1]
    start.z = position[2]
    end = Point()
    end.x = target_position[0]
    end.y = target_position[1]
    end.z = target_position[2]
    arrow.points.append(start)
    arrow.points.append(end)

    arrow.color.a = 1

    return arrow


def plot_arrows_rviz(pub, positions, directions):
    msg = MarkerArray()
    for i, (position, direction) in enumerate(zip(positions, directions)):
        msg.markers.append(rviz_arrow(position, position + direction, frame_id='world', idx=i, label='sdf_grad'))
    pub.publish(msg)


def main():
    rospy.init_node("random_forest_sdf")

    pc_pub = rospy.Publisher("points", PointCloud2, queue_size=10)
    sdf_pub = rospy.Publisher("sdf", PointCloud2, queue_size=10)
    sdf_grad_pub = rospy.Publisher("sdf_grad", MarkerArray, queue_size=10)

    rospy.sleep(0.1)

    pc = create_point_cloud()

    res = 0.2
    # shape, origin_point = compute_param(res, pc)
    origin_point = np.array([-10.0, -12.0, -10.0], dtype=np.float32)
    shape = [350, 240, 110]
    vg = point_cloud_to_voxel_grid(pc, shape, res, origin_point)
    rospy.loginfo("Computing sdf...")
    sdf, sdf_grad = compute_sdf_and_gradient(vg, res, origin_point)

    rospy.loginfo("Saving sdf to file...")
    prifix = "/home/dreaver/AcroflightDiffusionModel/save/"
    np.save(prifix + "sdf_param.npy", np.asarray([res, origin_point[0], origin_point[1], origin_point[2], shape[0], shape[1], shape[2]]))
    np.save(prifix + "sdf.npy", sdf)
    np.save(prifix + "sdf_grad.npy", sdf_grad)
    np.save(prifix + "pc.npy", pc)

    # grid_points = get_grid_points(origin_point, res, shape)
    # subsample = 8
    # grad_scale = 0.02

    # rospy.loginfo("Visualize sdf...")

    # while not rospy.is_shutdown():
    #     visualize_point_cloud(pc_pub, pc)
    #     visualize_sdf(sdf_pub, sdf, shape, res, origin_point)
    #     plot_arrows_rviz(sdf_grad_pub, grid_points.reshape([-1, 3])[::subsample], sdf_grad.reshape([-1, 3])[::subsample] * grad_scale)
    #     rospy.sleep(0.1)


if __name__ == '__main__':
    main()
