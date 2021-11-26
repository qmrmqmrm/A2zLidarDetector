import numpy as np
import os
import glob
import json
import pprint

EPSILON = 1.0e-10


def get_calibration(root_path):
    config = load_json(root_path)
    camera_view = config['cameras']['front_center']['view']
    lidar_view = config['lidars']['front_center']['view']
    rot = get_rot_to_global(camera_view)
    transform_V2C = transform_from_to(lidar_view, camera_view)
    transform_C2V = transform_from_to(camera_view, lidar_view)
    transform_V = get_transform_to_global(lidar_view)
    transform_C = get_transform_to_global(camera_view)
    calibriation_dict = {"R0": rot, "V2C": transform_V2C, "C2V": transform_C2V, "C":transform_C, "V" : transform_V}
    return calibriation_dict


def load_json(root_path):
    file_name = os.path.join(root_path, 'camera_lidars.json')
    with open(file_name, 'r') as f:
        config = json.load(f)
    return config


def get_rot_to_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = transform_to_global[0:3, 0:3]

    return rot


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), get_transform_to_global(src))
    return transform


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
    return trans


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = view["origin"]
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin
    return transform_to_global


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = np.linalg.norm(x_axis)
    y_axis_norm = np.linalg.norm(y_axis)

    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = np.linalg.norm(y_axis)
    z_axis_norm = np.linalg.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


if __name__ == '__main__':
    root_path = '/media/dolphin/intHDD/a2d2'
    a = get_calibration(root_path)
