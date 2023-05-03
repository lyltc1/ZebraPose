import numpy as np
from math import ceil
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt


def fibonacci_sphere(samples=1000):
    points = np.empty((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - z * z)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points[i] = np.asarray((x, y, z))
    return points


def calculate_J(point_to_code):
    code = np.asarray([point_to_code[i] for i in range(samples)])
    bit = ceil(np.log2(samples))
    code_distance = np.empty((samples, bit))
    for i in reversed(range(bit)):
        code_distance[:, i] = code % 2
        code = code // 2
    code_distance = np.logical_xor(code_distance[:, np.newaxis], code_distance[np.newaxis, :])
    J = np.sum(code_distance[index] != 0)
    return J


if __name__ == '__main__':
    ## args
    visualize = False
    samples = 20
    ##
    bit = np.ceil(np.log2(samples))
    points = fibonacci_sphere(samples)
    if visualize:
        print("points:", points)
        picture = plt.figure().add_subplot(111, projection='3d')
        picture = picture.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()
    point_to_code = dict()
    for i in range(samples):
        point_to_code[i] = i
    ## calculate index
    distance = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance = np.sum(np.square(distance), axis=2)
    threshold = (4 * 0.5 / (samples - 1), 4 * 3.5 / (samples - 1))
    index = np.where(np.logical_and(distance > threshold[0], distance < threshold[1]))
    print(f"index:{index}")
    min_J = index[0].size
    ## calculate J based on the distance between points
    J = calculate_J(point_to_code)
    print(J)
    print(f"point_to_code:", point_to_code)
    iter = 1
    while 1:
        # 有几个下标需要互换
        num = np.random.randint(2, 8)
        tmp_change_index = np.random.randint(1, samples - 1, size=num)
        tmp_point_to_code = point_to_code.copy()
        tmp_value = tmp_point_to_code[tmp_change_index[0]]
        for i in range(num - 1):
            tmp_point_to_code[tmp_change_index[i]] = tmp_point_to_code[tmp_change_index[i + 1]]
        tmp_point_to_code[tmp_change_index[num - 1]] = tmp_value

        tmp_J = calculate_J(tmp_point_to_code)
        if tmp_J < J:
            point_to_code = tmp_point_to_code.copy()
            J = tmp_J
            print(f"iter: {iter}, J:{J}, minJ:{min_J}")
            print(f"change index:{tmp_change_index}")
        iter = iter + 1
