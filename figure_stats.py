import cv2
import numpy as np
# import pdb
import os
import json


def node_color(i):
    if i <= 1:
        return (0, 0, 0)
    if i <= 15:
        return (0, 0, 255)
    if i <= 22:
        return (0, 255, 0)
    if i <= 32:
        return (0, 125, 255)
    if i == 33:
        return (0, 255, 0)
    if i <= 43:
        return (255, 0, 0)
    if i <= 50:
        return (0, 255, 0)
    return (255, 0, 125)


def edge_color(last_i):
    if last_i == 0:
        return (0, 0, 0)
    if last_i == 1:
        return (255, 255, 255)
    if last_i <= 15:
        return (0, 0, 255)
    if last_i <= 21:
        return (0, 255, 0)
    if last_i <= 32:
        return (0, 125, 255)
    if last_i <= 43:
        return (255, 0, 0)
    if last_i <= 49:
        return (0, 255, 0)
    return (255, 0, 125)


def draw_figure(canvas, coordinates, color):
    filtered_c = coordinates[coordinates[:, 0] >= 0]
    filtered_c = filtered_c[filtered_c[:, 1] >= 0]
    filtered_c = filtered_c[filtered_c[:, 0] <= 1]
    filtered_c = filtered_c[filtered_c[:, 1] <= 1]
    center = np.mean(filtered_c, axis=0)
    offset = np.array([0.5, 0.5]) - center
    filtered_c += offset

    last_i = 0
    for i in range(1, 65):
        if coordinates[i][0] < 0 or coordinates[i][1] < 0:
            continue
        cs = (24 + int(192 * coordinates[i][0]), 36 + int(360 * coordinates[i][1]))
        ce = (24 + int(192 * coordinates[last_i][0]), 36 + int(360 * coordinates[last_i][1]))
        color = edge_color(last_i)
        cv2.line(canvas, cs, ce, color, 1)
        last_i = i

    for i in range(65):
        if coordinates[i][0] < 0 or coordinates[i][1] < 0:
            continue
        coo = (24 + int(192 * coordinates[i][0]), 36 + int(360 * coordinates[i][1]))
        color = node_color(i)
        cv2.circle(canvas, coo, 2, color, 1)


canvas = np.ones((432, 240, 3), dtype='uint8') * 255
anno_list = os.listdir('contour_data/annotations_resized/train')
anno_list = map(lambda x: os.path.join('contour_data/annotations_resized/train', x), anno_list)

for i, anno_path in enumerate(anno_list):
    if i < 100:
        continue
    B = np.random.randint(0, 256)
    G = np.random.randint(0, 256)
    R = np.random.randint(0, 256)
    anno = json.load(open(anno_path, 'r'))
    coordinates = anno['coordinate']
    coordinates = coordinates.split(' ')
    coordinates = list(map(lambda x: float(x), coordinates))
    coordinates = np.array(coordinates).reshape(65, 2)
    # draw_figure(canvas, coordinates, (B, G, R))

    if i == 199:
        draw_figure(canvas, coordinates, (B, G, R))
        cv2.imwrite('check.jpg', canvas)
        image_name = anno_path.split('/')[-1].rstrip('.json') + '.jpg'
        image_name = os.path.join('contour_data/images_resized/train', image_name)
        image = cv2.imread(image_name)
        canvas[36:396, 24:216, :] = cv2.resize(image, (192, 360))
        draw_figure(canvas, coordinates, (B, G, R))
        os.system('cp ' + image_name + ' .')
        # pdb.set_trace()
        break

cv2.imwrite('figure_stat.jpg', canvas)
