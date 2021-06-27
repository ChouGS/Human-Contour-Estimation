import cv2
import numpy as np
import json
import os

canvas = np.ones((432, 240, 3), dtype='uint8') * 255
canvas[36:396, 24:216, :] = 255
anno_list = os.listdir('contour_data/annotations/train')
anno_list = map(lambda x: os.path.join('contour_data/annotations/train', x), anno_list)

canvas1 = np.ones((432, 240, 3), dtype='uint8') * 255
canvas2 = np.ones((432, 240, 3), dtype='uint8') * 255
canvas2[36:396, 24:216, :] = 0

for i, anno_path in enumerate(anno_list):
    if i < 835:
        continue
    B = np.random.randint(0, 256)
    G = np.random.randint(0, 256)
    R = np.random.randint(0, 256)
    # img_path = ('contour_data/images/train/' + anno_path.split('/')[-1]).rstrip('.json') + '.jpg'
    # I = cv2.imread(img_path)
    # print(img_path)
    # I = cv2.resize(I, (192, 360))
    # canvas1[36:396, 24:216, :] = I

    anno = json.load(open(anno_path, 'r'))
    lbound = int(24 + 192 * max(anno['human_box']['x'], 0))
    rbound = int(24 + 192 * min(anno['human_box']['x'] + anno['human_box']['w'], 1))
    ubound = int(36 + 360 * max(anno['human_box']['y'], 0))
    dbound = int(36 + 360 * min(anno['human_box']['y'] + anno['human_box']['h'], 1))
    cv2.rectangle(canvas, (lbound, ubound), (rbound, dbound), (B, G, R), 2)
    # cv2.rectangle(canvas1, (lbound, ubound), (rbound, dbound), (0, 255, 0), 2)
    # cv2.rectangle(canvas2, (lbound, ubound), (rbound, dbound), (0, 255, 0), 2)

cv2.imwrite('bbox_stat.jpg', canvas)
# cv2.imwrite('bbox_standard.jpg', canvas1)
# cv2.imwrite('bbox_sample.jpg', canvas2)
