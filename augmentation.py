import torch
import time
import cv2
import os
import random
import numpy as np
import json
import pdb

def rotate(image, target, weight):
    '''
    接受一组数据(opencv image + target坐标 + target weight)为参数
    将图片随机旋转-30 ~ +30度，返回新图片、新target坐标、新的target weight
    '''
    h, w = image.shape[:2]

    # 生成随机旋转的角度值，逆时针为正，顺时针为负
    angle = random.random() * 60 - 30
    center_w, center_h = w // 2, h // 2
    M = cv2.getRotationMatrix2D((center_w, center_h), angle, 1)
    image_outp = cv2.warpAffine(image, M, (w, h))

    # 转换每个gt坐标到旋转后的值，并修改weight的状态
    target_outp = target.copy()
    weight_outp = weight.copy()

    # pdb.set_trace()
    target_outp[:, 0] = (M[0, 0] * target[:, 0] * w + M[0, 1] * target[:, 1] * h + M[0, 2]) / w
    target_outp[:, 1] = (M[1, 0] * target[:, 0] * w + M[1, 1] * target[:, 1] * h + M[1, 2]) / h

    weight_outp[target_outp[:, 0] < 0] = 0 
    weight_outp[target_outp[:, 1] < 0] = 0 
    weight_outp[target_outp[:, 0] > 1] = 0 
    weight_outp[target_outp[:, 1] > 1] = 0 

    return image_outp, target_outp, weight_outp

def flip(image, target, weight):
    '''
    接受一组数据(opencv image + target坐标 + target weight)为参数
    将图片左右翻转，返回新图片、新target坐标、新的target weight
    '''
    h, w = image.shape[:2]

    # 图片左右翻转
    image_outp = cv2.flip(image, 1)

    # weight值原样保留
    weight_outp = weight.copy()
    weight_outp[2:] = weight_outp[-1:1:-1].copy()

    # 转换每个gt坐标到翻转后的值
    target_outp = target.copy()
    target_outp[:, 0] = 1 - target[:, 0]
    target_outp[2:] = target_outp[-1:1:-1].copy()

    return image_outp, target_outp, weight_outp

def crop(image, target, weight, box):
    '''
    接受一组数据(opencv image + target坐标 + target weight + boundingbox坐标)为参数
    依据boundingbox坐标信息重新生成一个相近的box（整体缩放比例0.75~1.25，各方向的截取/增益程度也根据对应轴长随机）
    box的数据格式：{'y': ..., 'x': ..., 'w': ..., 'h': ...}
    '''
    # 生成截取比例
    h_factor = random.random() * 0.5 + 0.75
    w_factor = random.random() * 0.5 + 0.75
    new_h = box['h'] * h_factor
    new_w = box['w'] * w_factor

    y_factor = random.random() * 0.15 - 0.05 if h_factor < 1 \
          else random.random() * 0.15 - 0.1
    new_y = box['y'] + y_factor
    new_u = np.clip(new_y, 0, 1)
    new_d = np.clip(new_y + new_h, 0, 1)
    new_h = new_d - new_u

    x_factor = random.random() * 0.15 - 0.05 if w_factor < 1 \
          else random.random() * 0.15 - 0.1
    new_x = box['x'] + x_factor
    new_l = np.clip(new_x, 0, 1)
    new_r = np.clip(new_x + new_w, 0, 1)
    new_w = new_r - new_l

    h, w = image.shape[:2]
    
    # 确定上界位置、进行截取和resize
    ubound = int(new_u * h)
    dbound = int(new_d * h)
    lbound = int(new_l * w)
    rbound = int(new_r * w)
    
    image_outp = image[ubound:dbound, lbound:rbound, :]
    try:
        image_outp = cv2.resize(image_outp, (192, 256))
    except cv2.error:
        image_outp = np.zeros((256, 192, 3))
    # 转换每个gt坐标到截取后的值，并修改weight的状态
    target_outp = target.copy()
    if new_h == 0 or new_w == 0:
        image = cv2.resize(image, (192, 256))
        return image, target, weight
    
    target_outp[:, 0] = (target[:, 0] - new_l) / new_w
    target_outp[:, 1] = (target[:, 1] - new_u) / new_h
    weight_outp = weight.copy()
    weight_outp[target_outp[:, 0] < 0] = 0 
    weight_outp[target_outp[:, 1] < 0] = 0 
    weight_outp[target_outp[:, 0] > 1] = 0 
    weight_outp[target_outp[:, 1] > 1] = 0 

    return image_outp, target_outp, weight_outp

def brightness_contrast(image, target, weight):
    '''
    接受一组数据(opencv image + target坐标 + target weight)为参数
    通过线性变换'y = ax + b'修改图片的亮度和对比度，返回新图片、新target坐标、新的target weight
    '''
    target_outp = target.copy()
    weight_outp = weight.copy()
    image_outp = image.copy()

    # 对比度因子（0.5~2）
    a = random.random() * 1.5 + 0.5

    # 亮度因子（-50 ~ +50）
    b = random.random() * 100 - 50

    # 逐像素修改像素值
    image_outp = np.clip(image * a + b, 0, 255).astype(np.uint8)

    return image_outp, target_outp, weight_outp

def hue_saturation(image, target, weight):
    '''
    接受一组数据(opencv image + target坐标 + target weight)为参数
    随机调整图片的色相（+0~360°）、饱和度和明度（*0.9~1.1 ± 0~0.1）
    返回新图片、新target坐标、新的target weight
    '''
    target_outp = target.copy()
    weight_outp = weight.copy()

    # 图像归一化，转为HLS颜色空间
    fimage = image.astype(np.float32) / 255.0
    hlsimage = cv2.cvtColor(fimage, cv2.COLOR_BGR2HLS)

    # 色相、饱和度、明度随机变换参数
    H = random.random() * 360
    L_a = random.random() * 0.2 + 0.9
    L_b = random.random() * 0.2 - 0.1
    S_a = random.random() * 0.2 + 0.9
    S_b = random.random() * 0.2 - 0.1
    
    # 依据变换参数将变换应用于图像，clip取值
    hlsimage[:, :, 0] += H
    hlsimage[:, :, 0][hlsimage[:, :, 0] >= 360] -= 360
    hlsimage[:, :, 1] = np.clip(hlsimage[:, :, 1] * L_a + L_b, 0, 1)
    hlsimage[:, :, 2] = np.clip(hlsimage[:, :, 2] * S_a + S_b, 0, 1)

    # 转回BGR颜色空间
    image_outp = (cv2.cvtColor(hlsimage, cv2.COLOR_HLS2BGR) * 255).astype(np.uint8)

    return image_outp, target_outp, weight_outp

def augmentation(image_inp, target_inp, weight_inp, bbox, aug=True):
    # pdb.set_trace()
    h, w = image_inp.shape[:2]
    bc_factor = random.random()
    hs_factor = random.random()
    rotate_factor = random.random()
    crop_factor = random.random()
    flip_factor = random.random()
    image = image_inp.copy()
    target = target_inp.copy()
    weight = weight_inp.copy()

    if crop_factor < 0.5 and aug:
        # e = time.time()
        image, target, weight = crop(image, target, weight, bbox)
        # pdb.set_trace()
        # print("C1: \t\t\t%.3fs"%(time.time() - e))
    else:
        # TODO: img crop 
        # e = time.time()
        # pdb.set_trace()
        lbound = max(bbox['x'], 0)
        ubound = max(bbox['y'], 0)
        rbound = min((bbox['x'] + bbox['w']), w)
        dbound = min((bbox['y'] + bbox['h']), h)
        lb = int(lbound * w)
        ub = int(ubound * h)
        rb = int(rbound * w)
        db = int(dbound * h)
        image = image[ub:db, lb:rb, :]
        image = cv2.resize(image, (192, 256))
        target[:, 0] = (target[:, 0] - lbound) / (rbound - lbound)
        target[:, 1] = (target[:, 1] - ubound) / (dbound - ubound)
        # pdb.set_trace()
        # print("C2: \t\t\t%.3fs"%(time.time() - e))
    if bc_factor < 0.5 and aug:
        # e = time.time()
        image, target, weight = brightness_contrast(image, target, weight)
        # pdb.set_trace()
        # print("BC: \t\t\t%.3fs"%(time.time() - e))
    if hs_factor < 0.5 and aug:
        # e = time.time()
        image, target, weight = hue_saturation(image, target, weight)
        # pdb.set_trace()
        # print("HS: \t\t\t%.3fs"%(time.time() - e))
    if rotate_factor < 0.5 and aug:
        # e = time.time()
        image, target, weight = rotate(image, target, weight)
        # pdb.set_trace()
        # print("RT: \t\t\t%.3fs"%(time.time() - e))
    # if flip_factor < 0.5 and aug:
    #     # e = time.time()
    #     image, target, weight = flip(image, target, weight)
    #     # pdb.set_trace()
    #     # print("FP: \t\t\t%.3fs"%(time.time() - e))
    if np.sum(weight) == 0:
        image, target, weight = augmentation(image_inp, target_inp, weight_inp, bbox, aug=False)
    return image, target, weight


if __name__ == '__main__':
    iroot = '../contour_data/images/train/'
    aroot = '../contour_data/annotations/train/'

    convert = [1, 1, 0]

    for i, iname in enumerate(os.listdir(iroot)):
        if i == 30:
            break

        ipath = iroot + iname
        apath = aroot + iname.rstrip('.jpg') + '.json'
        
        I = cv2.imread(ipath)
        anno = json.load(open(apath, 'r'))
        target = np.zeros((65, 2))
        weight = np.zeros((65, 1))

        flag = anno['flag'].split(' ')
        flag = [int(num) for num in flag]
        for j in range(65):
            weight[j, 0] = convert[flag[j]]
        
        coordinate = anno['coordinate'].split(' ')
        coordinate = [float(num) for num in coordinate]
        target[:, :] = np.array(coordinate).reshape(65, 2)

        h, w = I.shape[:2]
        bbox = anno['human_box']
        ul_point = (int(bbox['x'] * w), int(bbox['y'] * h))
        lr_point = (int((bbox['x'] + bbox['w']) * w), int((bbox['y'] + bbox['h']) * h))
        
        std_show = I.copy()
        cv2.rectangle(std_show, ul_point, lr_point, (0, 255, 0), 5)

        for j in range(65):
            pt_x = int(target[j, 0] * w)
            pt_y = int(target[j, 1] * h)
            cv2.circle(std_show, (pt_x, pt_y), 2, (255, 0, 0), 2)

        cv2.imwrite('augtest_output/' + str(i) + '_std.jpg', std_show)

        # image, target, weight = crop(I, target, weight, bbox)
        # image, target, weight = crop(I, target, weight, bbox)
        # image, target, weight = crop(I, target, weight, bbox)
        # image, target, weight = crop(I, target, weight, bbox)
        # image, target, weight, bbox_outp = rotate(I, target, weight)
        image, target, weight = augmentation(I, target, weight, bbox)

        for j in range(65):
            try:
                pt_x = int(target[j, 0] * 192)
                pt_y = int(target[j, 1] * 256)
                cv2.circle(image, (pt_x, pt_y), 2, (255, 0, 0), 2)
            except:
                print(j)
        
        # ul_point = (int(bbox_outp['x'] * w), int(bbox_outp['y'] * h))
        # lr_point = (int((bbox_outp['x'] + bbox_outp['w']) * w), int((bbox_outp['y'] + bbox_outp['h']) * h))
        # cv2.rectangle(image, ul_point, lr_point, (0, 255, 0), 5)

        # cv2.imwrite('augtest_output/' + str(i) + '_crop.jpg', image)
        