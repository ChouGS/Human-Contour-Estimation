import json
import os

print(len(os.listdir('/root/group-incubation-bj/contour/contour_data_res/annotations/train')))
print(len(os.listdir('/root/group-incubation-bj/contour/contour_data_res/annotations/val')))
exit()
for anno in os.listdir('/root/group-incubation-bj/contour/contour_data_res/annotations_resized/val'):
    an = json.load(open(os.path.join('/root/group-incubation-bj/contour/contour_data_res/annotations_resized/val', anno), 'r'))
    an_coo = list(map(int, an['flag'].split(' ')))
    if 3 in an_coo:
        print("!")
        assert 0 not in an_coo
        an_c = [i - 1 for i in an_coo]
        an['flag'] = an_c
    json.dump(an, open(os.path.join('/root/group-incubation-bj/contour/contour_data_res/annotations_resized/val', anno), 'w'))
