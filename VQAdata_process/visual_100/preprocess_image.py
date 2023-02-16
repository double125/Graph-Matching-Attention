#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import base64
import numpy as np
import csv
import sys
import h5py
import pandas as pd
import zarr
from tqdm import tqdm


csv.field_size_limit(sys.maxsize)

def IoU(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoU ratio (intersection over union) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
    return float(intersection) / union

def features_to_zarr(phase):
    FIELDNAMES = ['image_id', 'image_w', 'image_h',
                  'num_boxes', 'boxes', 'features']

    if phase == 'trainval':
        infiles = [
            'visual_100/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
            'visual_100/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
            'visual_100/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1',
            'visual_100/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',
        ]
    elif phase == 'test':
        infiles = [
            'visual_100/test2015/test2015_resnet101_faster_rcnn_genome.tsv',
        ]
    else:
        raise SystemExit('Unrecognised phase')

    # Read the tsv and append to files
    boxes = zarr.open_group(phase + '_boxes.zarr', mode='w')
    features = zarr.open_group(phase + '.zarr', mode='w')
    image_size = {}
    visual_graph = {}
    for infile in infiles:
        with open(infile, "r") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            print('Converting ' + infile + ' to zarr...')
            for item in tqdm(reader):
                item['image_id'] = str(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    encoded_str = base64.decodestring(
                        item[field].encode('utf-8'))
                    item[field] = np.frombuffer(encoded_str,
                                                dtype=np.float32).reshape((item['num_boxes'], -1))

                num_boxes = item['num_boxes']
                boxes_ = item['boxes']

                vg_edge = np.zeros([100, 100])
                for i in range(1, num_boxes):
                    for j in range(i):
                        box1 = boxes_[i]
                        box2 = boxes_[j]
                        if IoU(box1, box2) > 0.3:
                            vg_edge[i][j] = vg_edge[j][i] = 1
                visual_graph[item['image_id']] = vg_edge

                # append to zarr files
                boxes.create_dataset(item['image_id'], data=item['boxes'])
                features.create_dataset(item['image_id'], data=item['features'])
                # image_size dict
                image_size[item['image_id']] = {
                    'image_h':item['image_h'],
                    'image_w':item['image_w'],
                }
    pickle.dump(visual_graph, open('../VQA/visual_graph/vg_100/' + phase +'_visual_graph_1_03.pkl', 'w'))


    # convert dict to pandas dataframe
    

    # create image sizes csv
    print('Writing image sizes csv...')
    df = pd.DataFrame.from_dict(image_size)
    df = df.transpose()
    d = df.to_dict()
    dw = d['image_w']
    dh = d['image_h']
    d = [dw, dh]
    dwh = {}
    for k in dw.keys():
        dwh[k] = np.array([d0[k] for d0 in d])
    image_sizes = pd.DataFrame(dwh)
    image_sizes.to_csv(phase + '_image_size.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 image data')
    parser.add_argument('--data', nargs='+', help='trainval, and/or test, list of data phases to be processed', required=True)
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data

    save_dir = '../VQA/visual_graph/vg_100/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for phase in phase_list:
        # First download and extract
        features_to_zarr(phase)

    print('Done')
