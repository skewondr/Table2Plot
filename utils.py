import os
import ast
import gzip
import json
import pickle
from IPython import embed

import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from time import time
from collections import defaultdict
from IPython import embed

from PIL import Image, ImageDraw

class BaseDataset:
    def __init__(self, data_dir, data_name):
        """
        Dataset class
        :param str data_dir: base directory of data
        :param str dataset: name of dataset 
        :param int task: number of task 
        """
        self.data_dir = data_dir
        self.data_name = str(data_name)

    def check_dataset_exists(self):
        return os.path.exists(self.docs_data_file)

    def __str__(self):
        return 'BaseDataset'

class Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__(kwargs['data_dir'], kwargs['data_name'])
        print(kwargs)
        self.load_data_task3()

    def load_data_task3(self):
        """
            input: sample_problemsheet.json, sample_answer.json
        """
        print("> Generating train/dev samples for task3")
        eval_train = True
        
        self.train_samples = []
        self.dev_samples = []

        self.data_dir = os.path.join(self.data_dir, self.data_name)

        if self.data_name == 'KorWikiTQ':
            train_path = os.path.join(self.data_dir, 'KorWikiTQ_ko_train.json')
            dev_path = os.path.join(self.data_dir, 'KorWikiTQ_ko_dev.json')
            
            with open(train_path, 'rt', encoding='UTF8') as f:
                train_data = json.load(f)
            with open(dev_path, 'rt', encoding='UTF8') as f:
                dev_data = json.load(f)
            
            train_data = train_data['data']
            dev_data = dev_data['data']

            for prob in train_data:
                emp_dict = {}
                try:
                    emp_dict["question"] = prob["QAS"]["question"]
                    emp_dict["answer"] = prob["QAS"]["answer"]
                    emp_dict["table"] = prob["TBL"]
                    emp_dict["level"] = prob["QAS"]["qid"].split("_")[1]
                except: continue
                # answer_coordinates labeling
                # answer와 exact match가 되는 cell을 찾아서 answer_coordinates labeling하고, 없으면 건너뛰기
                # table_numpy = np.array(prob["TBL"][1:])
                # try:
                #     r_ids, c_ids = np.where(table_numpy == prob["QAS"]["answer"])
                # except:
                #     continue
                # emp_dict["answer_coordinates"] = []
                # for r_id, c_id in zip(r_ids, c_ids):
                #     emp_dict["answer_coordinates"].append((r_id, c_id))

                self.train_samples.append(emp_dict)
            
            for prob in dev_data:
                emp_dict = {}
                try:
                    emp_dict["question"] = prob["QAS"]["question"]
                    emp_dict["answer"] = prob["QAS"]["answer"]
                    emp_dict["table"] = prob["TBL"]
                    emp_dict["level"] = prob["QAS"]["qid"].split("_")[1]
                except: continue
                # answer_coordinates labeling
                # table_numpy = np.array(prob["TBL"][1:])
                # try:
                #     r_ids, c_ids = np.where(table_numpy == prob["QAS"]["answer"])
                # except:
                #     continue
                # emp_dict["answer_coordinates"] = []
                # for r_id, c_id in zip(r_ids, c_ids):
                #     emp_dict["answer_coordinates"].append((r_id, c_id))

                self.dev_samples.append(emp_dict)

        else:
            print('data_name is not valid!')

        print(f"Number of training samples: {len(self.train_samples)}")
        print(f"Number of dev samples: {len(self.dev_samples)}")

        return


import pickle

def save_dict(d, filename):
    with open(f'{filename}.pickle', 'wb') as handle:
        pickle.dump(d, handle)
    return

def load_dict(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        return pickle.load(handle)

# pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
# pip install imgaug

import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import matplotlib.colors as mcolors

def visual_bbox(table_plt_file, bbox_list):
    #-------------------------Example-------------------------#
    color_names = list(mcolors.CSS4_COLORS)
    img = Image.open(table_plt_file).convert('RGB')
    draw = ImageDraw.Draw(img)
    for i, bbox_theme in enumerate(bbox_list):
        for bb in bbox_theme:
            # print((box[0][0], box[0][1], box[1][0], box[1][1]))
            draw.rectangle((bb[0], bb[1], bb[2], bb[3]), outline=color_names[i], width = 5)

    # img.show()
    # img.save(f'./line_bbox/linebbox_{index}.png')
    return 

def save_bbox(bbox_list): #save to json file
    return 