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
import matplotlib.colors as mcolors

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
                    emp_dict["qid"] = prob["QAS"]["qid"]
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
                    emp_dict["qid"] = prob["QAS"]["qid"]
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

def save_pickle(d, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle)
    return

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

# pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
# pip install imgaug
import matplotlib
import random 

def visual_bbox(bbox_list, plt_size, fig_name, bbfig_name):
    #-------------------------Example-------------------------#
    width, height = plt_size
    color_names = list(matplotlib.colors.cnames.keys())
    random.shuffle(color_names)
    # color_names = list(mcolors.CSS4_COLORS)
    img = Image.open(fig_name).convert('RGB')
    draw = ImageDraw.Draw(img)
    box_color = []
    for i, l in enumerate(bbox_list):
        bbox_cls = l[0]
        if bbox_cls not in box_color:
            box_color.append(bbox_cls)
            # print(bbox_cls, color_names[len(box_color)])
        bb = l[1]
        draw.rectangle((bb[0], height-bb[1], bb[2], height-bb[3]), outline=color_names[len(box_color)], width = 5)
    img.save(bbfig_name)
    # img.show()
    # img.save(f'./line_bbox/linebbox_{index}.png')
    return 

# def save_bbox(plot_list, data_mode):
#     if data_mode == "train": 
#         path = './KorWikiTQ/KorWikiTQ_ko_train.pickle'
#         save_path = './KorWikiTQ/new_KorWikiTQ_ko_train.json'
#     else:
#         path = './KorWikiTQ/KorWikiTQ_ko_dev.json'
#         save_path = './KorWikiTQ/new_KorWikiTQ_ko_dev.json'

#     with open(path, 'rt', encoding='UTF8') as f:
#         data = json.load(f)
#         for i, prob in enumerate(data['data']):
#             if i in plot_list:
#                 prob["PNG_PATH"] = plot_list[i][0]
#                 prob["BBPNG_PATH"] = plot_list[i][1]
#                 prob["BB_CLS"] = dict()
#                 for j, (k, v) in enumerate(plot_list[i][2].items()):
#                     prob["BB_CLS"][k] = list()
#                     for jj, box in enumerate(v):
#                         bb_cls_key = f"{k}_{jj}"
#                         bb_cls = (bb_cls_key, box)                    
#                         prob["BB_CLS"][k].append(bb_cls)

#     with open(save_path, 'w', encoding='UTF8') as f:
#         json.dump(data, f, indent="\t")
#     return 

def getIndexes(dfObj, value):
    # Empty list
    listOfPos = []
     
    # isin() method will return a dataframe with
    # boolean values, True at the positions   
    # where element exists
    result = dfObj.isin([value])
     
    # any() method will return
    # a boolean series
    seriesObj = result.any()
 
    # Get list of column names where
    # element exists
    columnNames = list(seriesObj[seriesObj == True].index)
    
    # Iterate over the list of columns and
    # extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
 
        for row in rows:
            listOfPos.append((row, col))
             
    # This list contains a list tuples with
    # the index of element in the dataframe
    return listOfPos[0]
