import pandas as pd
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import matplotlib
from PIL import Image, ImageDraw
from utils import visual_bbox, save_bbox

def Table2Line(table, index):
    #-------------------------Example-------------------------#
    value_y = []
    value_y_cols = []
    for i, col in enumerate(table.columns):
        try: 
            table[col] = table[col].astype(int)
            y = list(table[col].values)
            value_y.append(y)
            value_y_cols.append(i)
        except:
            continue
    
    matplotlib.rcParams['font.family'] ='Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] =False

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots()
    value_y = value_y[0]
    value_x = np.arange(len(value_y))
    tick_y = np.arange(min(value_y), max(value_y), int((max(value_y)*1.1-min(value_y))/10))

    p = plt.plot(value_x, value_y)
    plt.xlabel(table.columns[randrange(len(table.columns))])
    plt.ylabel(table.columns[value_y_cols[0]])
    plt.xticks(value_x) 
    plt.yticks(tick_y) 
    p_text = []
    for x, y in zip(value_x, value_y):
        p_text.append(plt.text(x, y, f'{y:.2f}'))

    plt.tight_layout()
    fig_name = f'./line/line_{index}.png'
    # plt.savefig(fig_name, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_name)

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt_size = bbox.width*fig.dpi, bbox.height*fig.dpi

    plt_dict = {
        "plt_size": plt_size,
        "p_text": p_text,
    }
    bbox_list = Line_bbox(plt, ax, **plt_dict)
    visual_bbox(fig_name, bbox_list)
    save_bbox(bbox_list)

    return

def Line_bbox(plt, ax, **kwargs): 
    #-------------------------Example-------------------------#
    bbox_list=[]
    width, height = kwargs["plt_size"]
    p_text = kwargs["p_text"]

    bbox_theme = []
    for i, l in enumerate(ax.get_xticklabels()):
        box = np.array(l.get_window_extent())
        bbox_theme.append([box[0][0], height-box[0][1], box[1][0], height-box[1][1]])
    bbox_list.append(bbox_theme)

    bbox_theme = []
    for i, l in enumerate(ax.get_yticklabels()):
        box = np.array(l.get_window_extent())
        bbox_theme.append([box[0][0], height-box[0][1], box[1][0], height-box[1][1]])
    bbox_list.append(bbox_theme)

    bbox_theme = []
    for i, l in enumerate(p_text):
        box = np.array(l.get_window_extent().get_points())
        bbox_theme.append([box[0][0], height-box[0][1], box[1][0], height-box[1][1]])
    bbox_list.append(bbox_theme)

    return bbox_list



