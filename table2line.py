import pandas as pd
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, sample
import matplotlib
from PIL import Image, ImageDraw
import random 
import matplotlib.colors as mcolors
from utils import visual_bbox, getIndexes
from collections import Counter
from matplotlib.ticker import FormatStrFormatter

def get_Condition(table, question, answer):
    table_ = table.copy()
    try: #is the answer included in the table?
        arow, acol = getIndexes(table, answer)
    except:
        return 1, None
    cnt = 0
    for i, col in enumerate(table.columns):
        try: 
            table[col] = table[col].astype(float)
            cnt+=1
        except:
            continue
    if cnt == 0:
        return 2, None
    try: #can the answer be extracted as num value? 
        answer = float(answer)
    except:
        return 3, None
    try: #is the answer column available?
        table[acol] = table[acol].astype(float)
    except:
        return 4, None
    #does the answer column not informative?
    v = np.diff(table[acol])
    if len(Counter(v).values()) == 1:
        return 5, None
    #find possible column 2
    try: 
        arows = [str(i) for i in table.iloc[arow]] 
        acols = [i for i in arows if i in question]
        _, acol2 = getIndexes(table_, acols[0])
    except:
        return 6, None 
    return 0, (acol, acol2)

def Table2Line(table, index, question, answer, file_names=("line", "line_bbox")):
    #-------------------------Example-------------------------#
    error, _ = get_Condition(table.copy(), question, answer)
    if error > 0:
        return error, None 

    acol, acol2 = _
    value_y = []
    value_y_cols = []
    value_y = list(table[acol].astype(float).values)
    value_y_cols = acol
    value_x = list(table[acol2].values)
    value_x_cols = acol2
    # cand = [j for j in list(table.columns) if j!=acol]
    # rand_col = random.choice(cand)
    # value_x = list(table[rand_col].values)
    # value_x_cols = rand_col

    print(value_y, value_x)
    print(question, answer)
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.clf()
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    # value_x = np.arange(len(value_y))

    labels = [value_y_cols]
    cm = plt.get_cmap('gist_rainbow')
    color_list = cm(random.uniform(0,1)) 
    plt.style.use(sample(plt.style.available, 1))

    plt.plot(np.arange(len(value_x)), value_y, color = color_list)
    plt.xlabel(value_x_cols)
    plt.ylabel(value_y_cols)
    # plt.xticks(np.arange(len(value_x)), labels=value_x) 
    plt.xticks(np.arange(len(value_x)), labels=value_x) 
    y_tick = np.arange(min(value_y), max(value_y)+(max(value_y)-min(value_y))/10, (max(value_y)-min(value_y))/10)
    plt.yticks(y_tick, labels=[str(y) for y in y_tick]) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    p_text = []
    for x, y in zip(np.arange(len(value_x)), value_y):
        p_text.append(plt.text(x, y, f'{y:.2f}'))
    
    handles = [plt.Rectangle((0,0), 0, 0, color=color_list) for i, label in enumerate(labels)]
    plt.legend(handles, labels)

    plt.tight_layout()
    fig_name = f'./{file_names[0]}/{file_names[0]}_{index}.png'
    bbfig_name = f'./{file_names[1]}/{file_names[1]}_{index}.png'
    # plt.savefig(fig_name, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.savefig(fig_name)

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt_size = bbox.width*fig.dpi, bbox.height*fig.dpi
    # print(bbox.width, bbox.width, fig.dpi)
    # print(plt_size)
    plt_dict = {
        "plt_size": plt_size,
        "p_text": p_text,
        "x_value": value_x,
        "y_value": value_y,
        "x_label": value_x_cols,
        "y_label": value_y_cols,
        "y_tick": y_tick,
    }
    bbox_list = Line_bbox(ax, **plt_dict)
    visual_bbox(bbox_list, fig_name, bbfig_name)
    fig.clf()
    plt.close()
    return 0, (fig_name, bbfig_name, bbox_list)

def Line_bbox(ax, **kwargs): 
    #-------------------------Example-------------------------#
    bbox_list=[]
    width, height = kwargs["plt_size"]
    p_text = kwargs["p_text"]
    value_x = kwargs["x_value"]
    value_y = kwargs["y_value"]
    x_label = kwargs["x_label"]
    y_label = kwargs["y_label"]
    y_tick = kwargs["y_tick"]

    #xlabel
    box = np.array(ax.get_xaxis().get_label().get_window_extent().get_points())
    bbox_list.append(("x_label", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(x_label)))

    #ylabel
    box = np.array(ax.get_yaxis().get_label().get_window_extent().get_points())
    bbox_list.append(("y_label", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(y_label)))

    #xticklabel
    for i, l in enumerate(ax.get_xticklabels()):
        box = np.array(l.get_window_extent())
        bbox_list.append(("x_tick", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(value_x[i])))

    #yticklabel
    for i, l in enumerate(ax.get_yticklabels()):
        box = np.array(l.get_window_extent())
        bbox_list.append(("y_tick", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(y_tick[i])))

    #line value
    for i, l in enumerate(p_text):
        box = np.array(l.get_window_extent().get_points())
        bbox_list.append(("val", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(l.get_text())))

    #visual legend
    for i, patch in enumerate(ax.get_legend().get_patches()):
        box = np.array(patch.get_window_extent().get_points())
        bbox_list.append(("v_legend", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], None))

    #text legend
    for i, label in enumerate(ax.get_legend().get_texts()):
        box = np.array(label.get_window_extent().get_points())
        bbox_list.append(("t_legend", [box[0][0], height-box[0][1], box[1][0], height-box[1][1]], str(label.get_text())))

    x = np.arange(len(value_x))
    y = value_y
    for index, (i, j) in enumerate(zip(x, y)): #sub lines
        xmin, ymin = ax.transData.transform((i, j))
        xmax, ymax = ax.transData.transform((x[index+1], y[index+1]))
        bbox_list.append(("line_0", [xmin, height-ymin, xmax, height-ymax], None))
        if index == len(x)-2: break

    return bbox_list

