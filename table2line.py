import pandas as pd
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, sample, choices
import matplotlib
from PIL import Image, ImageDraw
import random 
import matplotlib.colors as mcolors
from utils import visual_bbox, getIndexes
from collections import Counter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import re

def get_Condition(table, question, answer):
    table_ = table.copy()
    try: #is the answer included in the table?
        arow, acol = getIndexes(table, answer)
    except:
        return 1, None

    try: #can the answer be extracted as num value? 
        answer = float(answer)
    except:
        return 3, None
    

    try: #is the answer column available?
        # table[acol] = [re.sub(r'[^0-9.]', '', a) for a in table[acol]]
        table[acol] = table[acol].astype(float)
    except:
        # print(table[acol])
        return 4, None

    # cnt = 0
    # for i, col in enumerate(table.columns):
    #     try: 
    #         table[col] = table[col].astype(float)
    #         cnt+=1
    #     except:
    #         continue
    # if cnt == 0:
    #     return 2, None

    try: #does the answer column not informative?
        v = np.diff(table[acol])
        if len(Counter(v).values()) == 1:
            return 5, None
    except:
        return 5, None
    #find possible column 2
    try: 
        arows = [str(i) for i in table.iloc[arow]] 
        acols = [i for i in arows if i in question]
        _, acol2 = getIndexes(table_, acols[0])
    except:
        return 6, None
    return 0, (acol, acol2)

def Table2Line(table, index, proxy_n, question, answer, file_names=("line", "line_bbox")):
    #-------------------------Example-------------------------#
    error, _ = get_Condition(table.copy(), question, answer)
    if error > 0:
        return error, None 
    # return error, (None, None, None) 

    acol, acol2 = _
    cand = [j for j in list(table.columns) if j!=acol and j!=acol2]
    for i in range(20):
        cand.append(f"수치 {i}")
    value_y = []
    table[acol] = [re.sub(r'[^0-9.]', '', a) for a in table[acol]]
    value_y = list(table[acol].astype(float).values)
    a = []
    for v in value_y:
        if  str(v)[-2:]==".0": 
            a.append(int(re.sub(".0$", "", str(v))))
        else :
            a.append(round(v,2))
    value_y = a
    value_y_cols = acol
    cand.pop(0)

    value_x = list(table[acol2].values)
    try:
        value_x = [x.replace(" ", "\n") for x in value_x]
    except: 
        return 7, None 

    value_x_cols = acol2
    # cand = [j for j in list(table.columns) if j!=acol]
    # rand_col = random.choice(cand)
    # value_x = list(table[rand_col].values)
    # value_x_cols = rand_col

    # print(value_y, value_x)
    # print(question, answer)
    ff = [(f.name, f.fname) for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["figure.figsize"] = (7,4)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['font.family'] = 'NanumGothic'
    BIGGER_SIZE = 5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12.
    plt.rcParams['xtick.labelsize'] = 12.
    plt.rcParams['ytick.labelsize'] = 12.
    plt.rcParams['axes.labelsize'] = 15.

    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.clf()
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['savefig.dpi'] = 200
    fig, ax = plt.subplots()
    # value_x = np.arange(len(value_y))
    #-----------------------------------------------------------#
    #                         GET VALUE                         #
    #-----------------------------------------------------------#
    #select 1~2 line proxy plot 
    # proxy_n = sample([0, 1, 2, 3], 1)[0]
    value_y_list = []  
    headers = [] 
    y_loc, proxy_min = 0, min(value_y)
    if proxy_n > 0:
        for i in range(proxy_n):
            proxy_value_y = np.array(value_y) + sample([-1, 1], 1)[0]*((max(value_y)-min(value_y))/3)*(i+1)**2
            for j in range(len(value_y)):
                proxy_value_y[j] += random.uniform(-2*((max(value_y)-min(value_y))/10),2*(max(value_y)-min(value_y))/10)
                proxy_value_y = [round(y,2) for y in proxy_value_y]
            value_y_list.append(proxy_value_y)
            headers.append(cand[0])
            cand.pop(0)
            if min(proxy_value_y) < proxy_min:
                proxy_min = min(proxy_value_y)
                y_loc += 1
            # print(proxy_n , i, value_y_list, headers)
    value_y_list.insert(y_loc, value_y)
    headers.insert(y_loc, acol)
    # print(y_loc, value_y_list, headers)
    flatten = [item for sublist in value_y_list for item in sublist]
    min_y, max_y = min(flatten), max(flatten)

    #-----------------------------------------------------------#
    #                        PLOT STYLE                         #
    #-----------------------------------------------------------#
    lines_array = list(matplotlib.lines.lineStyles.keys())
    markers_array = list(matplotlib.markers.MarkerStyle.markers.keys())
    l, m = np.random.choice(lines_array[:4], size=len(value_y_list), replace=True), sample(markers_array[:7], len(value_y_list))
    cm = plt.get_cmap('gist_rainbow')
    c = [cm(random.uniform(0,1)) for i in range(len(value_y_list))]
    
    
    handles = [mlines.Line2D([], [], color=c[i], marker=m[i], linestyle=l[i], linewidth=3, markersize=5) for i in range(len(value_y_list))]
    plt.legend(handles, headers, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt_style = [x for x in plt.style.available if x not in ["dark_background", "fast", "Solarize_Light2", "seaborn-v0_8-colorblind", "seaborn-v0_8-muted", "seaborn-v0_8-poster"]]
    pic = sample(plt_style, 1)[0]
    # print(index, pic)
    plt.style.use(pic)
    #-----------------------------------------------------------#
    p_texts = []
    for i, y_list in enumerate(value_y_list):
        plt.plot(np.arange(len(value_x)), y_list, color=c[i], marker=m[i], linestyle=l[i], linewidth=3, markersize=5)
        p_text = []
        for x, y in zip(np.arange(len(value_x)), y_list):
            p_text.append(plt.text(x, y, str(y), fontsize=12))
        p_texts.append(p_text)

    plt.xlabel(value_x_cols, labelpad=7)
    plt.ylabel(value_y_cols, labelpad=7)

    try: 
        plt.xticks(np.arange(len(value_x)), labels=value_x) 
    except:
        return 7, None

    y_tick = np.arange(min_y, max_y+(max_y-min_y)/5, (max_y-min_y)/5)
    plt.yticks(y_tick, labels=[str(y) for y in y_tick]) 

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.xlim([-1, len(value_x)])
    plt.ylim([min_y-((max_y-min_y)/10), max_y+((max_y-min_y)/10)])
    
    fig_name = f'{file_names[0]}/line_{index}.png'
    # bbfig_name = f'{file_names[1]}/{file_names[1]}_{index}.png'
    
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 2, h * 1)

    try:
        plt.tight_layout()
    except:
        return 7, None 
    
    bbox = fig.get_window_extent(None).transformed(fig.dpi_scale_trans.inverted())
    plt_size = bbox.width*fig.dpi, bbox.height*fig.dpi
    # plt.savefig(fig_name, dpi=200, bbox_inches='tight', pad_inches=0)

    # print(bbox.width, bbox.width, fig.dpi)
    print(plt_size)
    plt_dict = {
        "p_text": p_texts,
        "x_value": value_x,
        "y_value": value_y_list,
        "x_label": value_x_cols,
        "y_label": value_y_cols,
        "y_tick": y_tick,
    }
    plt.savefig(fig_name)
    bbox_list = Line_bbox(ax, **plt_dict)
    # visual_bbox(bbox_list, plt_size, fig_name, bbfig_name)
    fig.clf()
    
    plt.close()
    return 0, (fig_name, _, bbox_list)

def Line_bbox(ax, **kwargs): 
    #-------------------------Example-------------------------#
    bbox_list=[]
    p_text = kwargs["p_text"]
    value_x = kwargs["x_value"]
    value_y_list = kwargs["y_value"]
    x_label = kwargs["x_label"]
    y_label = kwargs["y_label"]
    y_tick = kwargs["y_tick"]

    #xlabel
    box = np.array(ax.get_xaxis().get_label().get_window_extent(None).get_points())
    bbox_list.append(("x_label", [box[0][0], box[0][1], box[1][0], box[1][1]], str(x_label)))

    #ylabel
    box = np.array(ax.get_yaxis().get_label().get_window_extent(None).get_points())
    bbox_list.append(("y_label", [box[0][0], box[0][1], box[1][0], box[1][1]], str(y_label)))

    #xticklabel
    for i, l in enumerate(ax.get_xticklabels()):
        box = np.array(l.get_window_extent(None))
        (x_min, y_min), (x_max, y_max) = box
        text_boxes = l.get_text().split('\n')
        height = y_max - y_min
        num_lines = len(text_boxes)
        sub_height = height/num_lines
        for i, tb in enumerate(text_boxes[::-1]): ## from bottom to top
            new_bbox = [x_min, y_min + sub_height*i, x_max, y_min + sub_height*(i+1)]
            bbox_list.append((f'x_tick', new_bbox, tb))
        # bbox_list.append(("x_tick", [box[0][0], box[0][1], box[1][0], box[1][1]], str(value_x[i])))

    #yticklabel
    for i, l in enumerate(ax.get_yticklabels()):
        box = np.array(l.get_window_extent(None))
        bbox_list.append(("y_tick", [box[0][0], box[0][1], box[1][0], box[1][1]], str(f'{y_tick[i]:.0f}')))

    #line value
    for i in p_text:
        for j, l in enumerate(i):
            box = np.array(l.get_window_extent(None).get_points())
            bbox_list.append(("val", [box[0][0], box[0][1], box[1][0], box[1][1]], str(l.get_text())))

    #visual legend
    for i, patch in enumerate(ax.get_legend().get_lines()):
        box = np.array(patch.get_window_extent(None).get_points())
        bbox_list.append(("v_legend", [box[0][0], box[0][1], box[1][0], box[1][1]], None))

    #text legend
    for i, label in enumerate(ax.get_legend().get_texts()):
        box = np.array(label.get_window_extent(None).get_points())
        bbox_list.append(("t_legend", [box[0][0], box[0][1], box[1][0], box[1][1]], str(label.get_text())))

    x = np.arange(len(value_x))
    for idx, y in enumerate(value_y_list):
        for index, (i, j) in enumerate(zip(x, y)): #sub lines
            xmin, ymin = ax.transData.transform((i, j))
            xmax, ymax = ax.transData.transform((x[index+1], y[index+1]))
            if ymax < ymin:
                temp = ymax
                ymax = ymin
                ymin = temp
            bbox_list.append((f"line_{idx}", [xmin, ymin, xmax, ymax], None))
            if index == len(x)-2: break

    return bbox_list

