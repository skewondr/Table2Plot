import numpy as np
import random
import copy
import operator
import itertools
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from PIL import Image

def preprocess_detections(_lines):
    lines = [line for line in _lines if len(line)]
    return lines

def find_center(bbox):
    """
    Helper method, used to find the center of a bboxe.
    Inputs:
    bbox (dictionary) : [x1, y1, x2, y2]
    Outputs
    (x, y) (tuple): X and Y co-ordinates of the bbox.
    """
    x1, y1, x2, y2 = bbox

    x = 0.5 * (float(x1) + float(x2))
    y = 0.5 * (float(y1) + float(y2))

    return (x,y)

def get_color(img, color_range=512):
    """
    Helper function, used to get the main color of the image.
    Inputs:
    img (PIL.Image.Image): A PIL Image croped from the original image.
    Outputs
    color (tuple): Value of the color as (R,G,B) tuple.
    """

    basewidth = 100
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    if hsize==0:
        hsize = 1
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    colors = img.getcolors(color_range) #put a higher value if there are many colors in your image

    max_occurence, most_present = 0, (0,0,0)

    try:
        for c in colors:
            if c[0] > max_occurence and c[1] not in [(255,255,255), (0,0,0), 0, (0)] and c[1]!=0 and colorDistance(c[1], [255,255,255],method="euclidian") > 50:
                (max_occurence, most_present) = c
    except TypeError:
        color_range=2*color_range
        if color_range < 10000:
            return get_color(img, color_range)

    return list(most_present)

def list_subtraction(l1, l2):
    """
    l1 = [1,2,3]
    l2 = [1,3]
    return [2]
    """
    return [item for item in l1 if item not in l2]

def find_Distance(p1,p2):
    x1, y1 = p1
    x2, y2 = p2

    d = ((x2-x1)**2 + (y2-y1)**2 )**0.5

    return d

def find_legend_orientation(legend_preview_data):
    """
    Helper method to find the orientation of the legends.
    Inputs:
    legend_preview_data : (list of dictionary) A dictionary which has the bbox of the preview.
    Outputs:
    orientation: (string): Can be horizontal or vertical depending on how the legend-preview bboxes are oriented.
    """
    if len(legend_preview_data) > 1:
        center_x = []
        center_y = []
        for preview_bbox in legend_preview_data:
            x,y = find_center(preview_bbox['bbox'])
            center_x.append(x)
            center_y.append(y)

        if abs(center_x[1] - center_x[0]) > abs(center_y[1] - center_y[0]):
            orientation = 'horizontal'
        else:
            orientation = 'vertical'
    else:
        orientation = 'unknown'
    return orientation

def legend_preview_association(legend_preview_data, legend_label_data, orientation):
    """
    Helper method, used to associate the legend preview with the corresponding legend label.
    Inputs:
    legend_preview_data (list of dictionary): A dictionary which has the bbox of the preview and the associated color of the preview.
    legend_label_data (list of dictionary): A dictionary which has the bbox of the legend label and the associated label.
    orientation (string) : orientation of the legends (vertical or horizontal)
    Output:
    legend_preview_data (list of dictionary): A dictionary which has the bbox of the preview and the associated color of the preview and value as the associated legend label.
    Pseudocode:

    for all preview bboxes :
       for all legend-label bboxes :
          if preview_bbox_center_X < legend_label_bbox_center_X :
                find distance between the (xmax, ymax) of preview_bbox and (xmin,ymax) of label_bbox
       associate that legend-label to the preview bbox whose distance is minimum

    """
    if orientation == "vertical":
        # sort preview bboxes according to xmin
        legend_preview_data = sorted(legend_preview_data, key=lambda k: k['bbox'][1])
        # sort legend-label bboxes according to xmin and corresponding legend-labels also
        legend_label_data = sorted(legend_label_data, key=lambda k: k['bbox'][1])
    else:
        # sort preview bboxes according to xmin
        legend_preview_data = sorted(legend_preview_data, key=lambda k: k['bbox'][0])
        # sort legend-label bboxes according to xmin and corresponding legend-labels also
        legend_label_data = sorted(legend_label_data, key=lambda k: k['bbox'][0])

    # find the center of preview bboxes
    preview_bboxes_center = []
    for bbox in legend_preview_data:
        center_x, center_y = find_center(bbox['bbox'])
        preview_bboxes_center.append((center_x, center_y))

    # find the center of legend-label bboxes
    legend_label_bboxes_center = []
    for bbox in legend_label_data:
        center_x, center_y = find_center(bbox['bbox'])
        legend_label_bboxes_center.append((center_x, center_y))

    for p_idx, preview_bbox in enumerate(legend_preview_data):

        preview_xmax = preview_bbox['bbox'][2]
        preview_ymax = preview_bbox['bbox'][3]

        min_distance = 1000000
        min_lbl_idx = -1

        for lbl_idx, label_bbox in enumerate(legend_label_data):
            # compare with only those legends which are ahead of the preview
            if preview_bboxes_center[p_idx][0] < legend_label_bboxes_center[lbl_idx][0]:

                label_xmin = label_bbox['bbox'][0]
                label_ymax = label_bbox['bbox'][3]

                distance = ((preview_xmax - label_xmin)**2 + (preview_ymax - label_ymax)**2)**(0.5)

                if distance < min_distance:
                    min_distance = distance
                    min_lbl_idx = lbl_idx

        preview_bbox['associated_label'] = legend_label_data[min_lbl_idx]['ocr_text']

    return legend_preview_data

def associate_legend_preview(image_data):
    legend_preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    if len(legend_preview_data) > 1:
        legend_label_data = [dd for dd in image_data if dd["pred_class"] == "legend_label"]
        legend_orientation = find_legend_orientation(legend_preview_data)
        lpa = legend_preview_association(legend_preview_data, legend_label_data, legend_orientation)
        image_data = list_subtraction(image_data, legend_preview_data)
        image_data = image_data + lpa

    return image_data, legend_orientation

def colorDistance(c1, c2, method=""):
    if method == "euclidian":
        return _colorDistance(c1, c2)
    elif method == "delta_e":
        '''
        Computes delta_e_cie2000 distance between two colors
        '''
        color1_rgb = sRGBColor(c1[0], c1[1], c1[2]);

        color2_rgb = sRGBColor(c2[0], c2[1], c2[2]);

        # Convert from RGB to Lab Color Space
        color1_lab = convert_color(color1_rgb, LabColor);

        # Convert from RGB to Lab Color Space
        color2_lab = convert_color(color2_rgb, LabColor);

        # Find the color difference
        delta_e = delta_e_cie2000(color1_lab, color2_lab);

        return delta_e

def _colorDistance(c1, c2):
    '''
    Calculates the Euclidian distance between two colors
    '''
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    d = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
    return d

def form_groups(visual_data, isHbar, sort_key=""):
    '''
    groups is a list of list of dictionaries.
    Ideally,
    len(groups) = number of groups in the plot = number of ticklabels
    len(groups[0]) = number of bars on each tick = number of preview_bboxes
    '''

    if isHbar and sort_key == "":
        sort_key = "y_value"
    elif not isHbar and sort_key == "":
        sort_key = "x_value"

    visual_data.sort(key=operator.itemgetter(sort_key))

    groups = []

    for key, items in itertools.groupby(visual_data, operator.itemgetter(sort_key)):
        groups.append(list(items))

    return groups

def random_assignments(group1, group2):
    '''
    Inputs:
    group1 (list of dictionary) : visual elements to whom legend-label is not assigned
    group2 (dictionary) : a mapping which has legend-label as key and corresponding preview color as value
    '''
    for g in group1:
        if "associated_label" in g.keys():
            continue

        try:
            k = random.choice(group2.keys())
            del group2[k]
        except:
            k = "legend-label"
        g["associated_label"] = k

    return group1

def match_colors(group, _mapping):

    unassigned_visual_elements = []

    mapping = copy.deepcopy(_mapping)

    for dd in group:
        visual_color = dd["color"]

        if visual_color == [255, 255, 255]:
            unassigned_visual_elements.append(dd)
            continue

        # tmp_lbls = [lbl for lbl,c in mapping.items() if colorDistance(c, visual_color) <= 20]
        #
        # if len(tmp_lbls) > 0:
        #     dd["associated_label"] = tmp_lbls[0]
        #     del mapping[tmp_lbls[0]]
        # else:
        #     unassigned_visual_elements.append(dd)

        # change the way you associate the color to the visual items
        tmp_lbls = [lbl for lbl,c in mapping.items() if colorDistance(c, visual_color, method="euclidian") <= 20]
        distance_with_preview = [colorDistance(c, visual_color, method="euclidian") for lbl,c in mapping.items() if colorDistance(c, visual_color, method="euclidian") <= 20]

        if len(tmp_lbls) > 0:
            min_index_ = np.argmin(distance_with_preview)
            dd["associated_label"] = tmp_lbls[min_index_]
            del mapping[tmp_lbls[min_index_]]
        else:
            unassigned_visual_elements.append(dd)
    if len(unassigned_visual_elements):
        group = random_assignments(group, mapping)

    return group

def find_plot_type(image_data):
    pred_classes = list(set([dd["pred_class"] for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"]]))
    if len(pred_classes):
        return pred_classes[0]
    else:
        return "empty"

def associate_bar_legend(image_data, isHbar):

    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]

    visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line"]]

    image_data = list_subtraction(image_data, visual_data)

    # Grouping the visual elements based on tick labels
    visual_groups = form_groups(visual_data, isHbar)

    _mapping = {}
    updated_visual_data = []

    # Create a map from legend-label to corresponding preview-clor
    for i in range(len(preview_data)):
        c = preview_data[i]["color"]
        lbl = preview_data[i]["associated_label"]
        _mapping[lbl] = c

    # NITESH: for each group of visual elements, find the associated color
    for group in visual_groups:
        _group = match_colors(group, _mapping)
        for item in _group:
            updated_visual_data.append(item)
    #
    # for i in range(len(visual_data)):
    #     tmp_lbls = [lbl for lbl,c in mapping.items() if colorDistance(c, visual_data[i]["color"]) <= 20]
    #
    #     if len(tmp_lbls) > 0:
    #         visual_data[i]["associated_label"] = tmp_lbls[0]
    #     else:
    #         visual_data[i]["associated_label"] = "Unknown"

    image_data = image_data + updated_visual_data

    return image_data

def associate_line_legend(image_data):
    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    visual_data = [dd for dd in image_data if dd["pred_class"] == "line"]

    preview_colors_mapping = [(c["color"], c["associated_label"]) for c in preview_data]
    image_data = list_subtraction(image_data, visual_data)

    for vd in visual_data:
        for cidx in range(len(preview_colors_mapping)):
            if vd["color"] == preview_colors_mapping[cidx][0]:
                vd["associated_label"] = preview_colors_mapping[cidx][1]
                break

    image_data = image_data + visual_data
    return image_data

def associate_visual_legend(image_data, isHbar, image):
    plot_type = find_plot_type(image_data)
    if plot_type in ["bar", "dot_line"]:
        return associate_bar_legend(image_data, isHbar)
    else:
        return associate_line_legend(image_data)

# Find slope of the line in an image
def find_slope(image, bb):

    if bb[1] == bb[3]:
        slope = "horizontal"

    else:

        bb_image = image.crop(bb).convert('1') #  0 (black) and 1 (white)

        bb_image_asarray = np.asarray(bb_image, dtype=np.float32)

        img_h, img_w = bb_image_asarray.shape

        row, col = img_h/2, img_w/2

        patchA = bb_image_asarray[0:row, 0:col]
        patchB = bb_image_asarray[0:row, col:]
        patchC = bb_image_asarray[row:, 0:col]
        patchD = bb_image_asarray[row:, col:]

        a,b,c,d = np.mean(patchA), np.mean(patchB), np.mean(patchC), np.mean(patchD)

        if (a < b) and (c > d):
            slope="negative" # take points x1, y1, x2, y2

        elif (a > b) and (c < d):
            slope="positive" # take points x1, y2, x2, y1
        else:
            slope = random.choice(["positive", "negative"])

    return slope

def split_image_data(image_data):
    '''
    Splits the image data according to the colors present in the legend-preview. We are doing it only for line plots.
    '''
    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]

    visual_data = [dd for dd in image_data if dd["pred_class"] == "line"]

    preview_colors = [c["color"] for c in preview_data]

    image_data = list_subtraction(image_data, visual_data)

    for vd in visual_data:
        min_d = 1e10
        color_index = -1
        for cidx, pc in enumerate(preview_colors):
            d = colorDistance(vd["color"], pc, method="delta_e")
            if d <= min_d:
                color_index = cidx
                min_d = d
        vd["color"] = preview_colors[color_index]

    _splits = form_groups(visual_data, False, sort_key="color")

    splits = []

    for each_split in _splits:
        splits.append(each_split + image_data)

    return splits
