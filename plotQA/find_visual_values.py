import numpy as np
from utils import *
import copy

'''
First co-ordinate  = x-value for vbar, dot-line and line and y-value for hbar
Second co-ordinate = y-value for vbar, dot-line and line and x-value for hbar
'''

def handle_negative_visuals(negative_visuals, isHbar):
    for dd in negative_visuals:
        assert dd["isNegative"] == True
        if isHbar:
            dd["x_value"] = 0.0
        else:
            dd["y_value"] = 0.0

    return negative_visuals

def find_first_coord(visual_data, isHbar, ticklabel):

    for bidx in range(len(visual_data)):
        x1, y1, x2, y2 = visual_data[bidx]["bbox"]
        minDistance = 1e10
        b_lbl_idx = -1
        for tidx in range(len(ticklabel)):
            a1, b1, a2, b2 = ticklabel[tidx]["bbox"]
            ax, by = find_center([a1, b1, a2, b2])
            if isHbar:
                visual_point = [x1, y2]
                lbl_point = [a2, b2]
            else:
                visual_point = [x1, y2] # Take x1,y2 instead of x2,y2
                lbl_point = [ax, b1] # Take ax, b1 instead of a2,b1
            d = find_Distance(lbl_point, visual_point)
            if d < minDistance:
                b_lbl_idx = tidx
                minDistance = d

        if isHbar:
            visual_data[bidx]["y_value"] = ticklabel[b_lbl_idx]["ocr_text"]
        else:
            visual_data[bidx]["x_value"] = ticklabel[b_lbl_idx]["ocr_text"]

    # Handle the last bbox for line plot
    if visual_data[0]["pred_class"] == "line":

        _visual_data = copy.deepcopy(visual_data)

        visual_data.append(visual_data[-1])

        visual_data[-1]["x_value"] = ticklabel[-1]["ocr_text"]

        _visual_data.append(visual_data[-1])

        return _visual_data

    return visual_data

def find_visual_values(image, image_data, isHbar, isSinglePlot):

    # Associate the bar with the x-label (if vertical bar) or y-label (if Hbar)
    if isHbar:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
    else:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
        ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][0])

    visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"]]
    visual_data = sorted(visual_data, key=lambda x: x['bbox'][0])

    if len(visual_data) == 0:
        return -1

    image_data = list_subtraction(image_data, visual_data)
    visual_data = find_first_coord(visual_data, isHbar, ticklabel)
    image_data = image_data + visual_data
    ###

    # Associate the bar with the y-label (if vertical bar) or x-label (if Hbar)
    if isHbar:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
        ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][0])
        yticks = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
        start = yticks[0]['bbox'][2] + 9 # added 9 so that the start starts from the center of the major tick
    else:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
        ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][1])
        xticks = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
        start = xticks[0]['bbox'][1] - 9 # added 9 so that the start starts from the center of the major tick

    for t_i in range(len(ticklabel)-1):
        tick1 = ticklabel[t_i]
        tick2 = ticklabel[t_i + 1]

        tick1['ocr_text'] = tick1['ocr_text'].replace(" ", "").replace("C","0").replace("+", "e+").replace("ee+", "e+").replace("O","0").replace("o","0").replace("B","8")

        tick2['ocr_text'] = tick2['ocr_text'].replace(" ", "").replace("C","0").replace("+", "e+").replace("ee+", "e+").replace("O","0").replace("o","0").replace("B","8")

        if tick1['ocr_text'][-1] == "-":
            tick1['ocr_text'] = tick1['ocr_text'][:-1]

        if tick2['ocr_text'][-1] == "-":
            tick2['ocr_text'] = tick2['ocr_text'][:-1]

        if len(tick1['ocr_text'])> 0 and len(tick2['ocr_text']) > 0 and tick2['ocr_text'] != tick1['ocr_text']:
            break
        #else:
        #    tick1['ocr_text'] = "Unknown"
        #    tick2['ocr_text'] = "Unknown"

    if len(tick1['ocr_text']) == 0 or len(tick2['ocr_text']) == 0:
        return "Skip, yticklabels are not detected by OCR"

    # Calculating pixel difference from tick-label's center instead of corners
    c_x1, c_y1 = find_center(tick1['bbox'])
    c_x2, c_y2 = find_center(tick2['bbox'])

    if isHbar:
        #pixel_difference = abs(tick1['bbox'][2] - tick2['bbox'][2])
        pixel_difference = abs(c_x2 - c_x1)
    else:
        #pixel_difference = abs(tick1['bbox'][3] - tick2['bbox'][3])
        pixel_difference = abs(c_y2 - c_y1)

    if "84-" in tick1['ocr_text']:
        tick1['ocr_text'] = tick1['ocr_text'].replace("84-", "e+")
    if "84-" in tick2['ocr_text']:
        tick2['ocr_text'] = tick2['ocr_text'].replace("84-", "e+")

    if "91-" in tick1['ocr_text']:
        tick1['ocr_text'] = tick1['ocr_text'].replace("91-", "e+")
    if "91-" in tick2['ocr_text']:
        tick2['ocr_text'] = tick2['ocr_text'].replace("91-", "e+")

    value_difference = abs(float(tick1['ocr_text']) - float(tick2['ocr_text']))
    scale = value_difference / pixel_difference

    visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" not in dd.keys()]
    negative_visuals = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" in dd.keys()]

    image_data = list_subtraction(image_data, visual_data)
    image_data = list_subtraction(image_data, negative_visuals)

    negative_visuals = handle_negative_visuals(negative_visuals, isHbar)

    if not isHbar:
        visual_data = sorted(visual_data, key=lambda x: x['bbox'][0])

    #NITESH : Finding second-coordinate here
    for bidx in range(len(visual_data)):
        if visual_data[bidx]["pred_class"] == "bar":
            if isHbar:
                compare_with = abs(visual_data[bidx]['bbox'][2] - start) # length of the bar
            else:
                compare_with = abs(visual_data[bidx]['bbox'][1] - start) # height of the bar
        # else:
        #     compare_with = abs(visual_data[bidx]['bbox'][3] - start)

        else:
            if visual_data[bidx]["pred_class"] == "dot_line":
                # center of the dot-line
                cx, cy = find_center(visual_data[bidx]['bbox'])
                compare_with = abs(cy - start)

            elif visual_data[bidx]["pred_class"] == "line":

                slope = find_slope(image, visual_data[bidx]['bbox'])

                x1, y1, x2, y2 = visual_data[bidx]['bbox']

                if slope == "positive":
                    compare_with = abs(y2 - start)
                # if slope is horizontal, both y1 and y2 are equal
                else:
                    compare_with = abs(y1 - start)

        value = compare_with * scale

        if isHbar:
            visual_data[bidx]["x_value"] = value
        else:
            visual_data[bidx]["y_value"] = value

    # Repeat the above steps for line plot to find the y-value of the last bbox
    if visual_data[-1]["pred_class"] == "line":
        slope = find_slope(image, visual_data[-1]['bbox'])
        x1, y1, x2, y2 = visual_data[-1]['bbox']
        if slope == "positive":
            compare_with = abs(y1 - start)
        # if slope is horizontal, both y1 and y2 are equal
        else:
            compare_with = abs(y2 - start)
        value = compare_with * scale
        visual_data[-1]["y_value"] = value

    image_data = image_data + visual_data
    image_data = image_data + negative_visuals

    return image_data
