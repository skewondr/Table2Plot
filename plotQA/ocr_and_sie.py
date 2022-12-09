'''
Sample Usage:
python ocr_and_sie.py [PATH_TO_PNG_DIR] [PATH_TO_DETECTIONS] [OUTPUT_DIR]
'''


import os
import pytesseract
import pyocr
import cv2
import random

import click
import time
import logging
import sys

import pandas as pd
import numpy as np
from scipy import ndimage
from PIL import Image
from tqdm import tqdm
from utils import *
from find_visual_values import *
from upscale_boxes import *


tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))


def find_box_orientation(bb):
    x1, y1, x2, y2 = bb
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w > h:
        return "horizontal"
    else:
        return "vertical"

def preprocess_image(cropped_image, size, preprocess_mode):
    if cropped_image.mode=='RGBA':
        cropped_image = cropped_image.convert('RGB')

    # load the image and convert it to grayscale
    image = np.asarray(cropped_image)

    image = cv2.resize(image, None, fx=size, fy=size, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the image
    if preprocess_mode == "thresh":
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove noise
    elif preprocess_mode == "blur":
        gray = cv2.medianBlur(gray, 3)

    return gray

def doOCR(im, role, isHbar):

    if role == "ylabel":
        angle = 270
    elif role == "xticklabel":
        if isHbar:
            angle = 0
        else:
            angle = -270
    else:
        angle = 0

    im = Image.fromarray(ndimage.rotate(im, angle, mode='constant', cval=(np.median(im)+np.max(im))/2))

    if isHbar:
        if role=="xticklabel":
            # numbers
            text = str(tool.image_to_string(im, lang="eng+osd",builder=pyocr.tesseract.DigitBuilder(tesseract_layout=6)))
        else:
            text = tool.image_to_string(im,lang="eng", builder=pyocr.builders.TextBuilder())

    else:
        if role=="yticklabel":
            # numbers
            text = str(tool.image_to_string(im, lang="eng+osd",builder=pyocr.tesseract.DigitBuilder(tesseract_layout=6)))
        else:
            text = tool.image_to_string(im,lang="eng", builder=pyocr.builders.TextBuilder())

    if isHbar:
        if role == "xticklabel":
            text = text.replace(" ","")
            text = text.replace("\n","")

        if role == 'yticklabel':
            #text = text.replace("-", "")
            text = text.replace("\n","")
            #text = text.replace(" ","")
    else:
        if role == "yticklabel" :
            text = text.replace(" ","")
            text = text.replace("\n","")

        if role == 'xticklabel':
            #text = text.replace("-", "")
            text = text.replace("\n","")
            #text = text.replace(" ","")

    if role in ["title","xlabel",'ylabel', 'legend_label']:
        text = text.replace("\n"," ")

    return text

def find_isHbar(lines, min_score=0.8):

    bar_boxes = []
    class_names = []
    isHbar = False

    for line in lines:
        role, score, x1, y1, x2, y2 = line.split()
        #role, x1, y1, x2, y2 = line.split()
        #score = 1.0
        #_tmp = line.split()
        #role, x1, y1, x2, y2, value = _tmp[0], _tmp[1],_tmp[2],_tmp[3],_tmp[4]," ".join(_tmp[5:])

        class_names.append(role)

        if role == "bar" and float(score) >= min_score:
            bar_boxes.append([float(x1), float(y1), float(x2), float(y2)])

    if "preview" in class_names:
        isSinglePlot = False
    else:
        isSinglePlot = True

    if len(bar_boxes) == 0:
        isHbar = False
        return isHbar, isSinglePlot

    x1_sorted = sorted(bar_boxes, key=lambda x:x[0])
    y2_sorted = sorted(bar_boxes, key=lambda x:x[3])

    x1_dist = []
    y2_dist = []

    for i in range(1, len(x1_sorted)):
        d = x1_sorted[i][0] - x1_sorted[i-1][0]
        x1_dist.append(d)

    for i in range(1, len(y2_sorted)):
        d = y2_sorted[i][3] - y2_sorted[i-1][3]
        y2_dist.append(d)

    if np.mean(x1_dist) >= np.mean(y2_dist):
        isHbar = False
    elif np.mean(x1_dist) < np.mean(y2_dist):
        isHbar = True

    return isHbar, isSinglePlot

def run(png_dir, detections_dir, csv_dir, MIN_CLASS_CONFIDENCE=0.8):
    """
    Arguments:
        png_dir: Path to the directory where plot images are stored.
        detections_dir: Path to the directory where the model predictions are stored.
                        The predictions should be of the format:
                        CLASS_LABEL CLASS_CONFIDENCE XMIN YMIN XMAX YMAX
        csv_dir: Path to the output directory where the tables (in .csv format) will be saved.
        MIN_CLASS_CONFIDENCE: Minimum class confidence threshold to be considered for processing.
    """
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    textual_elements = ["title", "xlabel", "ylabel", "xticklabel", "yticklabel", "legend_label"]
    visual_elements = ["bar", "line", "dot_line", "preview"]

    all_images = os.listdir(png_dir)
    NUM_IMAGES = len(all_images)

    random.seed( 1234 )
    random.shuffle(all_images)

    image_names = all_images[:NUM_IMAGES]

    image_names = [int(img_name.replace(".png", "")) for img_name in image_names]

    error_images = []
    error_trace = []
    empty_images = []

    for _ in tqdm(range(len(image_names))):
        image_index = image_names[_]

        try:
            image_data = []

            if not os.path.exists(os.path.join(detections_dir, str(image_index) + ".txt")):
                continue

            with open(os.path.join(detections_dir, str(image_index) + ".txt"), 'r') as f:
                lines = f.read().split("\n")[:-1]

            lines = preprocess_detections(lines)

            isHbar, isSinglePlot = find_isHbar(lines)

            image = Image.open(os.path.join(png_dir, str(image_index) + ".png"))
            if image.mode=='RGBA':
                image = image.convert('RGB')

            img_width, img_height = image.size
            target_size = [img_width, img_height]

            for detection in lines:

                class_name, score, x1, y1, x2, y2 = detection.split()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                # Upscale the detections
                # box = [float(x1), float(y1), float(x2), float(y2)]
                # x1, y1, x2, y2 = upscale_boxes(target_size, box, image, visualise_scaled_box=False)

                if class_name in ["bar", "dot_line"]:
                    doProcessing = False
                    # break
                else:
                    doProcessing = True

                if float(score) < MIN_CLASS_CONFIDENCE:
                    continue

                if class_name in textual_elements:
                    # crop that part of the image
                    if class_name == 'xticklabel':
                        c_bb = [float(x1)-5, float(y1)-2, float(x2)+5, float(y2)+2]
                        preprocess_mode = "thresh"
                        size=10
                    elif class_name == 'yticklabel':
                        c_bb = [float(x1)-2, float(y1)-5, float(x2)+2, float(y2)+5]
                        preprocess_mode = "thresh"
                        size=3
                    else:
                        c_bb = [float(x1), float(y1), float(x2), float(y2)]
                        preprocess_mode = "thresh"
                        size=4.5
                    cropped_image = image.crop(c_bb)

                    # preprocess the cropped-image
                    gray_image = preprocess_image(cropped_image, size=size, preprocess_mode=preprocess_mode)

                    # do OCR
                    text = doOCR(gray_image, class_name, isHbar)
                    #text = value

                    if isHbar and class_name=="xticklabel":
                        if len(text) > 15:
                            continue
                    elif not isHbar and class_name=="yticklabel":
                        if len(text) > 15:
                            continue

                    image_data.append({
                        "bbox" : [float(x1), float(y1), float(x2), float(y2)],
                        "pred_class" : class_name,
                        "confidence" : score,
                        "ocr_text" : text
                    })

                elif class_name in visual_elements:
                    bb = [float(x1), float(y1), float(x2), float(y2)]

                    if (float(x1) > float(x2)) or (float(y1) > float(y2)):
                        negativeBar = True
                    else:
                        negativeBar = False

                    if (float(x1) == float(x2)) or (float(y1) == float(y2)):
                        emptyBar = True
                    else:
                        emptyBar = False

                    # Handling the bars with no width or height
                    if emptyBar:
                        image_data.append({
                            "bbox" : bb,
                            "pred_class" : class_name,
                            "confidence" : score,
                            "color" : [255, 255, 255] # The color of visual elements with no width, height is assumed to be white
                        })
                    else:
                        if negativeBar:
                            image_data.append({
                                "bbox" : bb,
                                "pred_class" : class_name,
                                "confidence" : score,
                                "color" : [255, 255, 255],
                                "isNegative" : True
                            })
                        else:
                            c = get_color(image.crop(bb))
                            image_data.append({
                                "bbox" : bb,
                                "pred_class" : class_name,
                                "confidence" : score,
                                "color" : c
                            })
            ##ocr 예측값이 존재한다.
            plot_type = find_plot_type(image_data)

            if plot_type == "empty":
                empty_images.append(image_index)
                continue

            if plot_type == "line" and not isSinglePlot:
                _image_data = copy.deepcopy(image_data)
                image_data = []
                splits = split_image_data(_image_data)
                for each_split in splits:
                    tmp_items = find_visual_values(image, each_split, isHbar, isSinglePlot)
                    for item in tmp_items:
                        if item not in image_data:
                            image_data.append(item)
            else:
                image_data = find_visual_values(image, image_data, isHbar, isSinglePlot)

            # legend-preview association
            if not isSinglePlot:
                image_data, legend_orientation = associate_legend_preview(image_data)
                #image_data = associate_bar_legend(image_data, isHbar)
                # Associate line using different rules and bars and dot-lines using color match
                image_data = associate_visual_legend(image_data, isHbar, image)

            if image_data == "Skip, yticklabels are not detected by OCR":
                #print("Skip, yticklabels are not detected by OCR")
                error_images.append(image_index)
                continue

            # convert image_data to csv
            if isSinglePlot:
                if isHbar:
                    legend_names = [dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="xlabel"]
                else:
                    legend_names = [dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="ylabel"]
            else:
                legend_names = list(set([dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="legend_label"]))
            #title로 분류된 것 중에서는 가장 긴 텍스트를 뽑음. 
            tmp_title = [dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="title"]
            title = ''
            min_title_len = 0
            for t in tmp_title:
                if len(t) >= min_title_len:
                    title = t
                    min_title_len = len(t)

            xlabel = list(set([dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="xlabel"]))[0]
            ylabel = list(set([dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="ylabel"]))[0]
            if isHbar:
                row_indexes = [dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="yticklabel"]
            else:
                row_indexes = [dd["ocr_text"].encode('utf-8') for dd in image_data if dd["pred_class"]=="xticklabel"]

            if isSinglePlot:
                if isHbar:
                    visual_data = [(dd['x_value'], dd['y_value'].encode('utf-8'), xlabel) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]
                else:
                    visual_data = [(dd['x_value'].encode('utf-8'), dd['y_value'], ylabel) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]
            else:
                visual_data = [(dd['x_value'], dd['y_value'], dd['associated_label'].encode('utf-8')) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]

            if isSinglePlot:
                if isHbar:
                    columns = [ylabel] + legend_names + ["xlabel", "ylabel", "title"]
                else:
                    columns = [xlabel] + legend_names + ["xlabel", "ylabel", "title"]
            else:
                if isHbar:
                    columns = [ylabel] + legend_names + ["Unknown", "xlabel", "ylabel", "title", "legend orientation"]
                else:
                    columns = [xlabel] + legend_names + ["Unknown", "xlabel", "ylabel", "title", "legend orientation"]

            df = pd.DataFrame(columns=columns)

            if isHbar:
                df[ylabel] = row_indexes
            else:
                df[xlabel] = row_indexes

            df["title"] = [title]*len(df)
            df["xlabel"] = [xlabel]*len(df)
            df["ylabel"] = [ylabel]*len(df)
            if "legend orientation" in df.columns:
                df["legend orientation"] = [legend_orientation]*len(df)

            for vd in visual_data:
                try:
                    if isHbar:
                        if type(vd[1]) is str:
                            i_ = df[df[ylabel] == vd[1]].index.item()
                        else:
                            i_ = df[df[ylabel] == vd[1].encode('utf-8')].index.item()
                        df.iloc[i_][vd[2]] = vd[0]
                    else:
                        if type(vd[0]) is str:
                            i_ = df[df[xlabel] == vd[0]].index.item()
                        else:
                            i_ = df[df[xlabel] == vd[0].encode('utf-8')].index.item()
                        df.iloc[i_][vd[2]] = vd[1]
                except:
                    continue

            df.to_csv(os.path.join(csv_dir, str(image_index) + ".csv"), index=False)
            # print(str(image_index) + " dumped!")
            # try block ends here

        except Exception as e:
            error_trace.append(e)
            error_images.append(image_index)

    print("[Error Images]")
    for i in range(len(error_images)):
        print(error_images[i], error_trace[i])

    print("[Empty Images]")
    for i in range(len(empty_images)):
        print(empty_images[i])


@click.command()
@click.argument("png_dir")
@click.argument("detections_dir")
@click.argument("csv_dir")
#@click.argument("visualise_model_output")
#@click.argument("visualise_merged_output")

def main(**kwargs):
    """
    """
    logging.basicConfig(level=logging.INFO)
    run(**kwargs)


if __name__ == "__main__":
    main()
