#Upscale boxes
import matplotlib.pyplot as plt
from bbox_conversion import *

def upscale_boxes(target_size, box, image, visualise_scaled_box=False):
    x_scale, y_scale = getScale(650, 650, target_size)
    scaled_box = ResizeBox(box, x_scale, y_scale)
    if visualise_scaled_box:
        setup_plot(image)
        add_bboxes_to_plot(scaled_box, 'cyan')
        plt.show()
    return scaled_box
