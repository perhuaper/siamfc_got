from __future__ import absolute_import

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
import cv2

fig_dict = {}
patch_dict = {}


def show_frame(image, boxes=None, fig_n=1, pause=0.001,
               linewidth=3, cmap=None, colors=None, legends=None):
    r"""Visualize an image w/o drawing rectangle(s).
    
    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        fig_n (integer, optional): Figure ID. Default is 1.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        cmap (string): Color map. Default is None.
        color (tuple): Color of drawed rectanlge. Default is None.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if not fig_n in fig_dict or \
        fig_dict[fig_n].get_size() != image.size[::-1]:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]
        
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if not fig_n in patch_dict:
            patch_dict[fig_n] = []
            for i, box in enumerate(boxes):
                patch_dict[fig_n].append(patches.Rectangle(
                    (box[0], box[1]), box[2], box[3], linewidth=linewidth,
                    edgecolor=colors[i % len(colors)], facecolor='none',
                    alpha=0.7 if len(boxes) > 1 else 1.0))
            for patch in patch_dict[fig_n]:
                fig_dict[fig_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[fig_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])
        
        if legends is not None:
            fig_dict[fig_n].axes.legend(
                patch_dict[fig_n], legends, loc=1,
                prop={'size': 8}, fancybox=True, framealpha=0.5)

    plt.pause(pause)
    plt.draw()

def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):

    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img
