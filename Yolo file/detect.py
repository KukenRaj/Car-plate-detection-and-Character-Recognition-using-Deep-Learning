import argparse
import shutil
import time
from pathlib import Path
from sys import platform
import numpy as np
import imutils
import cv2
import os
import pytesseract
import random
try:
  from PIL import Image
except ImportError:
  import Image

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        weights,
        video,
        outcome,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    args = vars(parser.parse_args())
    vs = cv2.VideoCapture(args["video"])
    writer = None
    (H, W) = (None, None)

    try:
      prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
      else cv2.CAP_PROP_FRAME_COUNT
      total = int(vs.get(prop))
      TimCount = 0
      print("[INFO] {} total frame in original video".format(total))

    except:
      print("[INFO] could not determine # of frames in video")
      print("[INFO] no aprrox. completion time can be provided")
      total = -1


    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadWebcam(args["video"], img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            try :
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

                # Rescale boxes from 416 to true image size
                detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

                # Print results to screen
                unique_classes = detections[:, -1].cpu().unique()
                for c in unique_classes:
                    n = (detections[:, -1].cpu() == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write('%g %g %g %g %g %g\n' %
                                    (x1, y1, x2, y2, cls, cls_conf * conf))

                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])
            except:
                print("sth wrong")

        dt = time.time() - t
        fps = 1 / dt
        TimCount = TimCount + dt
        CDown = total / fps
        Remain = CDown - TimCount
        print('%.2f remaining time' % Remain)

        if H is None or W is None:
          (H, W) = im0.shape[:2]

        if save_images:  # Save generated image with detections
            if writer is None:
              fourcc = cv2.VideoWriter_fourcc(*'mp4v')
              writer = cv2.VideoWriter(args["outcome"], fourcc, 30, 
              (im0.shape[1], im0.shape[0]), True)

        writer.write(im0)
        if webcam:  # Show live webcam
            #cv2.imshow(weights + ' - %.2f FPS' % (1 / dt), im0)
            cv2.imshow("im",im0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

    writer.release()
    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--video', type=str, default='data/samples', help='path to images')
    parser.add_argument('--outcome', type=str, default='/content/yolov3/License-plate-detection/output/outcomevideo.mp4', help='path to output video')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.video,
            opt.outcome,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )

print("[INFO] cleaning up...")