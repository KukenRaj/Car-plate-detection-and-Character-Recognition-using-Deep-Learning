# License-plate-detection-and-Character-Recognition-using-Deep-Learning

The project is about car plate detection and character recognition using Deep Learning. 
Yolo v3 architecture is used as it use COCO model dataset and easy to train on
Pytorch is used as framework for coding 
The detection system is tested mainly on cars in Malaysia.
Google Colab is used mainly to train and test the coding of this project


The coding is executable and the simulated result is demonstrated as below

![successcarplate](https://user-images.githubusercontent.com/70626062/91957656-e722f200-ed38-11ea-88d8-491568cae6d4.PNG)

# Requirement
- Torch > 1.0
- Opencv
- Python
- Numpy
- Tesseract 0.3.4 
- Input- video

# Procedure 
1) The .ipynb file contain the steps to implement the project and start implement the coding from RUNNING MAIN CODING
   section.(Use Google colab for easy startup)
2) The Yolo files folder contain all yolo configuration file and executable file
3) Use detect.py to start up the coding 
4) Use the command line below to initialize the detect.py file to redirect the input video path, redirect the path
   to weight file, redirect path to output video, confidence level setting and Nms suppression setting.
  
  !python3 detect.py --video /content/warpimages/guard_in2.mp4 --cfg /content/yolov3/License-plate-detection/cfg/yolov3.cfg --weights /content/yolov3/License-plate-detection/carplate.weights --conf-thres 0.1

5) The output video should be released after few minutes

# References
This project using yolov3 to license plate detection and character recognition using repo https://github.com/ultralytics/yolov3
and https://github.com/ThorPham/License-plate-detection....
