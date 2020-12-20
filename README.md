# people-detection

Requirements:
- argparse
- cv2
- numpy
- matplotlib
- json
- os
- pathlib

## Setup
First, clone the repository.
```
git clone https://github.com/ByMic/people-detection.git
```
Then, download the weights for YOLO. The weights were pretrained on COCO and downloaded from https://pjreddie.com/darknet/yolo/ (YOLOv3-416).
You are now ready to run the code!

## Running instructions

If you run the following command, one of the example images will be loaded and it's output will be saved to the specified directory.
```
python main.py
```
If you would like to change the test image, please refer to the arguments specified in main.py, it should be very intuitive to understand.
One also has the option to change certain parameters such as confidence and NMS threshold. You can also just use the function defined in yolo.py and pass in the image path manually.
