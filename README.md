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
Then, download the weights for YOLO. The weights were pretrained on COCO and downloaded from https://pjreddie.com/darknet/yolo/ (YOLOv3-416). I decided to reupload them to my google drive because the server of the mentioned website is very slow. You can download the weights from here https://drive.google.com/file/d/1HBvHn55G7hS8lHt9ZklU0L1dc82nVWN3/view?usp=sharing and copy it into the model directory or use the commands below.
```
cd people-detection/model
curl --header 'Host: doc-14-4s-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --cookie 'AUTH_4ip9opr6ac35ogbon7bbb5aa2kedhp76_nonce=cg9i2i50ej7g0' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-14-4s-docs.googleusercontent.com/docs/securesc/3lvt5kf2sonriqfd9faaa01269qieg10/b4ek8onmpspe5ro26aohlnsbaps8q2nm/1601108775000/06214964262441732686/06214964262441732686/1HBvHn55G7hS8lHt9ZklU0L1dc82nVWN3?e=download&authuser=0&nonce=cg9i2i50ej7g0&user=06214964262441732686&hash=f9qvd6619vdbvsqf5prcudga7ebq7umj' --output 'yolov3-416.weights'
```
You are now ready to run the code!

## Running instructions

If you run the following command, one of the example images will be loaded and it's output will be saved to the specified directory.
```
python main.py
```
If you would like to change the test image, please refer to the arguments specified in main.py, it should be very intuitive to understand.
One also has the option to change certain parameters such as confidence and NMS threshold. You can also just use the function defined in yolo.py and pass in the image path manually.
