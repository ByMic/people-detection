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
Then, download the weights for YOLO. The weights were pretrained on COCO and downloaded from https://pjreddie.com/darknet/yolo/. I decided to reupload them to my google drive because the server of the mentioned website is very slow.
```
cd people-detection/model
curl --header 'Host: doc-14-4s-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' --header 'Accept-Language: de,en-US;q=0.7,en;q=0.3' --cookie 'AUTH_4ip9opr6ac35ogbon7bbb5aa2kedhp76_nonce=m8obbruo6i1d4' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-14-4s-docs.googleusercontent.com/docs/securesc/3lvt5kf2sonriqfd9faaa01269qieg10/lrnq95c0iaf0jotmn95je1qqn3goikn4/1601044725000/06214964262441732686/06214964262441732686/1HBvHn55G7hS8lHt9ZklU0L1dc82nVWN3?e=download&authuser=0&nonce=m8obbruo6i1d4&user=06214964262441732686&hash=8ucjg1td4lbmjcfegh2phbfce5ig9ivv' --output 'yolov3-416.weights'
```
You are now ready to run the code!

## Running instructions

If you run the following command, one of the example images will be loaded and it's output will be saved to the specified directory.
```
python main.py
```
If you would like to change the test image, please refer to the arguments specified in main.py, it should be very intuitive to understand.
One also has the option to change certain parameters such as confidence and NMS threshold.
