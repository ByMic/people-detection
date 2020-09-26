from yolo import detectPersons
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path',default='test_images[1]/test_images/1.jpg', help='Path to test image.')
    parser.add_argument('--save_dir',default='save_dir', help='Where to save the results')
    parser.add_argument('--model_config',default='model/yolov3-416.cfg', help='Path to YOLO model config.')
    parser.add_argument('--model_weights',default='model/yolov3-416.weights', help='Path to YOLO model weights.')
    parser.add_argument('--classes',default='model/coco.names', help='Path to class names.')
    parser.add_argument('--confidence',default=0.3, type=float, help='Confidence threshold.')
    parser.add_argument('--NMS',default=0.2, type=float, help='Non-maximum supression threshold.')
    parser.add_argument('--whT',default=416,type=int, help='Input size for YOLO model.')
    args = parser.parse_args()

    summary, crop_imgs = detectPersons(args.img_path, args.save_dir, args.model_weights, args.model_config, args.classes, args.confidence,args.NMS,args.whT)

    
    #you can also just pass in the img_path argument and rely on the default values defined in yolo.py
    #summary, crop_imgs = detectPersons(args.img_path)

if __name__ == '__main__':
    main()