import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '-c', '--config', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--img-size', 
        default=None,
        dest='img_size',
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    OUT_DIR = set_infer_dir()

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['config']['NC']
            CLASSES = checkpoint['config']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
#########################################################
    name_image = []
    classes = []
    xs = []
    ys = []
    ws = []
    hs = []
    conf = []
    img_w_dim = []
    img_h_dim = []
########################################################
    for i in range(len(test_images)):
#######################################################
        line_ls = test_images[i].split("/")
        #name = line_ls[len(line_ls) - 1][:-4]
        img = cv2.imread(test_images[i])
        img_h, img_w = img.shape[:2]
#######################################################
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        #print("Analizzando2: ",image_name)
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['img_size'] != None:
            RESIZE_TO = args['img_size']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(orig_image, RESIZE_TO)
        image = image_resized.copy()
        res_h, res_w, _ = image.shape
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
#####################################################################################################################################
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(outputs[0]['boxes'].detach().tolist())]
            pred_score = list(outputs[0]['scores'].detach().tolist())
            for i in range(len(pred_score)):
                # [x,y,w,h]
                name_image.append(int(image_name))
                classes.append("acino")
                xs.append(int(pred_boxes[i][0][0]/res_w*img_w))
                ys.append(int(pred_boxes[i][0][1]/res_h*img_h))
                ws.append(int(pred_boxes[i][1][0]/res_w*img_w - pred_boxes[i][0][0]/res_w*img_w))
                hs.append(int(pred_boxes[i][1][1]/res_h*img_h - pred_boxes[i][0][1]/res_h*img_h))
                conf.append(pred_score[i])
                img_w_dim.append(img_w)
                img_h_dim.append(img_h)
#####################################################################################################################################
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            orig_image = inference_annotations(
                outputs, 
                detection_threshold, 
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
            if args['show_image']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)
################################################################################################################################
    print(name_image)
    raw_data = {
        'image-name':name_image,
        'class':classes,
        'x':xs,
        'y':ys,
        'w':ws,
        'h':hs,
        'confidence':conf,
        'image-w-dim':img_w_dim,
        'image-h-dim':img_h_dim
    }

    data = pd.DataFrame(raw_data,columns=['image-name','class','x','y','w','h','confidence','image-w-dim','image-h-dim'])
    data.to_csv("faster_val_detected_bb.csv",index=False)
##################################################################################################################################
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)