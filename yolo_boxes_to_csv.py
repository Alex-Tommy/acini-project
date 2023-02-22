import torch
import cv2
import os
import pandas as pd


name_img = []
classes = []
xs = []
ys = []
ws = []
hs = []
confidences = []
img_w_dim = []
img_h_dim = []


model = torch.hub.load(".", 'custom', path='runs/train/new-yolo/weights/best.pt', source='local') 
folder = "../dataset/images/test"
for strimg in os.listdir(folder):
    name = strimg[:-4]
    img = cv2.imread(folder+'/'+strimg)
    img_h, img_w = img.shape[:2]
    results = model(img)
    df = results.pandas().xywh[0]
    for index, row in df.iterrows():
        # [xcenter, ycenter, w, h] Absolute
        name_img.append(name)
        classes.append("acino")
        w = int(row["width"])
        h = int(row["height"])
        ws.append(w)
        hs.append(h)
        xs.append(int(row["xcenter"] - w / 2))
        ys.append(int(row["ycenter"] - h / 2))
        confidences.append(row["confidence"])
        img_w_dim.append(img_w)
        img_h_dim.append(img_h)

raw_data = {
    'image-name':name_img,
    'class':classes,
    'x':xs,
    'y':ys,
    'w':ws,
    'h':hs,
    'confidence':confidences,
    'image-w-dim':img_w_dim,
    'image-h-dim':img_h_dim
}

data = pd.DataFrame(raw_data,columns=['image-name','class','x','y','w','h','confidence','image-w-dim','image-h-dim'])
data.to_csv("yolo_test_detected_bb.csv",index=False)
