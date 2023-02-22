import cv2 
import os
import pandas as pd

test_groundtruths_bb_folder = "../dataset/labels/test"
test_images_folder = "../dataset/images/test"
name_images = []
classes = []
xs = []
ys = []
ws = []
hs = []
img_w_dim = []
img_h_dim = []
for label in os.listdir(test_groundtruths_bb_folder):
    name = label[:-4]
    with open(test_groundtruths_bb_folder+'/'+label, "r") as f:
        lines = f.readlines()
        if name+'.jpg' in os.listdir(test_images_folder):
            img = cv2.imread(test_images_folder+'/'+name+'.jpg')
            extension = '.jpg'
            print(f"Analizzando {name}{extension}")
        else :
            img = cv2.imread(test_images_folder+'/'+name+'.png')
            extension = '.png'
            print(f"Analizzando {name}{extension}")
            continue
        img_h, img_w = img.shape[:2]
        for line in lines:
            line_ls = line.split(" ")
            if line_ls == ["\n"]:
                continue
            name_images.append(name)
            classes.append("acino")
            w = int(float(line_ls[3]) * img_w)
            ws.append(w)
            h = int(float(line_ls[4]) * img_h)
            hs.append(h)
            x = int(float(line_ls[1]) * img_w - w / 2)
            xs.append(x)
            y = int(float(line_ls[2]) * img_h - h / 2)
            ys.append(y)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            img_w_dim.append(img_w)
            img_h_dim.append(img_h)

raw_data = {
    'image-name':name_images,
    'class':classes,
    'x':xs,
    'y':ys,
    'w':ws,
    'h':hs,
    'image-w-dim':img_w_dim,
    'image-h-dim':img_h_dim
}
data = pd.DataFrame(raw_data,columns=['image-name','class','x','y','w','h','image-w-dim','image-h-dim'])
data.to_csv("test_groundtruths_bb.csv",index=False)
