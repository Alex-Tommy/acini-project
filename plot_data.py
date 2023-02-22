from BoundingBox import *
from BoundingBoxes import *
from Evaluator import *
import os
import cv2
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

# Le parti commentate fanno riferimento a risultati di altre reti

def plot_pr_curve(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #ax.plot(yolov5n[0],yolov5n[1],color="r",label="Yolov5n")
    #ax.plot(yolov5s[0],yolov5s[1],color="b",label="Yolov5s")
    ax.plot(yolov5m[0],yolov5m[1],color="g",label="Yolov5m")
    ax.plot(resnet[0],resnet[1],color="y",label="ResNet50")
    #ax.plot(mobilenet[0],mobilenet[1],color="orange",label="MobileNet")
    #ax.plot(squeezenet[0],squeezenet[1],color="gray",label="SqueezeNet")
    ax.set_title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_AP(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #models = ["yolov5n","yolov5s","yolov5m","resnet50","mobilenetv3","squeezenet"]
    #res = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    models = ["yolov5m","resnet50"]
    res = [yolov5m,resnet]
    ax.bar(models,res)
    for i in range(len(models)):
        plt.text(i,res[i],res[i])
    ax.set_title("AP")
    plt.ylim((0.7,1))
    plt.show()

def plot_Prec(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #models = ["yolov5n","yolov5s","yolov5m","resnet50","mobilenetv3","squeezenet"]
    #res = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    models = ["yolov5m","resnet50"]
    res = [yolov5m,resnet]
    ax.bar(models,res)
    for i in range(len(models)):
        plt.text(i,res[i],res[i])
    ax.set_title("Precision")
    plt.ylim((0.4,0.9))
    plt.show()

def plot_Rec(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #models = ["yolov5n","yolov5s","yolov5m","resnet50","mobilenetv3","squeezenet"]
    #res = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    models = ["yolov5m","resnet50"]
    res = [yolov5m,resnet]
    ax.bar(models,res)
    for i in range(len(models)):
        plt.text(i,res[i],res[i])
    ax.set_title("Recall")
    plt.ylim((0.7,1.0))
    plt.show()

def plot_F1(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #models = ["yolov5n","yolov5s","yolov5m","resnet50","mobilenetv3","squeezenet"]
    #res = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    models = ["yolov5m","resnet50"]
    res = [yolov5m,resnet]
    ax.bar(models,res)
    for i in range(len(models)):
        plt.text(i,res[i],res[i])
    ax.set_title("F1")
    plt.ylim((0.5,0.95))
    plt.show()

def plot_CM(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    #cm = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    cm = [yolov5m,resnet]
    x_axis_labels = ["acino","background"] 
    y_axis_labels = ["acino","background"]
    for el in cm: 
        sns.heatmap(el, annot=True, fmt='.4g',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

def plot_Acc(yolov5n=None,yolov5s=None,yolov5m=None,resnet=None,mobilenet=None,squeezenet=None):
    fig, ax = plt.subplots()
    #models = ["yolov5n","yolov5s","yolov5m","resnet50","mobilenetv3","squeezenet"]
    #res = [yolov5n,yolov5s,yolov5m,resnet,mobilenet,squeezenet]
    models = ["yolov5m","resnet50"]
    res = [yolov5m,resnet]
    ax.bar(models,res)
    for i in range(len(models)):
        plt.text(i,res[i],res[i])
    ax.set_title("Accuracy")
    plt.ylim((0.4,0.9))
    plt.show()

# Mostra a schermo i risultati di tutte le reti
def display_all_data():
    groundtruths = pd.read_csv("test_groundtruths_bb.csv")
    yolov5n = pd.read_csv("yolov5n_val_detected_bb.csv")
    yolov5s = pd.read_csv("yolov5s_val_detected_bb.csv")
    yolov5m = pd.read_csv("yolov5m_val_detected_bb.csv")
    resnet50 = pd.read_csv("resnet50_val_detected_bb.csv")
    mobilenetv3 = pd.read_csv("mobilenetv3_val_detected_bb.csv")
    squeezenet = pd.read_csv("squeezenet_val_detected_bb.csv")
    bb_yolov5n = BoundingBoxes()
    bb_yolov5s = BoundingBoxes()
    bb_yolov5m = BoundingBoxes()
    bb_resnet = BoundingBoxes()
    bb_mobilenet = BoundingBoxes()
    bb_squeezenet = BoundingBoxes()
    for index, row in groundtruths.iterrows():
        bb_yolov5n.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_yolov5s.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_yolov5m.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_resnet.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_mobilenet.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_squeezenet.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
    gt_bb = len(bb_resnet._boundingBoxes)
    #print(f"Number of groundtruths bounding boxes: {gt_bb}")
    for index, row in yolov5n.iterrows():
        bb_yolov5n.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in yolov5s.iterrows():
        bb_yolov5s.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in yolov5m.iterrows():
        bb_yolov5m.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in resnet50.iterrows():
        bb_resnet.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in mobilenetv3.iterrows():
        bb_mobilenet.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in squeezenet.iterrows():
        bb_squeezenet.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    evaluator = Evaluator()
    yolov5n_eval = evaluator.GetPascalVOCMetrics(bb_yolov5n,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    yolov5n_p = yolov5n_eval[0]["interpolated precision"]
    yolov5n_r = yolov5n_eval[0]["interpolated recall"]
    yolov5n_AP = round(yolov5n_eval[0]["AP"],3)
    yolov5n_prec = round(yolov5n_eval[0]["total TP"] / (yolov5n_eval[0]["total TP"] + yolov5n_eval[0]["total FP"]),3)
    yolov5n_rec = round(yolov5n_eval[0]["total TP"] / yolov5n_eval[0]["total positives"] ,3)
    yolov5n_f1 = round(2*(yolov5n_prec * yolov5n_rec) / (yolov5n_prec + yolov5n_rec),3)
    yolov5n_cm = np.array([[yolov5n_eval[0]["total TP"],yolov5n_eval[0]["total FP"]],[gt_bb - yolov5n_eval[0]["total TP"],0]])
    yolov5n_acc = round(yolov5n_eval[0]["total TP"] / (yolov5n_eval[0]["total positives"] + yolov5n_eval[0]["total FP"]),3)
    yolov5s_eval = evaluator.GetPascalVOCMetrics(bb_yolov5s,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    yolov5s_p = yolov5s_eval[0]["interpolated precision"]
    yolov5s_r = yolov5s_eval[0]["interpolated recall"]
    yolov5s_AP = round(yolov5s_eval[0]["AP"],3)
    yolov5s_prec = round(yolov5s_eval[0]["total TP"] / (yolov5s_eval[0]["total TP"] + yolov5s_eval[0]["total FP"]),3)
    yolov5s_rec = round(yolov5s_eval[0]["total TP"] / yolov5s_eval[0]["total positives"] ,3)
    yolov5s_f1 = round(2*(yolov5s_prec * yolov5s_rec) / (yolov5s_prec + yolov5s_rec),3)
    yolov5s_cm = np.array([[yolov5s_eval[0]["total TP"],yolov5s_eval[0]["total FP"]],[gt_bb - yolov5s_eval[0]["total TP"],0]])
    yolov5s_acc = round(yolov5s_eval[0]["total TP"] / (yolov5s_eval[0]["total positives"] + yolov5s_eval[0]["total FP"]),3)
    yolov5m_eval = evaluator.GetPascalVOCMetrics(bb_yolov5m,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    yolov5m_p = yolov5m_eval[0]["interpolated precision"]
    yolov5m_r = yolov5m_eval[0]["interpolated recall"]
    yolov5m_AP = round(yolov5m_eval[0]["AP"],3)
    yolov5m_prec = round(yolov5m_eval[0]["total TP"] / (yolov5m_eval[0]["total TP"] + yolov5m_eval[0]["total FP"]),3)
    yolov5m_rec = round(yolov5m_eval[0]["total TP"] / yolov5m_eval[0]["total positives"] ,3)
    yolov5m_f1 = round(2*(yolov5m_prec * yolov5m_rec) / (yolov5m_prec + yolov5m_rec),3)
    yolov5m_cm = np.array([[yolov5m_eval[0]["total TP"],yolov5m_eval[0]["total FP"]],[gt_bb - yolov5m_eval[0]["total TP"],0]])
    yolov5m_acc = round(yolov5m_eval[0]["total TP"] / (yolov5m_eval[0]["total positives"] + yolov5m_eval[0]["total FP"]),3)
    resnet_eval = evaluator.GetPascalVOCMetrics(bb_resnet,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    resnet_p = resnet_eval[0]["interpolated precision"]
    resnet_r = resnet_eval[0]["interpolated recall"]
    resnet_AP = round(resnet_eval[0]["AP"],3)
    resnet_prec = round(resnet_eval[0]["total TP"] / (resnet_eval[0]["total TP"] + resnet_eval[0]["total FP"]),3)
    resnet_rec = round(resnet_eval[0]["total TP"] / resnet_eval[0]["total positives"] ,3)
    resnet_f1 = round(2*(resnet_prec * resnet_rec) / (resnet_prec + resnet_rec),3)
    resnet_cm = np.array([[resnet_eval[0]["total TP"],resnet_eval[0]["total FP"]],[gt_bb - resnet_eval[0]["total TP"],0]])
    resnet_acc = round(resnet_eval[0]["total TP"] / (resnet_eval[0]["total positives"] + resnet_eval[0]["total FP"]),3)
    mobilenet_eval = evaluator.GetPascalVOCMetrics(bb_mobilenet,
                                                  IOUThreshold=0.5,
                                                  method=MethodAveragePrecision.EveryPointInterpolation)
    mobilenet_p = mobilenet_eval[0]["interpolated precision"]
    mobilenet_r = mobilenet_eval[0]["interpolated recall"]
    mobilenet_AP = round(mobilenet_eval[0]["AP"],3)
    mobilenet_prec = round(mobilenet_eval[0]["total TP"] / (mobilenet_eval[0]["total TP"] + mobilenet_eval[0]["total FP"]),3)
    mobilenet_rec = round(mobilenet_eval[0]["total TP"] / mobilenet_eval[0]["total positives"] ,3)
    mobilenet_f1 = round(2*(mobilenet_prec * mobilenet_rec) / (mobilenet_prec + mobilenet_rec),3)
    mobilenet_cm = np.array([[mobilenet_eval[0]["total TP"],mobilenet_eval[0]["total FP"]],[gt_bb - mobilenet_eval[0]["total TP"],0]])
    mobilenet_acc = round(mobilenet_eval[0]["total TP"] / (mobilenet_eval[0]["total positives"] + mobilenet_eval[0]["total FP"]),3)
    squeezenet_eval = evaluator.GetPascalVOCMetrics(bb_squeezenet,
                                                   IOUThreshold=0.5,
                                                   method=MethodAveragePrecision.EveryPointInterpolation)
    squeezenet_p = squeezenet_eval[0]["interpolated precision"]
    squeezenet_r = squeezenet_eval[0]["interpolated recall"]
    squeezenet_AP = round(squeezenet_eval[0]["AP"],3)
    squeezenet_prec = round(squeezenet_eval[0]["total TP"] / (squeezenet_eval[0]["total TP"] + squeezenet_eval[0]["total FP"]),3)
    squeezenet_rec = round(squeezenet_eval[0]["total TP"] / squeezenet_eval[0]["total positives"] ,3)
    squeezenet_f1 = round(2*(squeezenet_prec * squeezenet_rec) / (squeezenet_prec + squeezenet_rec),3)
    squeezenet_cm = np.array([[squeezenet_eval[0]["total TP"],squeezenet_eval[0]["total FP"]],[gt_bb - squeezenet_eval[0]["total TP"],0]])
    squeezenet_acc = round(squeezenet_eval[0]["total TP"] / (squeezenet_eval[0]["total positives"] + squeezenet_eval[0]["total FP"]),3)

    #plot_pr_curve(yolov5n=(yolov5n_r,yolov5n_p),yolov5s=(yolov5s_r,yolov5s_p),yolov5m=(yolov5m_r,yolov5m_p),
                    #resnet=(resnet_r,resnet_p),mobilenet=(mobilenet_r,mobilenet_p),squeezenet=(squeezenet_r,squeezenet_p))
    #plot_AP(yolov5n=yolov5n_AP,yolov5s=yolov5s_AP,yolov5m=yolov5m_AP,resnet=resnet_AP,mobilenet=mobilenet_AP,squeezenet=squeezenet_AP)
    #plot_Prec(yolov5n=yolov5n_prec,yolov5s=yolov5s_prec,yolov5m=yolov5m_prec,resnet=resnet_prec,mobilenet=mobilenet_prec,squeezenet=squeezenet_prec)
    #plot_Rec(yolov5n=yolov5n_rec,yolov5s=yolov5s_rec,yolov5m=yolov5m_rec,resnet=resnet_rec,mobilenet=mobilenet_rec,squeezenet=squeezenet_rec)
    #plot_F1(yolov5n=yolov5n_f1,yolov5s=yolov5s_f1,yolov5m=yolov5m_f1,resnet=resnet_f1,mobilenet=mobilenet_f1,squeezenet=squeezenet_f1)
    #plot_CF(yolov5n=yolov5n_cf,yolov5s=yolov5s_cf,yolov5m=yolov5m_cf,resnet=resnet_cf,mobilenet=mobilenet_cf,squeezenet=squeezenet_cf)
    #plot_Acc(yolov5n=yolov5n_acc,yolov5s=yolov5s_acc,yolov5m=yolov5m_acc,resnet=resnet_acc,mobilenet=mobilenet_acc,squeezenet=squeezenet_acc)

    
# Mostra a schermo i risultati di yolov5m e resnet50
def display_less_data():
    groundtruths = pd.read_csv("test_groundtruths_bb.csv")
    yolov5m = pd.read_csv("yolo_test_detected_bb.csv")
    resnet50 = pd.read_csv("faster_test_detected_bb.csv")
    bb_yolov5m = BoundingBoxes()
    bb_resnet = BoundingBoxes()
    for index, row in groundtruths.iterrows():
        bb_yolov5m.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
        bb_resnet.addBoundingBox(BoundingBox(row["image-name"],
                                              row["class"],
                                              row["x"],
                                              row["y"],
                                              row["w"],
                                              row["h"],
                                              typeCoordinates=CoordinatesType.Absolute,
                                              imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                              bbType=BBType.GroundTruth,
                                              classConfidence=None,
                                              format=BBFormat.XYWH))
    gt_bb = len(bb_resnet._boundingBoxes)
    for index, row in yolov5m.iterrows():
        bb_yolov5m.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    for index, row in resnet50.iterrows():
        bb_resnet.addBoundingBox(BoundingBox(row["image-name"],
                                      row["class"],
                                      row["x"],
                                      row["y"],
                                      row["w"],
                                      row["h"],
                                      typeCoordinates=CoordinatesType.Absolute,
                                      imgSize=(row["image-w-dim"],row["image-h-dim"]),
                                      bbType=BBType.Detected,
                                      classConfidence=row["confidence"],
                                      format=BBFormat.XYWH))
    evaluator = Evaluator()
    yolov5m_eval = evaluator.GetPascalVOCMetrics(bb_yolov5m,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    yolov5m_p = yolov5m_eval[0]["interpolated precision"]
    yolov5m_r = yolov5m_eval[0]["interpolated recall"]
    yolov5m_AP = round(yolov5m_eval[0]["AP"],3)
    yolov5m_prec = round(yolov5m_eval[0]["total TP"] / (yolov5m_eval[0]["total TP"] + yolov5m_eval[0]["total FP"]),3)
    yolov5m_rec = round(yolov5m_eval[0]["total TP"] / yolov5m_eval[0]["total positives"] ,3)
    yolov5m_f1 = round(2*(yolov5m_prec * yolov5m_rec) / (yolov5m_prec + yolov5m_rec),3)
    yolov5m_cm = np.array([[yolov5m_eval[0]["total TP"],yolov5m_eval[0]["total FP"]],[gt_bb - yolov5m_eval[0]["total TP"],0]])
    yolov5m_acc = round(yolov5m_eval[0]["total TP"] / (yolov5m_eval[0]["total positives"] + yolov5m_eval[0]["total FP"]),3)
    resnet_eval = evaluator.GetPascalVOCMetrics(bb_resnet,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
    resnet_p = resnet_eval[0]["interpolated precision"]
    resnet_r = resnet_eval[0]["interpolated recall"]
    resnet_AP = round(resnet_eval[0]["AP"],3)
    resnet_prec = round(resnet_eval[0]["total TP"] / (resnet_eval[0]["total TP"] + resnet_eval[0]["total FP"]),3)
    resnet_rec = round(resnet_eval[0]["total TP"] / resnet_eval[0]["total positives"] ,3)
    resnet_f1 = round(2*(resnet_prec * resnet_rec) / (resnet_prec + resnet_rec),3)
    resnet_cm = np.array([[resnet_eval[0]["total TP"],resnet_eval[0]["total FP"]],[gt_bb - resnet_eval[0]["total TP"],0]])
    resnet_acc = round(resnet_eval[0]["total TP"] / (resnet_eval[0]["total positives"] + resnet_eval[0]["total FP"]),3)

    plot_pr_curve(yolov5n=None,yolov5s=None,yolov5m=(yolov5m_r,yolov5m_p),resnet=(resnet_r,resnet_p),mobilenet=None,squeezenet=None)
    plot_AP(yolov5n=None,yolov5s=None,yolov5m=yolov5m_AP,resnet=resnet_AP,mobilenet=None,squeezenet=None)
    plot_Prec(yolov5n=None,yolov5s=None,yolov5m=yolov5m_prec,resnet=resnet_prec,mobilenet=None,squeezenet=None)
    plot_Rec(yolov5n=None,yolov5s=None,yolov5m=yolov5m_rec,resnet=resnet_rec,mobilenet=None,squeezenet=None)
    plot_F1(yolov5n=None,yolov5s=None,yolov5m=yolov5m_f1,resnet=resnet_f1,mobilenet=None,squeezenet=None)
    plot_CM(yolov5n=None,yolov5s=None,yolov5m=yolov5m_cm,resnet=resnet_cm,mobilenet=None,squeezenet=None)
    plot_Acc(yolov5n=None,yolov5s=None,yolov5m=yolov5m_acc,resnet=resnet_acc,mobilenet=None,squeezenet=None)
                      
display_less_data()

