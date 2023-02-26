# acini-project

Progetto sul confronto tra reti neurali per l'individuazione di acini d'uva.
## DATASET

Il dataset utilizzato è stato creato da zero, etichettando manualmente immagini fornite tramite questo [software](https://github.com/heartexlabs/labelImg), per allenare le reti ho effettuato una suddivisione del dataset con queste percentuali: 50% training set, 17% validation set, 33% test set.

## RETI NEURALI

Le reti utilizzare sono [Yolov5](https://github.com/ultralytics/yolov5) e [Faster R-CNN](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline) 

## ALLENAMENTO

Andamento dell'allenamento yolov5:
![](https://github.com/Alex-Tommy/acini-project/blob/main/repo-images/yolo_differences.png)

Andamento dell'allenamento di Faster R-CNN
![](https://github.com/Alex-Tommy/acini-project/blob/main/repo-images/faster_differences.png)

## RISULTATI METRICHE

Per ottenere alcune metriche sul test set ho usufruito di alcune funzioni di questa [pagina](https://github.com/rafaelpadilla/review_object_detection_metrics), il resto le ho calcolate io stesso utilizzando il file `plot_data.py`. Per ottenere i file csv (le cui righe descrivono la posizione di una bounding box) necessari per utilizzare `plot_data.py` bisogna eseguire :
- `groundtruths_bb.py`
- `yolo_boxes_to_csv.py`
- `inference.py`, questo è il file che è presente nella repo di faster R-CNN modificato da me per ottenere il file csv

Ecco i risultati delle performance

<table cellspacing="2" cellpadding="2" width="1200" border="0">
<tbody>
<tr>
<td valign="center" width="400"><img src="repo-images/AP.png"></td>
<td valign="center" width="400"><img src="repo-images/Accuracy.png"></td>
</tr>
<tr>
<td valign="center" width="400"><img src="repo-images/Precision.png"></td>
<td valign="center" width="400"><img src="repo-images/Recall.png"></td>
</tr>
<tr>
<td valign="center" width="400"><img src="repo-images/F1.png"></td>
<td valign="center" width="400"><img src="repo-images/pr_curve.png"></td>
</tr>
</tbody>
</table>

## RISULTATI IMMAGINI

Confronto tra le diversi reti di backbone di Faster R-CNN, in rosso ResNet50, in blu Mobilenetv3, in verde SqueezeNet

<table cellspacing="3" cellpadding="3" width="900" border="0">
<tbody>
<tr>
<td valign="center" width="300"><img src="repo-images/detected_resnet.jpg"></td>
<td valign="center" width="300"><img src="repo-images/detected_mobilenet.jpg"></td>
<td valign="center" width="300"><img src="repo-images/detected_squeezenet.jpg"></td>
</tr>
</tbody>
</table>



