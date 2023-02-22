# acini-project

Progetto sul confronto tra reti neurali per l'individuazione di acini d'uva.
- DATASET

Il dataset utilizzato è stato creato da zero, etichettando manualmente immagini fornite tramite questo [software](https://github.com/heartexlabs/labelImg), per allenare le reti ho effettuato una suddivisione del dataset con queste percentuali: 50% training set, 17% validation set, 33% test set.

- RETI NEURALI

Le reti utilizzare sono [Yolov5](https://github.com/ultralytics/yolov5) e [Faster R-CNN](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline) 

- ALLENAMENTO

Andamento dell' allenamento yolov5:
![](https://github.com/Alex-Tommy/acini-project/blob/main/repo-images/yolo-report.png)

Andamento dll'allenamento di Faster R-CNN
![](https://github.com/Alex-Tommy/acini-project/blob/main/repo-images/fasterrcnn-report.png)

- RISULTATI

Per ottenere alcune metriche sul test set ho sfruttato questa [pagina](https://github.com/rafaelpadilla/review_object_detection_metrics) (il resto le ho calcolate io stesso), che calcola le performance in base alle bounding boxes groundtruths e quelle generate dalla rete.

Per generare le groundtruths ho utilizzato il file `groundtruths_bb.py`, per le bounding boxes di yolo bisogna usare `yolo_boxes_to_csv.py` mentre per faster r-cnn ho modificato il file `inference.py` presente nella repo citata precedentemente.

