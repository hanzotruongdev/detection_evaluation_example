from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np
from tqdm import tqdm
from dataset import get_dataloader
from model import MyModel
from detection_metrics import ap_per_class, box_iou
from dataset import COCO_NAMES


device = 'cpu'

model = MyModel(device=device)
dataloader = get_dataloader()


if __name__=='__main__':
    """
    # run inference
    """
    stats = []
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc = len(COCO_NAMES)  # number of classes

    for batch_i, (imgs, annotations) in enumerate(tqdm(dataloader)):
        imgs = [i.to(device) for i in imgs]

        with torch.no_grad():
            # Run model
            out = model.predict(imgs)  

        # Statistics per image
        for i, pred in enumerate(out):

            # extract labels
            tcls = annotations[i]['labels']
            tbox = annotations[i]['boxes']

            # extract predictions
            pcls = pred['labels']
            pbox = pred['boxes']
            pscore = pred['scores'] 

            num_label = len(tbox)
            seen += 1

            if len(pred) == 0:
                if num_label:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = torch.zeros(pbox.shape[0], niou, dtype=torch.bool, device=device)
            if num_label:
                detected = []  # target indices

                # Per target class
                for cls in torch.unique(tcls):
                    ti = (cls == tcls).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pcls).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        best_ious, best_i = box_iou(pbox[pi], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (best_ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[best_i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = best_ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == num_label:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pscore.cpu(), pcls.cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    
    
    p, r, ap, f1, ap_class = ap_per_class(*stats)
    p, r, f1, ap50, ap = p[:, 0], r[:, 0], f1[:,0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    mp, mr, mf1, map50, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

    print('map@0.5:0.95 ', map, 'map@0.5 ', map50, 'f1: ', f1.item())