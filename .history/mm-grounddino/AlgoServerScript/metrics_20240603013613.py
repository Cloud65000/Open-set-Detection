
"""
Model validation metrics
"""

# import math
# import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import xml.dom.minidom as minidom
from translate import Translator

def filter_label(label, filter_names, names):
    minSize, mediumSize = 30, 100            
    minLable, mediumLabel, largeLabel = [], [], [] 
    
    if len(label) == 0:
        return torch.from_numpy(np.array(label)), torch.from_numpy(np.array(minLable)), torch.from_numpy(np.array(mediumLabel)), torch.from_numpy(np.array(largeLabel))
    
    if len(label[0]) == 5:
        position = 0
        idx = 1
    elif len(label[0]) == 6:
        position = 5
        idx = 0

    if len(filter_names) != 0:
        filter_index = []
        names_list = list(names.values())
        for na in filter_names:
            filter_index.append(names_list.index(na))
            
        i = 0
        for lab in label:
            if lab[position].item() not in filter_index:
                label = np.delete(label,i,axis=0)
            else:
                i += 1
                
    for lab in label:      #判断bbox框定的是大目标中目标小目标
        w = lab[2+idx] - lab[0+idx] 
        h = lab[3+idx] - lab[1+idx]
        if w*h < minSize*minSize:  #bbox最小框900临界值
            minLable.append(np.array(lab))
        elif minSize*minSize <= w*h < mediumSize*mediumSize:
            mediumLabel.append(np.array(lab))
        else:
            largeLabel.append(np.array(lab))

    return torch.from_numpy(np.array(label)), torch.from_numpy(np.array(minLable)), torch.from_numpy(np.array(mediumLabel)), torch.from_numpy(np.array(largeLabel))

def merge_label(targets, merge_all_dict, merge_dic, names_dict):
    if len(targets) == 0:
        return targets
    
    if len(targets[0]) == 5:
        for target in targets:
            label = names_dict[target[0]]
            for key, value in merge_all_dict.items():
                if label in value:
                    for k, v in merge_dic.items():
                        if v == key:
                            target[0] = k
    elif len(targets[0]) == 6:
        for target in targets:
            label = names_dict[int(target[5])]
            for key, value in merge_all_dict.items():
                if label in value:
                    for k, v in merge_dic.items():
                        if v == key:
                            target[5] = float(k)
    else:
        print('label size error !')
        exit()
        
    return targets
def PL_match(pred,label,prompt_dict,names_dic):
    translator=Translator(from_lang="chinese",to_lang="english")

    for p,pvalue in prompt_dict.items():
        if p==pred:
            pvalue_rec=pvalue
    for l,lvalue in names_dic.items():
        if l==label:
            lvalue_rec=translator.translate(lvalue)
    if pvalue_rec==lvalue_rec:
        return True
    else:
        return False
def get_params(stats, Target, Pred, iouv, niou):
    nl = len(Target)
    tcls = Target[:, 0].tolist() if nl else []  # target class

    predn = Pred.clone()
    if nl:
        correct = process_batch(predn, Target, iouv)
    else:
        correct = torch.zeros(Pred.shape[0], niou, dtype=torch.bool)
    stats.append((correct, Pred[:, 4], Pred[:, 5], tcls))  # (correct, conf, pcls, tcls)
    
    return stats
        
def mAP(stats, names_dic):
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir='labels.jpg', names=names_dic)
        ap25, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map25, map = p.mean(), r.mean(), ap25.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names_dic))  # number of targets per class
    else:
        nt = torch.zeros(1)

    return mp, mr, map25, map, nt, p, r, ap25, ap, ap_class
        
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    print(detections[:,:4].shape)
    # print(iou, iou.shape)
    print(detections[:,5])
    print(labels[:,0:1])
    # whether_match=PL_match(detections[:, 5],labels[:, 0:1],prompt_dict,names_dic)

    x = torch.where((iou >= iouv[0]) & (detections[:, 5]==labels[:, 0:1]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def read_xml(xml_path, classes):
    dom_r = minidom.parse(xml_path)
    rot = dom_r.documentElement

    if rot.getElementsByTagName("size"):
        size = rot.getElementsByTagName("size")[0]
        width = int(size.getElementsByTagName('width')[0].firstChild.data)
        height = int(size.getElementsByTagName('height')[0].firstChild.data)
    else:
        print(xml_path,'----------- The xml file have not size attribute -----------')

    rects = []
    for ob in rot.getElementsByTagName("object"):
        cls = ob.getElementsByTagName('name')[0].firstChild.data
        cls = cls.strip()
        label = -1
        for i in range(len(classes)):
            if cls == classes[i]:
                label = i

        bndbox_r = ob.getElementsByTagName("bndbox")[0]
        xmin = bndbox_r.getElementsByTagName("xmin")[0].firstChild.data
        ymin = bndbox_r.getElementsByTagName("ymin")[0].firstChild.data
        xmax = bndbox_r.getElementsByTagName("xmax")[0].firstChild.data
        ymax = bndbox_r.getElementsByTagName("ymax")[0].firstChild.data

        xmin = int(xmin) if int(xmin)>0 else 0
        ymin = int(ymin) if int(ymin)>0 else 0
        xmax = int(xmax) if int(xmax)<width else width-1  #防止越界
        ymax = int(ymax) if int(ymax)<height else height-1

        w = xmax - xmin
        h = ymax - ymin
        c_x = xmin + w/2
        c_y = ymin + h/2

        if c_x/width >=1 or c_y/height >=1:
            print(xml_path,'----------- The coordinates exceed the picture boundary -----------')

        if label == -1:
            print(classes)
            print(cls, '----------- The label not in classes -----------')
            exit()

        rect = [label, xmin, ymin, xmax, ymax]

        rects.append(rect)

    return rects

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
