# -*- coding: utf-8 -*-
import requests
import threading
import cv2
import pdb
import torch
from label_fmerge import further_merge
# import debugpy
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import torch
import argparse
import numpy as np
from prettytable import PrettyTable
import glob
from metrics import (filter_label, merge_label, get_params, read_xml, mAP)
import OPEN_LABELS
from mmdet.apis import DetInferencer
from http.client import HTTPConnection
from torch.nn import DataParallel
from torch.cuda import device_count
HTTPConnection._http_vsn_str = "HTTP/1.0"

class Resquest(BaseHTTPRequestHandler):
    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())

    def do_GET(self):
        print(self.requestline)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        data = {'state': 0}

    # 接受post请求
    def do_POST(self):
        # 读取数据
        req_datas = self.rfile.read(int(self.headers['content-length']))
        print(req_datas.decode())

class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass

def Regist(nvidia_, port=1001):
    res = requests.post(f'http://{nvidia_}/info')
    print(nvidia_, port)
    host = ('0.0.0.0', port)
    myServer = ThreadingHttpServer(host, Resquest)
    myServer.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser(description='AlgoLibPlus test')
    parser.add_argument('--type', default='model-list',
                        help='test type : [image,video,upload-model,model-list]', required=True)
    parser.add_argument('--algo-type', default='object-detect',
                        help='algo type : [object-detect]', required=False)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids to use, e.g., "0,1,2,3"')
    parser.add_argument('--model-name', default='person-car',
                        help='you can use the type of \'model-list\' to get model name', required=False)
    parser.add_argument('--file-path', default='',
                        help="input test file path", required=False)
    parser.add_argument('--server-ip', default='10.10.1.184',
                        help='server address', required=False)
    # parser.add_argument(
    #     '--input_img', type=str, help='Input image file or folder path.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    # parser.add_argument(
    #     '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument('--config', help='config file', required=False)

    return parser.parse_args()
def init_inferencer():
    # pdb.set_trace()
    init_args=parse_args()
    
    inferencer_args={
        'model':init_args.config,
        'weights':init_args.weights,
        # 'device':init_args.device
    }
    # pdb.set_trace()
    if ',' in init_args.gpus:
        device_ids=list(map(int,init_args.gpus.split(',')))
        inferencer=DetInferencer(**inferencer_args)
        if hasattr(inferencer,'model') and isinstance(inferencer.model,torch.nn.Module):
            inferencer.model=DataParallel(
                inferencer.model,
                device_ids=device_ids
            )
            inferencer.model.to('cuda')
    else:
        inferencer=DetInferencer(**inferencer_args)
        inferencer.model.to('cuda')
    # inferencer_model.eval()
    return inferencer
def prompt_preprocess(prompt):
    assert isinstance(prompt,list)
    processed=""
    for i in range(len(prompt)):
        if i !=len(prompt)-1:
            processed+=prompt[i]+'.'+' '
        else:processed+=prompt[i]+'.'
    print(processed)
    return processed
def obj_detect(inferencer,batch_images, prompt , thred):
    if batch_images is None:
        ValueError("Image can not be read!")
    call_args={
        'inputs': batch_images,
        'texts': prompt,
        'pred_score_thr': thred,
        'batch_size': len(batch_images),  
        'show': False,
        'no_save_vis': True, 
        'no_save_pred': True,  
        'print_result': False  
    }
    # pdb.set_trace()**
    mid_results=inferencer(**call_args)
    results=mid_results['predictions']
    # scores=
    # print(results)
    candidates=[]
    fin_scores=[]
    fin_labels=[]
    for result in results:
        # for i in range(len(result)):
        labels=result['labels']
        # print(type(labels))
        scores=result['scores']
        # print(type(scores))
        bboxes=result['bboxes']
        # print(type(bboxes))
        cand_scores=[]
        cand_labels=[]
        cand_bboxes=[]
        assert len(labels)==len(scores) and len(scores)==len(bboxes)
        for j in range(len(scores)):
            if scores[j]>=thred:
                cand_bboxes.append(bboxes[j])
                cand_scores.append(scores[j])
                cand_labels.append(labels[j])
            else: continue
        # pdb.set_trace()**
        cand_bboxes=np.array(cand_bboxes)
        cand_scores=np.array(cand_scores)
        cand_labels=np.array(cand_labels)
        # print(type(cand_bboxes))
        candidates.append(cand_bboxes)
        fin_scores.append(cand_scores)
        fin_labels.append(cand_labels)
    return candidates,fin_scores,fin_labels
def load_imgs(paths,batch_size):
    images = []
    Targets=[]
    minTargets=[]
    mediumTargets=[]
    largeTargets=[]
    filter_names = []
    # pdb.set_trace() **
    names = OPEN_LABELS.open_names
    names_merge = OPEN_LABELS.open_names_merge
    names_merge=list(names_merge.items())
    names_merge=dict(names_merge[:22])
    label_merge = True
    if label_merge:
        names_merge_list = []
        for key in names_merge.keys():
            names_merge_list.append(key)
        # pdb.set_trace()**
        names_merge_dic = {k: v for k, v in enumerate(names_merge_list)} #所需
        names_dic = names_merge_dic#与上相同
        # pdb.set_trace()
        print(names_dic)
    for idx, path in enumerate(paths):
        image = cv2.imread(path)
        xml_path = 'xml'.join(path.replace('/images/','/labels/').rsplit(path.split('.')[-1], 1))
        Target = read_xml(xml_path, names)
        if label_merge:
            Target = merge_label(Target, names_merge, names_merge_dic, names)
        Target, minTarget, mediumTarget, largeTarget = filter_label(Target, filter_names, names_dic)
        print("idx,path")
        print(idx,path)
        if image is not None:
            images.append(image)
            Targets.append(Target)
            if len(minTarget)!=0:
                minTargets.append(minTarget)
            if len(mediumTarget)!=0:
                mediumTargets.append(mediumTarget)
            if len(largeTarget)!=0:
                largeTargets.append(largeTarget)
        if len(images)==batch_size:
            yield images,Targets,minTargets, mediumTargets, largeTargets
            images=[]
            Targets=[]
            minTargets = []
            mediumTargets = []
            largeTargets=[]
        
    if images:
        yield images,Targets,minTargets, mediumTargets, largeTargets
            
        
        

def mAP_metric(server_ip):
    filter_names = []  #列表中可指定需要测试的标签类别

    model_name='PersonCarTest_x-d-416-768-nonms-nvidia_20240418'
    # pdb.set_trace()
    val_data = '/home/cxc/mm-gdino/openness_checking_data/person_car_227'  #数据集测试格式和训练集一致
    # val_data = '/media/40T/cxc_mmdino/data/smoke_fire/test'
    
    names = OPEN_LABELS.open_names
    names_merge = OPEN_LABELS.open_names_merge
    label_merge = True
    further_merge_names=further_merge(names_merge)

    
    print(len(names))

    if val_data.split('/')[-1] == 'smoke_fire':
        iouv = torch.linspace(0.1, 0.55, 10)
    else:
        iouv = torch.linspace(0.25, 0.7, 10)  # iou vector for mAP@0.5:0.95
    
    niou = iouv.numel()
    verbose = True  #report mAP by class
    names_dic = {k: v for k, v in enumerate(names)}
    print(names_dic)
    
    if label_merge:
        names_merge_list = []
        for key in names_merge.keys():
            names_merge_list.append(key)
        names_merge_dic = {k: v for k, v in enumerate(names_merge_list)} #所需
        names_dic = names_merge_dic#与上相同
        # pdb.set_trace()
        print("names_dic is:")
        print(names_dic)

    Stats, minStats, mediumStats, largeStats = [], [], [], []
    paths = glob.glob(val_data + '/**/images/*', recursive=True)
    prompt=[ 'person', 'delivery person', 'lying' 'squatting', 'sitting', 
            'lying-sitting', 'lying prone', 'climbing', 'car', 'police car',
            'pickup-police-vehicle', 'van-police-vehicle', 'ambulance',
            'pickup', 'RV', 'caravan', 'truck', 'van', 'tractor', 'dump truck', 
            'pallet truck', 'flatbed', 'fence', 'rescue trailer', 'transport trailer',
            'construction vehicle trailer', 'roadside assistance', 'hazardous chemical vehicle',
            'forklift truck', 'construction vehicle', 'bulldozer', 'crane', 'road roller',
            'excavator', 'application vehicle', 'cement tanker',
            'Tanker trucks', 'Sprinklers', 'Concrete trucks - mixer type', 
            'Concrete trucks - earth pump type', 'Fire trucks', 'Refuse collection trucks', 
            'Vehicles in the public transport category', 'Coaches', 'Double-decker buses',
            'agricultural tractors', 'Bicycles', 'Bicycles - yellow', 'Bicycles - blue', 
            'Bicycles - green', 'non-motorised vehicles', 'Trolleys', 'Motorised scooters', 'Motorcycles',
            'Motorcycle - four-wheeled', 'Electric motorbike', 'Electric bicycle', 'Tricycle', 
            'Man-powered tricycle', 'Cargo tricycle', 'Passenger tricycle', 'Train', 'Other vehicle', 
            'tower crane','tire', 'reflectors']
    # prompt_g1=prompt[:22]
    # prompt_g2=prompt[22:44]
    # prompt_g3=prompt[44:]
    prompt_groups = [prompt[i:i + 22] for i in range(0, len(prompt), 22)]
    further_prompt=["person","car","motorcycle","bicycle","bus","train","truck","tricycle","bulldozer","crane","excavator","application vehicle","agricultural tractors","non-motorised vehicles","tower crane","tire","reflectors"]
    prompt_dict={index:value for index,value in enumerate(prompt) }
    prompt_in=prompt_preprocess(prompt)
    # pdb.set_trace()
    print(prompt_in)
    inferencer=init_inferencer()
    batch_size = 7
    for batch_images, batch_labels,minTarget, mediumTarget, largeTarget in load_imgs(paths,batch_size):
        cand_boxes,fin_labels,fin_scores = obj_detect(inferencer, batch_images, prompt_in , 0.15)
   
       
        # Preds= []
        # pdb.set_trace()**
        i=0
        for rects, cls, confs in zip(cand_boxes,fin_labels,fin_scores):
            Pred = []
            # i=0
            # pdb.set_trace()**
            j=0
            for rect, cl, conf in zip(rects, cls, confs):
                if not isinstance(Pred,list):
                    Pred=Pred.tolist()
                Pred.append([rect[0], rect[1], rect[2], rect[3], conf, cl])
                Pred = np.array(Pred)
                Pred, minPred, mediumPred, largePred = filter_label(Pred, filter_names, names_dic)
                if len(Pred) == 0:
                    nl = len(batch_labels[i])
                    tcls = batch_labels[i][:, 0].tolist() if nl else []  # target class
                    if nl:
                        Stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                else:
                    # while len(Stats)<=i:
                    #     Stats.append([])
                    # while len(Pred)<=i:
                    #     Pred.append([])
                    Stats = get_params(Stats, batch_labels[i], Pred, iouv, niou)
                if len(minTarget)>i:
                    if len(minPred) == 0:
                        nl = len(minTarget[i])
                        tcls = minTarget[i][:, 0].tolist() if nl else []  # target class
                        if nl:
                            minStats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    else:
                        if len(minTarget)>0:
                            # for i in range(len(minTarget)):
                            # while len(minStats)<=i:
                            #     minStats.append([])
                            # while len(minPred)<=i:
                            #     minPred.append([])
                            temp = get_params(minStats, minTarget[i], minPred, iouv, niou)
                            minStats=temp
                if len(mediumTarget)>i:
                    if len(mediumPred) == 0:
                        nl = len(mediumTarget[i])
                        tcls = mediumTarget[i][:, 0].tolist() if nl else []  # target class
                        if nl:
                            mediumStats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    else:
                        if len(mediumTarget)>0:
                            # for i in range(len(mediumTarget)):
                            # while len(mediumStats)<=i:
                            #     mediumStats.append([])
                            # while len(mediumPred)<=i:
                            #     mediumPred.append([])
                            temp = get_params(mediumStats, mediumTarget[i], mediumPred, iouv, niou)
                            mediumStats=temp
                if len(largeTarget)>i:
                    if len(largePred) == 0:
                        nl = len(largeTarget[i])
                        tcls = largeTarget[i][:, 0].tolist() if nl else []  # target class
                        if nl:
                            largeStats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    else:  
                        # pdb.set_trace()**
                        if len(largeTarget)>0:
                            # for i in range(len(largeTarget)):
                            # while len(largeStats)<=i:
                            #     largeStats.append([])
                            # while len(largePred)<=i:
                            #     largePred.append([])
                            temp=get_params(largeStats, largeTarget[i], largePred
                                            , iouv, niou)
                            largeStats=temp
                # pdb.set_trace()**
                j+=1
            i+=1
        #     Preds.append(Pred)
        # Preds = np.array(Preds)
        
        # if label_merge:
        #     Pred = merge_label(Pred, names_merge, names_merge_dic, names)
        
        
    ####################################输出格式###############################################    Contribute by Li jiahui on 2022.10.11
    x = PrettyTable()
    x.field_names = ["classes", "images", "objects", "mean_p", "mean_r", "map25", "map25-70"]
    seen = len(paths)
    predict_group_names=names_dic
    a=[]
    print("+++++")
    print(Stats)
    print("++++")
    print(predict_group_names)
    print("++++")
    mp, mr, map25, map, nt, p, r, ap25, ap, ap_class = mAP(Stats, predict_group_names)
    print(mp)
    if verbose and len(predict_group_names) > 1 and len(Stats):
        for i, c in enumerate(ap_class):
            a.append([predict_group_names[c],seen, nt[c], round(p[i],4),round(r[i],4), round(ap25[i],4), round(ap[i],4)])   
    a.append(["All", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4)])
    
    mp, mr, map25, map, nt, _, _, _, _, _ = mAP(minStats, predict_group_names)
    a.append(["minSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4)])
    
    mp, mr, map25, map, nt, _, _, _, _, _ = mAP(mediumStats, predict_group_names)
    a.append(["mediumSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4)])
    
    mp, mr, map25, map, nt, _, _, _, _, _ = mAP(largeStats, predict_group_names)
    a.append(["largeSize", seen, nt.sum(), round(mp,4), round(mr,4), round(map25,4), round(map,4)])
   
    for mu in a: 
        x.add_row(mu)

    print(x)

nvidia_85 = '192.168.2.161:9011'

if __name__ == '__main__':
    
    platform = nvidia_85

    mAP_metric(server_ip=platform)


