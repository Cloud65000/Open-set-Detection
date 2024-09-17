import torch
from transformers import BertModel

vis_path="/home/cxc/mm-grounddino/visualize_vocab/vocab.txt"
vocab_file=open(vis_path)
lines=vocab_file.readlines()
total_vocab=[]
for line in lines:
    line=line.replace("\n","").replace("\r","")
    # print(line)
    total_vocab.append(line)
print(total_vocab)
print(len(total_vocab))
#ours
prompts=['people', 'car', 'bicycle', 'pickup', 'bus', 'van', 'truck', 'train', 'tricycle', 'motorcycle', 'electric scooter','tire' , 'other vehicle',
        'reflector', 'crane', 'tanker','agricultural tractor', 'sprinkler', 'application vehicle', 'forklift', 'engineering vehicle', 'bulldozer', 
        'road roller', 'excavator', 'non-motor vehicle','hand vehicle']
# check_dict={"prompt":str,
#             "tf":bool
#             }
final_check=[]
for prompt in prompts:
    check_dict=dict()
    check_dict['prompt']=prompt
    if prompt in total_vocab:
        check_dict['tf']=True
    else:check_dict['tf']=False
    final_check.append(check_dict)
print(final_check)
#自翻译的prompt有一半不在词表范围内

#coco2odvg
ori_map = {
        '1': 'person',
        '2': 'bicycle',
        '3': 'car',
        '4': 'motorcycle',
        '5': 'airplane',
        '6': 'bus',
        '7': 'train',
        '8': 'truck',
        '9': 'boat',
        '10': 'traffic light',
        '11': 'fire hydrant',
        '13': 'stop sign',
        '14': 'parking meter',
        '15': 'bench',
        '16': 'bird',
        '17': 'cat',
        '18': 'dog',
        '19': 'horse',
        '20': 'sheep',
        '21': 'cow',
        '22': 'elephant',
        '23': 'bear',
        '24': 'zebra',
        '25': 'giraffe',
        '27': 'backpack',
        '28': 'umbrella',
        '31': 'handbag',
        '32': 'tie',
        '33': 'suitcase',
        '34': 'frisbee',
        '35': 'skis',
        '36': 'snowboard',
        '37': 'sports ball',
        '38': 'kite',
        '39': 'baseball bat',
        '40': 'baseball glove',
        '41': 'skateboard',
        '42': 'surfboard',
        '43': 'tennis racket',
        '44': 'bottle',
        '46': 'wine glass',
        '47': 'cup',
        '48': 'fork',
        '49': 'knife',
        '50': 'spoon',
        '51': 'bowl',
        '52': 'banana',
        '53': 'apple',
        '54': 'sandwich',
        '55': 'orange',
        '56': 'broccoli',
        '57': 'carrot',
        '58': 'hot dog',
        '59': 'pizza',
        '60': 'donut',
        '61': 'cake',
        '62': 'chair',
        '63': 'couch',
        '64': 'potted plant',
        '65': 'bed',
        '67': 'dining table',
        '70': 'toilet',
        '72': 'tv',
        '73': 'laptop',
        '74': 'mouse',
        '75': 'remote',
        '76': 'keyboard',
        '77': 'cell phone',
        '78': 'microwave',
        '79': 'oven',
        '80': 'toaster',
        '81': 'sink',
        '82': 'refrigerator',
        '84': 'book',
        '85': 'clock',
        '86': 'vase',
        '87': 'scissors',
        '88': 'teddy bear',
        '89': 'hair drier',
        '90': 'toothbrush'
    }
coco_classes=[]
coco_classes_check=[]
for key in ori_map.keys():
    coco_classes.append(key)
for i in coco_classes:
    check_dict_coco=dict()
    check_dict_coco['prompt']=i
    if i in total_vocab:
        check_dict_coco['tf']=True
    else:check_dict_coco['tf']=False
    coco_classes_check.append(check_dict_coco)
print(coco_classes_check)

#coco2odv
odv_classes=['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
                'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
                'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee',
                'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'microwave', 'oven', 'toaster',
                'refrigerator', 'book', 'clock', 'vase', 'toothbrush','airplane', 'bus', 'cat', 'dog', 'cow', 'elephant',
                 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife',
                 'cake', 'couch', 'keyboard', 'sink', 'scissors']
odv_classes_check=[]
for i in odv_classes:
    check_dict_odv=dict()
    check_dict_odv['prompt']=i
    if i in total_vocab:
        check_dict_odv['tf']=True
    else:check_dict_odv['tf']=False
    odv_classes_check.append(check_dict_odv)
print(odv_classes_check)

#obj365
obj_365classes=['']












