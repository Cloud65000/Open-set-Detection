import os
import json
import re
def list_files(directory):
    counti=0
    for root,dirs,files in os.walk(directory):
        for name in files:
            # print(os.path.join(root,name))
            x=os.path.join(root,name)
            if name.endswith('JPG'):
                namey=name.replace('JPG','jpg')
                y=os.path.join(root,namey)
                os.rename(x,y)
                counti+=1
            if ' ' in name:
                namez=name.replace(' ','')
                z=os.path.join(root,namez)
                os.rename(x,z)
                counti+=1
                print(str(counti)+" files has been renamed")
            else: continue
        for name in dirs:
            print(os.path.join(root,name))
# directory="/home/cxc/mmdetection-main/data/smoke_fire/train/images"            
# list_files(directory)
def json_filename_correct(direc,direcnew):
    with open(direc, 'r') as f:
        data = json.load(f)
    wait_correct=data["images"]
    count_unformal=0
    for item in wait_correct:
        if "file_name" in item:
            new_file_name = re.sub(r"\s", "", item["file_name"])
            if new_file_name != item["file_name"]:
                count_unformal += 1
                item["file_name"] = new_file_name
    print(count_unformal)
    with open(direcnew, 'w') as f:
        json.dump(data, f, indent=4)
json_filename_correct("/home/cxc/mm-grounddino/data/smoke_fire/train/coco_annotations_v1_ftmerge.json","/home/cxc/mm-grounddino/data/smoke_fire/train/coco_annotations_v1_ftmerge.json")