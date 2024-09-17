import torch
pth_file="/home/cxc/mm-grounddino/weights/official/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
content=torch.load(pth_file)
# print("k={0},v={1}".format(content.keys(),content.values))
cvcv=content.values
# print(list(cvcv.keys()))
print(list(cvcv.values()))
