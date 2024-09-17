# def load_imgs(paths,batch_size):
#     images = []
#     for idx, path in enumerate(paths):
#         image = cv2.imread(path)

#         print("idx,path")
#         print(idx,path)
#         if image is not None:
#             images.append(image)
#         if len(images)==batch_size:
#             yield images
#             images=[]
#     if images:
#         yield images

# import OPEN_LABELS
# names_merge = OPEN_LABELS.open_names_merge
# label_merge = True
# if label_merge:
#         names_merge_list = []
#         for key in names_merge.keys():
#             names_merge_list.append(key)
# names_merge_dic = {k: v for k, v in enumerate(names_merge_list[:22])}
# print(names_merge_dic)

import OPEN_LABELS
names=OPEN_LABELS.open_names
count=0
open_names_merged_f=OPEN_LABELS.open_names_mergeed_b1

names_set=[]
keys_set=[]
for key,values in open_names_merged_f.items():
    for value in values:
        # if value not in names_set:
        names_set.append(value)
    # if key not in keys_set:
    keys_set.append(key)
print(len(names_set))
print(keys_set)
print(len(keys_set))
