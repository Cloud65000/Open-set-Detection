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
import OPEN_LABELS
names_merge = OPEN_LABELS.open_names_merge
label_merge = True
if label_merge:
        names_merge_list = []
        for key in names_merge.keys():
            names_merge_list.append(key)
names_merge_dic = {k: v for k, v in enumerate(names_merge_list[:22])}
print(names_merge_dic)