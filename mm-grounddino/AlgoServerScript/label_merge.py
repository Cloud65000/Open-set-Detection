import OPEN_LABELS
def merge_label(targets, merge_all_dict, merge_dic, names_dict):
    if len(targets) == 0:
        return targets
    merge_Targets=[]
    if len(targets[0]) == 5:
        for target in targets:
            label = names_dict[target[0]]
            for key, value in merge_all_dict.items():
                if label in value:
                    for k, v in merge_dic.items():
                        if v == key:
                            target[0] = k
                            merge_Targets.append(target)
                # else:
                #     continue
    elif len(targets[0]) == 6:
        for target in targets:
            label = names_dict[int(target[5])]
            for key, value in merge_all_dict.items():
                if label in value:
                    for k, v in merge_dic.items():
                        if v == key:
                            target[5] = float(k)
                            merge_Targets.append(target)
                # else:
                #     continue

    else:
        print('label size error !')
        exit()
    print(merge_Targets)
    return merge_Targets
targets=[[49, 3, 572, 1149, 1011]]
merge_all_dict=OPEN_LABELS.open_names_merge
names_merge_list = []
for key in merge_all_dict.keys():
    names_merge_list.append(key)
merge_dic = {k: v for k, v in enumerate(names_merge_list)} #所需
names_dict = OPEN_LABELS.open_names
merge_label(targets, merge_all_dict, merge_dic, names_dict)
