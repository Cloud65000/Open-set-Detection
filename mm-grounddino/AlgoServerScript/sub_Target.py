import OPEN_LABELS
def get_Target_sub(wait_filt,names_merge,group_no,num):
    counter_l=0
    counter_sub1=0
    counter_sub2=0
    boundary_h=(group_no+1)*num
    boundary_l=group_no*num
    counter_r=boundary_l
    for k,vs in names_merge.items():       
        for v in vs:
            if counter_l<boundary_l:
                counter_sub1+=1
            else:
                counter_sub2=counter_sub1
                break
        if counter_l<boundary_l:
            counter_l+=1
        else:
            counter_r=counter_l
            counter_sub2=counter_sub1
            break
    names_merge=list(names_merge.items())
    # print(names_merge[0:2])
    # a=names_merge[boundary_l:boundary_h]
    names_merge=dict(names_merge[boundary_l:])
    for key,values in names_merge.items():       
        for v in values:
            if counter_r<boundary_h:
                counter_sub2+=1
            else:
                break
        if counter_r<boundary_h:
            counter_r+=1
        else:
            break 
    if wait_filt<counter_sub2 and wait_filt>=counter_sub1:
        return True
    else: return False

names_merge = OPEN_LABELS.open_names_merge
# names_merge=list(names_merge.items())
# names_merge=dict(names_merge[22:44])
Target1=[[212, 3, 572, 1149, 1011]]
Target2=[[49, 3, 572, 1149, 1011]]
Target=[]
a=get_Target_sub(Target2[0][0],names_merge,0,22)
print(a)
if a:
    Target=Target2
    print(Target)
else:
    print("cksl")