import OPEN_LABELS
def further_merge(initial_merge):
    further_merge_names={
        "人":[],
        "汽车":[],
        "摩托车":[],
        "自行车":[],
        "巴士":[],
        "火车":[],
        "货车":[],
        "三轮车":[],
    }
    person_merge=["人","外卖员","躺","蹲","坐","躺坐","趴","攀爬"]
    car_merge=["小汽车","小汽车-警车","房车","面包车-警车","皮卡","面包车","救护车","皮卡-警车"]
    motorcycle_merge=["电动摩托车","摩托车","摩托车-四轮"]
    bicycle_merge=["电动自行车","电动滑板车","自行车","自行车-黄色","自行车-蓝色","自行车-绿色"]
    bus_merge=["公共交通类车","大巴车","双层巴士"]
    train_merge=["火车"]
    truck_merge=["卡车","厢式","牵引车","渣土车","栏板式","平板式","仓栅式","危化车","叉车","工程车","水泥罐车","油罐车","洒水车","混凝土车-搅拌式","混凝土车-土泵式","消防车","垃圾车","救援拖车","运输拖车","工程车拖车","道路救援"]
    tri_merge=["三轮车","人力三轮车","货三轮车","客三轮车"]
    # others=["塔吊","轮胎","反光镜"]
    count=0
    counter=0
    for k,v in initial_merge.items():
        if k in person_merge:
            for i in v:
                further_merge_names["人"].append(i)
                count+=1
        elif k in car_merge:
            for j in v:
                further_merge_names['汽车'].append(j)
                count+=1
        elif k in motorcycle_merge:
            for m in v:
                further_merge_names["摩托车"].append(m)
                count+=1
        elif k in bicycle_merge:
            for b in v:
                further_merge_names["自行车"].append(b)
                count+=1
        elif k in bus_merge:
            for bus in v:
                further_merge_names["巴士"].append(bus)
                count+=1
        elif k in train_merge:
            for t in v:
                further_merge_names["火车"].append(t)
                count+=1
        elif k in truck_merge:
            for c in v: 
                further_merge_names["货车"].append(c)
                count+=1
        elif k in tri_merge:
            for c in v: 
                further_merge_names["三轮车"].append(c)
                count+=1
        else:
            further_merge_names[k]=v
            for o in v:
                count+=1
    # print(count)
    # for k,_ in further_merge_names.items():
    #     counter+=1
    # print(counter)

    return further_merge_names

# names = OPEN_LABELS.open_names
# names_merge = OPEN_LABELS.open_names_merge
# label_merge = True
# further_merge_names=further_merge(names_merge)
# print(further_merge_names)