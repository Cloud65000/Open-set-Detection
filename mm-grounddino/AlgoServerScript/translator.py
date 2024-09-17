from translate import Translator
import pdb

wait_translate=['人','外卖员','躺''蹲','坐', '躺坐', '趴', '攀爬', '小汽车', '小汽车-警车','皮卡-警车', '面包车-警车', '救护车',
                '皮卡', '面包车', '房车', '卡车', '厢式', '牵引车', '渣土车', '栏板式', '平板式', '仓栅式', '救援拖车', '运输拖车',
                '工程车拖车', '道路救援', '危化车', '叉车', '工程车', '推土机', '吊车', '压路机', '挖掘机', '应用类车', '水泥罐车',
                '油罐车', '洒水车', '混凝土车-搅拌式', '混凝土车-土泵式', '消防车', '垃圾车', '公共交通类车', '大巴车', '双层巴士',
                '农用拖拉机', '自行车', '自行车-黄色', '自行车-蓝色', '自行车-绿色', '非机动车', '手推车', '电动滑板车', '摩托车',
                '摩托车-四轮', '电动摩托车', '电动自行车', '三轮车', '人力三轮车', '货三轮车', '客三轮车', '火车', '其他车', '塔吊',
                '轮胎', '反光镜']
translator=Translator(from_lang="ZH",to_lang="EN")
translated_one=[]
for item in wait_translate:
    # pdb.set_trace()
    t=translator.translate(item)
    translated_one.append(t)
    print(t)
print(translated_one)
