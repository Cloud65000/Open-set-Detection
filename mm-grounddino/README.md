# 用法说明
注：以下用法说明仅针对“使用烟火数据集对模型进行微调和测评”具体用法说明参照官方[英文版](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md)官方[中文版](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage_zh-CN.md)
## 安装

在按照 [get_started](../../docs/zh_cn/get_started.md) 一节的说明安装好 MMDet 之后，需要安装额外的依赖包：

```shell
cd $MMDETROOT

pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git"
```

请注意由于 LVIS 第三方库暂时不支持 numpy 1.24，因此请确保您的 numpy 版本符合要求。建议安装 numpy 1.23 版本。
## 说明
### 不同大小模型权重下载
可前往[mm_grounding_dino](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)或[grounding_dino](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino)按需下载不同大小的模型权重，如：
```shell
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth
```
## 数据准备
当前使用的微调数据集为烟火数据集。此处仅以烟火数据集为例进行说明，若需使用其他目标数据集，请按需调整。以下为具体数据准备的过程，若金针对烟火数据集，为方便直接使用，可直接用`traintestneed`内的数据

将项目克隆到本地后，在项目根目录创建data文件夹以及data下的smoke_fire文件夹，获取烟火数据集并存储到smoke_fire文件夹中。
### 图片数据命名规范
原始烟火数据集中可能存在包括但不仅限于以下问题：图片命名时名字中间有空格，图片后缀名不规范，如出现“.JPG"等大写的后缀名。项目中提供有命名规范代码，可调用执行`mm-grounddino/tools/firename_correct.py`中的函数。
### 数据集GT类别收集与翻译
由于官方发布的mm_grounding_dino暂时只支持英文版的文字prompt,故在进行测评和微调时，需要先收集所有目标类别并将其翻译成英文版。
```shell
python3 mm-grounddino/mmdet/.mim/tools/xml_class_collector.py
```
上述执行文件中的`xml_directory`，`output_file_path`请根据实际存储位置进行修改。
注：这里的翻译使用的是python中的translate,若已经有用其他翻译工具翻译出的英文类别文件则无需执行此代码，且应在`mm-grounddino/tools/dataset_converters/classmerge_smokefire.py`正则化匹配中做出相应修改。
### 数据集GT类别融合
在使用`mm-grounddino/demo/image_demo.py`进行单张图片烟火类别识别效果评测后，发现太过细粒度的烟火类别模型识别效果较差（可能也与直接中译英/细粒度类别中含有特殊字符'-'有关），故需要对目标类别进行类别融合。
```shell
python3 mm-grounddino/tools/dataset_converters/classmerge_smokefire.py
```
其中调用函数时传入的classes_path_en请根据实际存储位置进行修改
### 官方评测方式的COCO格式数据准备
为了方便起见，可先将`/mm-grounddino/data/smoke_fire/test`下所有图片和标注scp到该目录下的`images_all`,`label_all`中，执行如下命令，并按需替换相应参数
```shell
python3 mm-grounddino/tools/dataset_converters/xml2coco.py -i mm-grounddino/data/smoke_fire/test/images_all -x mm-grounddino/data/smoke_fire/test/label_all -c mm-grounddino/data/smoke_fire/test/your_merged_class_file.txt -o mm-grounddino/data/smoke_fire/test/self_defined_cocoanno_output_fie.json
```
### 官方微调方式的ODVG格式数据准备
原始烟火数据集中，每一张图片都对应有一个`.xml`的标注数据，需要将所有标注数据合并成单个odvg格式的json文件。
```shell
python3 mm-grounddino/tools/dataset_converters/firesmokecoco2odvg.py
```
上述执行文件中的`cocofile_path`，`map_path`，`output_file`请根据实际存储位置进行修改

## 模型评测
### 配置准备
针对烟火数据集，在[mm_grounding_dinoL*](grounding_dino_sein-l_finetune_smokefire_test.py),[grounding_dinob](grounding_dino_swin-b_finetune_16xb2_1x_firesmoke.py)的评测/微调配置文件已经准备好，其内部具体参数调整可按需修改。若需对其他数据集/其他模型进行配置文件编写，可参考[官方配置文件学习](https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/config.html)
单卡评测mm_grounding_dinoL*和groungding_dinob在烟火数据集上的效果时使用[algo_server](AlgoServerScript/algo_server.py)
单卡评测mm_grounding_dinoL*和groungding_dinob在开放人车数据集上的效果时使用[algo_server_gdino-b](AlgoServerScript/algo_server_gdino-b.py)
多卡评测mm_grounding_dinoL*和groungding_dinob在开放人车数据集上的效果时使用[algo_server_gdino-b_multithreds3*22.py](AlgoServerScript/algo_server_gdino-b_multithreds3*22.py)
评测命令框架相同，以评测烟火数据集在官方groundingdinoT权重上的表现效果为例：
```shell
mm-grounddino/AlgoServerScript/algo_server.py  --weights mm-grounddino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth  --config mm-grounddino/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_smokefire.py --type image
```
## 模型微调
模型微调命令遵循官方所给命令框架，此处以微调mm_groundingdinoL*为例
```shell
./tools/dist_train.sh mm-grounddino/configs/mm_grounding_dino/grounding_dino_sein-l_finetune_smokefire_test.py  8 --work-dir smokefirecf_work_dir
```