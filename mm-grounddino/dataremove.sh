 cd /home/cxc/mmdetection-main/data/coco/train2017
 find . -type f | while read file; do if [ -e "../stuffthingmaps_trainval2017/train2017/${file}" ]; then rm "${file}"; fi; done
