import os
import xml.etree.ElementTree as ET
from translate import Translator

def collect_classes(xml_path,classes):
            tree = ET.parse(xml_path)
            root=tree.getroot()
            for obj in root.findall('object'):
                class_name=obj.find('name').text
                if class_name not in classes:
                    classes.append(class_name)
                # else:continue
def traverse_get_label_path(parent_dir,classes=[]):
    for whether_xml in os.listdir(parent_dir):
        entry_path=os.path.join(parent_dir, whether_xml)
        if os.path.isfile(entry_path) and entry_path.endswith('.xml'):
            collect_classes(entry_path,classes)        
        elif os.path.isdir(entry_path):
            traverse_get_label_path(entry_path,classes)
    return classes





translator= Translator(from_lang="ZH",to_lang="EN")
xml_directory="/home/cxc/mmdetection-main/data/smoke_fire/train/labels/"
unique_class=traverse_get_label_path(xml_directory)
# unique_class.append('')
print(unique_class)
output_file_path = '/home/cxc/mmdetection-main/data/smoke_fire/train/classes_finetune_en.txt'
with open(output_file_path, 'w', encoding='utf-8') as txt_file:
    for item in unique_class:
        translation_item = translator.translate(item)
        item_lower=translation_item.lower()
        txt_file.write(item_lower+'\n')
