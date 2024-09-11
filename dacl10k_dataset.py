from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import os 
from PIL import ImageDraw,Image
import json
import numpy as np
from tqdm import tqdm

TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars',
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

def polygon_to_bbox(polygon):
    polygon_np = np.array(polygon)
    x_min = np.min(polygon_np[:, 0])
    y_min = np.min(polygon_np[:, 1])
    x_max = np.max(polygon_np[:, 0])
    y_max = np.max(polygon_np[:, 1])
    return [x_min, y_min, x_max, y_max]


def get_dacl(split):
    if split in ['train','val']:
        rets=[]
        annotations_filepath=os.path.join(MetadataCatalog.get('dacl10k_train').dataset_root,f'annotations/{split}')
        images_filepath = os.path.join(MetadataCatalog.get('dacl10k_train').dataset_root,f'images/{split}')
        annotations_files=os.listdir(annotations_filepath)
        for annotations_file in annotations_files:
            ret={}
            annotation = json.load(open(os.path.join(annotations_filepath,annotations_file),'r'))
            annotation_seg=[]
            for shape in annotation['shapes']:
                category_id=TARGET_LIST.index(shape['label'])
                bbox = polygon_to_bbox(shape['points'])
                bbox_mode = BoxMode.XYXY_ABS
                segmentation = []
                for x,y in shape['points']:
                    segmentation.append(x)
                    segmentation.append(y)

                annotation_seg.append({
                    'bbox':bbox,
                    'bbox_mode':bbox_mode,
                    'category_id':category_id,
                    'segmentation':[segmentation]
                })
            ret={
                'height':annotation['imageHeight'],
                'width':annotation['imageWidth'],
                'file_name':os.path.join(images_filepath,annotation['imageName']),
                'image_id':annotation['imageName'].split('.')[0].split('_')[-1],
                'annotations':annotation_seg
            }
            rets.append(ret)
        return rets
    elif split == 'test':
        # 返回测试集的数据
        pass
    else:
        raise ValueError(f"Unknown split: {split}")

# 注册训练集和测试集
DatasetCatalog.register('dacl10k_train', lambda: get_dacl('train'))
DatasetCatalog.register('dacl10k_val', lambda: get_dacl('val'))
# DatasetCatalog.register('dacl10l_test',lambda: get_dacl('test'))
MetadataCatalog.get('dacl10k_train').dataset_root='./dacl10k_v2_devphase'
MetadataCatalog.get('dacl10k_val').dataset_root='./dacl10k_v2_devphase'
MetadataCatalog.get('dacl10k_val').evaluator_type='coco'
MetadataCatalog.get('dacl10k_val').thing_classes=TARGET_LIST
# MetadataCatalog.get('dacl10k_test').dataset_root='./dacl10k_v2_devphase'

if __name__ == '__main__':
    dacl10k_train = DatasetCatalog.get('dacl10k_train')