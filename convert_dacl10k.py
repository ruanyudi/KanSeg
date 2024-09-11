import json
import os 
from PIL import Image
from tqdm import tqdm
root_filepath='./dacl10k_v2_devphase'
new_filepath='./dacl10k_v2_512_devphase'

def convert_seg_poly_ann(original_height,original_width,new_height,new_width,polygon):
    width_scale = new_width / original_width
    height_scale = new_height / original_height
    for i in range(len(polygon)):
        polygon[i][0]=polygon[i][0]*width_scale
        polygon[i][1]=polygon[i][1]*height_scale
    return polygon

def convert_dacl10k(split):
    if split in ['train','validation']:
        annotations_filepath = os.path.join(root_filepath,f'annotations/{split}')
        images_filepath = os.path.join(root_filepath,f'images/{split}')
        annotations_files = os.listdir(annotations_filepath)
        for annotation_file in tqdm(annotations_files):
            annotation = json.load(open(os.path.join(annotations_filepath,annotation_file),'r'))
            for i in range(len(annotation['shapes'])):
                annotation['shapes'][i]['points']=convert_seg_poly_ann(annotation['imageHeight'],annotation['imageWidth'],512,512,annotation['shapes'][i]['points'])
            annotation['imageHeight']=512
            annotation['imageWidth']=512
            img=Image.open(os.path.join(images_filepath,annotation['imageName']))
            img = img.resize((512,512))
            ### save annotations and images
            json.dump(annotation,open(os.path.join(new_filepath,f'annotations/{split}/{annotation_file}'),'w'))
            img.save(os.path.join(new_filepath,f"images/{split}/{annotation['imageName']}"))
    elif split == 'testdev':
        images_filepath = os.path.join(root_filepath,f'images/{split}')
        images_files = os.listdir(images_filepath)
        for images_file in tqdm(images_files):
            img=Image.open(os.path.join(images_filepath,images_file))
            img=img.resize((512,512))
            img.save(os.path.join(new_filepath,f"images/{split}/{images_file}"))
    else:
        return NotImplementedError

if __name__ == '__main__':
    convert_dacl10k('train')
    convert_dacl10k('validation')
    convert_dacl10k('testdev')