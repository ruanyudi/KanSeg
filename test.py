from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
import argparse
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import cv2
import numpy as np
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
OUTPUT_PATH='./test_output'
THRESHOLD=0.2
TARGET_LIST = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
               'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars',
               'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

def mask2ann(mask_img):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []

        for point in object:
            coords.append([int(point[0][0]),int(point[0][1])])

        polygons.append(coords)
    return polygons
        #[[131, 48, 130, 49, 129, 50, 128, 51, ...]]


def jsons2jsonl(jsons_dir, jsonl_path):
    """Combine multiple json files to a jsonl ('json lines') file."""
    with open(jsonl_path, 'w') as f:
        for json_path in sorted(Path(jsons_dir).glob("*.json")):
            with open(json_path, "r") as ff:
                f.write(ff.read() + "\n")



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    for file in tqdm(args.input):
        json_file = {
        "imageName": file.split('/')[-1],
        "imageWidth": 512,
        "imageHeight": 512,
        }
        img = read_image(file,format='BGR')
        prediction = predictor(img)['instances']
        anns = []
        keep = prediction.scores>THRESHOLD
        pred_classes=prediction.pred_classes[prediction.scores>0.2]
        pred_masks=prediction.pred_masks[prediction.scores>0.2]
        for i in range(pred_masks.shape[0]):
            ann = {"label":TARGET_LIST[int(pred_classes[i])]}
            pred_mask = pred_masks[i].to(dtype=torch.uint8).cpu().numpy()
            points=mask2ann(pred_mask)
            ann["shape_type"]="polygon"
            for point in points:
                ann['points']=point
                anns.append(ann)
        json_file['shapes']=anns
        # print(len(anns))
        json.dump(json_file,open(os.path.join(OUTPUT_PATH,f"{file.split('/')[-1].split('.')[-2]}.json"),'w'))
    jsons2jsonl('./test_output','submission.jsonl')