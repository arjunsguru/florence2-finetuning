import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import cv2
import numpy
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import List

class PBRDataset(Dataset):
    """
    T-LESS dataset loader.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset (e.g. 'train', 'val', 'test').
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_files = glob.glob(os.path.join(root_dir, "*", "rgb", "*.jpg"))
        self.obj_annotations_files = glob.glob(os.path.join(root_dir, "*", "scene_gt.json"))
        self.obj_info_annotations_files = glob.glob(os.path.join(root_dir, "*", "scene_gt_info.json"))

        self.obj_number_to_ann_dict = {}
        for annotation_file in self.obj_annotations_files:
            obj_number = annotation_file.split("/")[-2]
            ann_dict = self.process_annotation_file(annotation_file)
            self.obj_number_to_ann_dict[obj_number] = ann_dict
        self.obj_number_to_ann_info_dict = {}
        for annotation_file in self.obj_info_annotations_files:
            obj_number = annotation_file.split("/")[-2]
            ann_dict = self.process_annotation_file(annotation_file)
            self.obj_number_to_ann_info_dict[obj_number] = ann_dict
        
    
    def process_annotation_file(self, ann_file):
        # Reads in the annotaion JSON file and returns the object annotations
        with open(ann_file, "r") as f:
            # Read the JSON file
            ann_data = json.load(f)
        return ann_data

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        image_path = self.images_files[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        obj_number = image_path.split("/")[-3]
        frame_number = str(int(image_path.split("/")[-1].split(".")[0]))
        frame_ann_list = self.obj_number_to_ann_dict[obj_number][frame_number]
        frame_ann_info_list = self.obj_number_to_ann_info_dict[obj_number][frame_number]

        all_object_strings = []
        all_object_keys = []
        for frame_ann, frame_ann_info in zip(frame_ann_list, frame_ann_info_list):
            bbox = frame_ann_info["bbox_obj"]
            bbox_area = bbox[2] * bbox[3]
            obj_id = frame_ann["obj_id"]
            if frame_ann_info["px_count_valid"] == 0:
                continue
            prompt = "<OD>"
            quantized_box_str = self.quantize_box(bbox, obj_id, image.shape[1], image.shape[0], bins=1000)
            if "<loc_0><loc_0><loc_0><loc_0>" in quantized_box_str:
                continue
            all_object_keys.append(bbox_area)
            all_object_strings.append(quantized_box_str)
        all_object_strings = [x for _, x in sorted(zip(all_object_keys, all_object_strings), key=lambda pair: pair[0])]
        answer = "".join(all_object_strings)

        return prompt, answer, image
    
    def quantize_box(self, box: List[int], object_id: int, size_w, size_h, bins=1000):
        bins_w, bins_h = bins, bins  # Quantization bins.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, w, h = box[0], box[1], box[2], box[3]
        xmax, ymax = xmin + w, ymin + h

        quantized_xmin = int(min(max((xmin / size_per_bin_w) // 1, 0), bins_w - 1))
        quantized_ymin = int(min(max((ymin / size_per_bin_h) // 1, 0), bins_h - 1))
        quantized_xmax = int(min(max((xmax / size_per_bin_w) // 1, 0), bins_w - 1))
        quantized_ymax = int(min(max((ymax / size_per_bin_h) // 1, 0), bins_h - 1))

        answer_str = f"Object_{object_id}<loc_{quantized_xmin}><loc_{quantized_ymin}><loc_{quantized_xmax}><loc_{quantized_ymax}>"

        return answer_str

def get_tless_dataloader(root_dir, split, batch_size, transform=None):
    """
    Get a PyTorch DataLoader for the T-LESS dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Split of the dataset (e.g. 'train', 'val', 'test').
        batch_size (int): Batch size of the DataLoader.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        DataLoader: PyTorch DataLoader for the T-LESS dataset.
    """
    dataset = TLESSDataset(root_dir, split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def vis_label(image, answer):
    # Visualize the label on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    fig, ax = plt.subplots()
    ax.imshow(image)
    for obj_str in answer.split("Object_")[1:]:
        print(obj_str.split("<loc_"))
        xmin, ymin, xmax, ymax = [int(x[:-1]) for x in obj_str.split("<loc_")[1:]]
        print(xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(image, str(obj_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="g", facecolor="none"))
    fig.savefig("test_images/test.png")
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset = PBRDataset("/home/scratch.driveix_50t_3/aguru/train_pbr")
    print("length of dataset is {}".format(len(dataset)))
    prompt, answer, image = dataset[0]
    print("prompt is {} with type {}".format(prompt, type(prompt)))
    print("answer is {} with type {}".format(answer, type(answer)))
    print("image shape is {}".format(image.shape))
    vis_label(image, answer)

    # dataset = TLESSDataset("/home/scratch.driveix_50t_3/aguru/t-less_v2", "test")
    # print("length of dataset is {}".format(len(dataset)))
    # prompt, answer, image = dataset[0]
    # print("prompt is {} with type {}".format(prompt, type(prompt)))
    # print("answer is {} with type {}".format(answer, type(answer)))
    # print("image shape is {}".format(image.shape))