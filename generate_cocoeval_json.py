import os
import logging
import random
import time
import json
import statistics
from unittest.mock import patch

from PIL import Image
import glob
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers.dynamic_module_utils import get_imports

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data import DocVQADataset
from metrics import average_normalized_levenshtein_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fixed_get_imports(filename):
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# create model
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    config = AutoConfig.from_pretrained("model_checkpoints/epoch_3/", trust_remote_code=True)
    config.attn_implementation = "eager"  # Disable flash attention
    model = AutoModelForCausalLM.from_pretrained(
        "model_checkpoints/epoch_3/", config=config, attn_implementation="eager", trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "model_checkpoints/epoch_3/", config=config, trust_remote_code=True
    )

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    start_processor = time.time()
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    end_processor = time.time()
    print(f"Time taken for processor: {end_processor - start_processor:.2f}s")
    start = time.time()
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        return_dict_in_generate=True,
        output_scores=True,
    )
    transition_scores = model.compute_transition_scores(
        generated_ids.sequences, generated_ids.scores, beam_indices=generated_ids.beam_indices, normalize_logits=True
    )
    input_length = inputs.input_ids.shape[1]
    # print("transition scores are ", transition_scores)
    generated_tokens = generated_ids.sequences#[:, input_length:]
    scores_per_object = []
    curr_scores = []
    items_added_for_curr_box = 0
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        decoded_token = processor.decode(tok)
        if decoded_token == "Object":
            if len(curr_scores) > 0:
                scores_per_object.append(statistics.mean(curr_scores))
                curr_scores = []
                items_added_for_curr_box = 0
            continue
        score_prob = np.exp(score.cpu().numpy())
        # Check if decoded token is an integer
        #if decoded_token.isdigit() or "loc_" in decoded_token and items_added_for_curr_box < 5:
        if decoded_token.isdigit() and items_added_for_curr_box < 1:
            curr_scores.append(score_prob)
            items_added_for_curr_box += 1
        #print(f"| {tok:5d} | {processor.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
    if len(curr_scores) > 0:
        scores_per_object.append(statistics.mean(curr_scores))
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]
    # print("generated text is ", generated_text)
    # print("full generated text is ", generated_text)
    # print("scores per object are ", scores_per_object)
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer, scores_per_object


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers


# Create DataLoader
batch_size = 4  # Adjust the batch size based on your GPU memory
num_workers = 0  # Number of worker processes to use for data loading
prefetch_factor = None  # Number of batches to prefetch

# test_dataset = DocVQADataset("validation")

# # Create a subset of the dataset
# subset_size = int(0.2 * len(test_dataset))  # 10% of the dataset
# indices = random.sample(range(len(test_dataset)), subset_size)
# subset_dataset = Subset(test_dataset, indices)

# test_loader = DataLoader(
#     subset_dataset,
#     batch_size=batch_size,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     prefetch_factor=prefetch_factor,
# )


def run_batch(inputs):
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts


def evaluate_model(test_loader):
    task_prompt = "<DocVQA>"
    predicted_answers = []
    ground_truth = []

    for inputs, batch_answers in tqdm(test_loader, desc="Evaluating"):
        generated_texts = run_batch(inputs)

        for generated_text, answers in zip(generated_texts, batch_answers):
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(
                    inputs["pixel_values"].shape[-2],
                    inputs["pixel_values"].shape[-1],
                ),
            )
            predicted_answers.append(parsed_answer[task_prompt].replace("<pad>", ""))
            ground_truth.append(answers)
            # print("Ans:", parsed_answer[task_prompt])
            # print("GT:", answers)

    avg_levenshtein_similarity = average_normalized_levenshtein_similarity(
        ground_truth, predicted_answers
    )
    return answers, avg_levenshtein_similarity

def vis_model_prediction(image, bbox_output, scores_per_object, output_path):
    # Get the shape of the image
    orig_w, orig_h = image.size
    image_area = orig_w * orig_h
    image_np = np.array(image)
    # height, width, _ = image_np.shape
    # bbox = [box_idx[0] / 1000 * width, box_idx[1] / 1000 * height, box_idx[2] / 1000 * width, box_idx[3] / 1000 * height]
    # print(bbox)

    # Parse the bbox output
    bboxes = bbox_output["bboxes"]
    categories = bbox_output["labels"]

    # Uses matplotlib to display the image and overlay the x1, y1, x2, y2 bounding box over it

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    for bbox, category, score in zip(bboxes, categories, scores_per_object):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area > 0.5 * image_area:
            continue
        # Eliminate cases where bbox contains corner of the image, as it is likely a false positive
        if (x1 < 2 and y1 < 2) or (x2 > orig_w - 2 and y1 < 2) or (x1 < 2 and y2 > orig_h - 2) or (x2 > orig_w - 2 and y2 > orig_h - 2):
            continue
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], category, fontsize=14, color='blue')
        ax.text(bbox[0], bbox[3], f"score: {score:.2f}", fontsize=14, color='magenta')
    fig.savefig(output_path)

# Run the evaluation
#answers, average_similarity = evaluate_model(test_loader)
#print(f"Average Normalized Levenshtein Similarity: {average_similarity:.4f}")

BENCHMARK_PATH = "/home/scratch.driveix_50t_3/aguru/tless_test_primesense_bop19/tless/test_primesense"
folders = sorted(glob.glob(os.path.join(BENCHMARK_PATH, "*")))

# Load in the targets json now
test_targets_json = "test_targets_bop19.json"
test_targets = dict()
predictions = list()
with open(test_targets_json, "r") as file:
    targets = json.load(file)
    for target in targets:
        im_id = target["im_id"]
        scene_id = target["scene_id"]
        if scene_id not in test_targets.keys():
            test_targets[scene_id] = [im_id]
        else:
            if test_targets[scene_id][-1] != im_id:
                test_targets[scene_id].append(im_id)

for folder_idx, folder in enumerate(folders):
    folder_name = os.path.basename(folder)
    rgb_folder = os.path.join(folder, "rgb")
    # img_paths = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
    img_paths = sorted([
        os.path.join(rgb_folder, str(number).zfill(6)+".png")
        for number in test_targets[int(os.path.basename(folder_name))]
    ])

    num_imgs = len(img_paths)
    
    scene_gt_file = open(os.path.join(folder, "scene_gt.json"))
    scene_gt_info_file = open(os.path.join(folder, "scene_gt_info.json"))
    scene_gt_dict = json.load(scene_gt_file)
    scene_gt_info_dict = json.load(scene_gt_info_file)

    print(f"starting inference for {folder_name}, containing {num_imgs} images")
    missed_detections = 0

    # def per_batch_ap(img_pred_labels, img_pred_probs, img_pred_boxes, gt_boxes, gt_labels, iou_threshold):
    # for img_path in img_paths:
    for img_id in range(0, len(img_paths)):
        # Read the image
        image = Image.open(img_paths[img_id])
        image_name = os.path.basename(img_paths[img_id]).split(".")[0]

        # Get the shape of the image
        orig_w, orig_h = image.size
        image_area = orig_w * orig_h

        # Run the model on the image
        start_time = time.time()
        output, scores_per_object = run_example(task_prompt = "<OD>", text_input = "", image = image)
        end_time = time.time()
        print(output)
        # Find all instances of "loc_" in the output, and get the integer that follows each time
        bbox_output = output["<OD>"]
        print("scores per object are ", scores_per_object)

        print("len boxes is ", len(bbox_output["bboxes"]))
        print("len labels is ", len(bbox_output["labels"]))

        # We now dump the visualization of the model output
        vis_model_prediction(image, bbox_output, scores_per_object, "/home/scratch.driveix_50t_3/aguru/vis_preds_florence_filtered/{}_{}.png".format(folder_name, image_name))

        # Parse the bbox output
        for box_id in range(len(bbox_output["bboxes"])):

            pred_box = bbox_output["bboxes"][box_id]
            x1, y1, x2, y2 = pred_box

            curr_object_score = float(scores_per_object[box_id]) if box_id < len(scores_per_object) else float(0.0)
            if curr_object_score < 0.1:
                continue

            try:
                start = (int(pred_box[0]), int(pred_box[1]))
                end   = (int(pred_box[2]), int(pred_box[3]))
            except:
                import ipdb; ipdb.set_trace()

            #label_id = torch.argmax(probs[box_id]).item()
            #prob = probs[box_id][label_id].item()
            #class_id = int(bbox_output["labels"][box_id].split("_")[1])
            # Parse the int out of the string bbox_output["labels"][box_id]
            found_ints = re.findall(r'\d+', bbox_output["labels"][box_id])
            if len(found_ints) == 0:
                continue
            class_id = int(found_ints[0])

            #area = float(pred_box[2] - pred_box[0]) * float(pred_box[3] - pred_box[1])
            area = (x2 - x1) * (y2 - y1)
            # If image is more than half the area, ignore- probably a wrong box
            if area > 0.5 * image_area:
                continue
            # Eliminate cases where bbox contains corner of the image, as it is likely a false positive
            if (x1 == 0 and y1 == 0) or (x2 == orig_w and y1 == 0) or (x1 == 0 and y2 == orig_h) or (x2 == orig_w and y2 == orig_h):
                continue

            found = True
            detection = dict()
            detection["scene_id"] = int(folder_name)
            detection["image_id"] = int(image_name)
            detection["category_id"] = class_id
            # Confidence score is always 1 with florence-2
            detection["score"] = curr_object_score
            detection["bbox"] = [
                float(x1),# * 1/resize_scale.x * orig_w/640, # x1
                float(y1),# * 1/resize_scale.y * orig_h/480, # y1
                float(x2 - x1),#float(pred_box[2] - pred_box[0]) * 1/resize_scale.x * orig_w/640, # w
                float(y2 - y1)#float(pred_box[3] - pred_box[1]) * 1/resize_scale.y * orig_h/480, # h
            ]
            detection["time"] = float(end_time - start_time)
            predictions.append(detection)
        if not found:
            missed_detections += 1



    scene_gt_file.close()
    scene_gt_info_file.close()
    print("Missed detections: ", missed_detections)

#import ipdb; ipdb.set_trace()


fileName = 'florence2ep3wclassscoresthresh01_tless-test.json'

with open(fileName, 'w') as f:
    json.dump(predictions, f)



# Load in image as PIL
image = Image.open("test_images/0010.jpg")
output = run_example(task_prompt = "<OD>", text_input = "", image = image)
print(output)
# Find all instances of "loc_" in the output, and get the integer that follows each time
bbox_output = output["<OD>"]
# pattern = r"loc_(\d+)"
# matches = re.findall(pattern, str_output)
# print(matches)
# box_idx = [int(x) for x in matches]
image_np = np.array(image)
# height, width, _ = image_np.shape
# bbox = [box_idx[0] / 1000 * width, box_idx[1] / 1000 * height, box_idx[2] / 1000 * width, box_idx[3] / 1000 * height]
# print(bbox)

# Parse the bbox output
bboxes = bbox_output["bboxes"]
categories = bbox_output["labels"]

# Uses matplotlib to display the image and overlay the x1, y1, x2, y2 bounding box over it
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(image_np)
for bbox, category in zip(bboxes, categories):
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], category, fontsize=14)
fig.savefig("test_images/0010detect_2.png")