import logging
import random
import time

from PIL import Image
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from data import DocVQADataset
from metrics import average_normalized_levenshtein_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
# model = AutoModelForCausalLM.from_pretrained(
#     "model_checkpoints/gigantic_fukuiraptor/epoch_9/", trust_remote_code=True
# ).to(device)
# processor = AutoProcessor.from_pretrained(
#     "model_checkpoints/gigantic_fukuiraptor/epoch_9/", trust_remote_code=True
# )

# Load the model and processor
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
# ).to(device)
# processor = AutoProcessor.from_pretrained(
#     "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
# )

model = AutoModelForCausalLM.from_pretrained(
    "model_checkpoints/epoch_2/", trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "model_checkpoints/epoch_2/", trust_remote_code=True
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
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer


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


# Run the evaluation
#answers, average_similarity = evaluate_model(test_loader)
#print(f"Average Normalized Levenshtein Similarity: {average_similarity:.4f}")
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