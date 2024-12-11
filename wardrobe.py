from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import numpy as np
import torch.nn as nn
import threading

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b3-fashion")
processor_nlp = ViltProcessor.from_pretrained("yanka9/vilt_finetuned_deepfashionVQA_v2")
model_nlp = ViltForQuestionAnswering.from_pretrained("yanka9/vilt_finetuned_deepfashionVQA_v2")

categories = model.config.id2label
del categories[0]
categories[1] = "shirt or blouse"
categories[2] = "top, t-shirt or sweatshirt"

other_cat = ["color", "style"]

def create_columns():
    columns = []
    for idx in range(1, 9):
        if idx in [52,53]:
            continue
        it_col = []
        it_col.append(f"item {idx} type")
        for i in other_cat:
            it_col.append(f"item {idx} {i}")
        columns.extend(it_col)
    return columns
    

fash_col = create_columns()

with open("dataset.csv", "a") as file:
    file.write(",".join(fash_col)+"\n")


def get_metadata(path, item):
    image = Image.open(path)
    answer = []
    questions = [f"what is the color of the {item}?", f"is this {item} casual or formal"]

    for text in questions:

    # prepare inputs
        encoding = processor_nlp(image, text, return_tensors="pt")

        # forward pass
        outputs = model_nlp(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer.append(model_nlp.config.id2label[idx])
    return tuple(answer)

def ask_question(path, items):
    image = Image.open(path)
    text = f"{items}?"

    # prepare inputs
    encoding = processor_nlp(image, text, return_tensors="pt")

    # forward pass
    outputs = model_nlp(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model_nlp.config.id2label[idx]
    

takeouts = ["glasses", "collar", "sleeve", "buckle", "belt", ]

def get_categories(path):
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    unsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )
    pred_seg = unsampled_logits.argmax(dim=1)[0]
    unique_cat = np.unique(pred_seg.cpu().numpy())
    result = []
    for group_no in unique_cat:
        if group_no in [1,2]:
            group = categories[group_no]
            result.append(ask_question(path, group))
        elif group_no == 0: continue
        else:
            result.append(categories[group_no])
    for i in takeouts:
        if i in result:
            result.remove(i)
    return result

        
def write_to_csv(row, csv_file="dataset.csv"):
    with open(csv_file, "a") as file:
        file.write(row + "\n")


def handle_image(path):
    labels = get_categories(path)
    result = []
    for label in labels:
        result.append(label)
        metadata = get_metadata(path, label)
        result.extend(metadata)
    t1 = threading.Thread(target=write_to_csv, args=(",".join(result),))
    t1.start()


images = ["1.jpg"]

def main():
    print("running main...")
    for image_path in images:
        p1 = threading.Thread(target=handle_image, args=(f"./data/{image_path}",))
        p1.start()


if __name__ == "__main__":
    main()