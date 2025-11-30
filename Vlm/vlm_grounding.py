import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import CLIPProcessor, CLIPModel
import spacy

# Load spaCy English model for noun phrase extraction
# Make sure en_core_web_sm is available, either via pip install wheel or symlink into your project
nlp = spacy.load("en_core_web_sm")  # or a local path if needed

# Load CLIP and SAM models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
sam = sam_model_registry["vit_b"](checkpoint="Models/sam_vit_b_01ec64.pth")

def extract_prompt_variants(prompt):
    """
    Generate CLIP-friendly prompt variants from natural language.
    Includes original prompt and all noun chunks extracted by spaCy.
    """
    doc = nlp(prompt)
    variants = [prompt.strip()]
    variants += [chunk.text.strip() for chunk in doc.noun_chunks]
    # Remove duplicates, preserve order
    seen = set()
    result = []
    for v in variants:
        if v not in seen and v:
            result.append(v)
            seen.add(v)
    return result

def get_image_regions(image, min_mask_area=500):
    """
    Generate bounding boxes for regions in the image using SAM AutomaticMaskGenerator.
    Only keeps boxes above min_mask_area.
    """
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    masks = mask_generator.generate(image)
    boxes = []
    for mask in masks:
        x0, y0, w, h = mask['bbox']
        if w * h < min_mask_area:
            continue
        x1 = x0 + w
        y1 = y0 + h
        boxes.append([x0, y0, x1, y1])
    return boxes

def rank_boxes_by_text(image, boxes, prompts):
    """
    For each box, scores all prompt variants with CLIP.
    Returns the box with the highest score.
    """
    h, w, _ = image.shape
    best_score = float('-inf')
    best_box = None
    for idx, box in enumerate(boxes):
        x0, y0, x1, y1 = map(int, box)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_pil = Image.fromarray(crop)
        for prompt in prompts:
            inputs = clip_processor(text=[prompt], images=crop_pil, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                score = outputs.logits_per_image[0][0].item()
            if score > best_score:
                best_score = score
                best_box = [x0, y0, x1, y1]
    if best_box is None:
        raise ValueError("No valid boxes found for scoring.")
    return best_box

def draw_bbox(image, bbox):
    x0, y0, x1, y1 = map(int, bbox)
    image_out = image.copy()
    cv2.rectangle(image_out, (x0, y0), (x1, y1), (0,255,0), 2)
    return image_out

def draw_all_boxes(image, boxes, color=(0,255,0)):
    image_out = image.copy()
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image_out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image_out, f'{i}', (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image_out

def localize_object(image_path, prompt, min_mask_area=500, debug_boxes=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    boxes = get_image_regions(image, min_mask_area=min_mask_area)
    if debug_boxes:
        image_with_boxes = draw_all_boxes(image, boxes)
        cv2.imwrite("all_boxes_debug.jpg", image_with_boxes)
        print("Saved all proposed boxes as all_boxes_debug.jpg")
    prompt_variants = extract_prompt_variants(prompt)
    print(f"Using prompt variants: {prompt_variants}")
    best_box = rank_boxes_by_text(image, boxes, prompt_variants)
    vis_image = draw_bbox(image, best_box)
    return vis_image, best_box

if __name__ == "__main__":
    print("Running Open-Vocabulary Vision-Language Localization Demo...")
    prompt = input("Enter your prompt (any natural language!): ")
    img_path = "input_img.jpg"  # Change as needed
    output_img, bbox = localize_object(img_path, prompt)
    cv2.imwrite("output.jpg", output_img)
    print("Output saved as output.jpg")