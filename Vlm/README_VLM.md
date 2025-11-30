# Open-Vocabulary Vision-Language Object Localization (Edge Device)

## Objective

A computer vision tool to detect and localize objects in images, running entirely on an edge device and guided by natural language prompts (e.g., “Pick the pen”).  
No cloud access, no anchor-based detectors (YOLO, Faster R-CNN, etc.), and no fixed class labels.

---

## Approach

- **Region Proposal:** [SAM (Segment Anything Model)] generates segmentation masks (object-agnostic).
- **Matching:** [CLIP](https://github.com/openai/CLIP) ranks each region against all prompt variants for open-vocabulary localization.
- **Prompt Parsing:** [spaCy](https://spacy.io/) (`en_core_web_sm`) extracts object phrases from natural language user prompts.
- **Output:** Bounding box drawn on the image for the region best matching the prompt.

---

## Features

- **Fully offline and edge-friendly:** All processing and inference is local.
- **Open-vocabulary:** No reliance on fixed class lists or retraining.
- **Natural language support:** Handles a wide range of user instructions (e.g., “Take the scissors”, “Pick the green object”).
- **Generalizes to unseen objects.**

---

## How to Use

1. **Install dependencies:**
   ```bash
   pip install torch opencv-python numpy pillow transformers spacy
   ```
2. **Download weights:**
   - https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth
     
     It should be put into 'Models' folder
	      
3. **Run:**
   ```bash
   py vlm_grounding.py
   ```
   Enter any natural language prompt. The output image (`output.jpg`) will show the localized object.

---

## How it Works

1. **SAM** proposes possible object regions.
2. **spaCy** extracts all noun phrases from the user's prompt.
3. **CLIP** evaluates every region with all prompt variants; the highest match wins.
4. The bounding box is drawn and saved.

---

## Evaluation

- **Correctness:** Finds objects matching any prompt.
- **Generalization:** Can localize objects never seen in training.
- **Efficiency:** Designed for edge devices—no internet required after setup.

---

## Example

Prompt:  
```
Pick the screwdriver
```
Output:  
Bounding box drawn around the screwdriver in the image.

---

## Notes

- All models and weights must be downloaded prior to deployment.
- No distribution or external API calls—completely self-contained.

---
Confidential & Property – do not distribute.