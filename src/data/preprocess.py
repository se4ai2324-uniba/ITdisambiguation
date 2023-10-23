#!/usr/bin/python
import torch
import open_clip
import sys
import os
from PIL import Image

args = sys.argv
if len(args) < 4:
    sys.exit("Usage: preprocess.py INPUT_FOLDER OUTPUT_FOLDER IS_TRAIN")

input_folder = sys.argv[1]
output_folder = sys.argv[2]
is_train = sys.argv[3].lower() == 'true'

if not os.path.isdir(input_folder):
    sys.exit("Error: input folder does not exist.")
imgs = os.listdir(input_folder)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

print("[+] Starting image preprocessing... [+]")
preproc = open_clip.image_transform(224, is_train)
for i, img in enumerate(imgs):
    if i % 100 == 0:
        print(f"{i}/{len(imgs)}")
    preproc_image = preproc(Image.open(os.path.join(input_folder, img)))
    torch.save(preproc_image, os.path.join(output_folder, img))
print("[+] Finished [+]")
