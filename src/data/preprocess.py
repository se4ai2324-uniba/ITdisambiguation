""" Module used to preprocess the data """

# !/usr/bin/python
import sys
import os
import torch
import open_clip
from PIL import ImageFile, Image
from tqdm import tqdm

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

Image.MAX_IMAGE_PIXELS = 122080000
ImageFile.LOAD_TRUNCATED_IMAGES = True

for i in tqdm(range(len(imgs))):
    img = imgs[i]
    preproc_image = preproc(Image.open(os.path.join(input_folder, img)))
    torch.save(preproc_image, os.path.join(output_folder, img))

print("[+] Finished [+]")
