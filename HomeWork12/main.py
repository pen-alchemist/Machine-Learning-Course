import os
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def dataset_load(dir_path, csv_path):
    """Read csv file with pandas and extract data"""
    abs_path = f'{os.getcwd()}/{dir_path}'
    df = pd.read_csv(f'{abs_path}/{csv_path}')
    images = []
    labels = [cls for cls in df['class']]
    img_column = df['image']

    for _img_path in img_column:
        image = preprocess(Image.open(f'{abs_path}/{_img_path}'))
        images.append(image)

    return images, labels


train_images, train_labels = dataset_load('train_dataset', 'train_ds.csv')
test_images, test_labels = dataset_load('test_dataset', 'test_ds.csv')


for test_image in test_images:
    for train_image in train_images:
        test_image /= test_image.norm(dim=-1, keepdim=True)
        train_image /= train_image.norm(dim=-1, keepdim=True)
        similarity = train_image.cpu().numpy() @ test_image.cpu().numpy()
        print(similarity)


compare_list = ['a photo of a labrador', 
                'a photo of a cat', 
                'a photo of yellow cat',
                'a photo of black cat',
                'a photo of white cat',
                'a photo of black and white cat',
                'a photo of airplane',
                'a photo of buggati']

image = preprocess(Image.open(f'{os.getcwd()}/cat.jpg')).unsqueeze(0).to(device)
text = clip.tokenize(compare_list).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
np.argmax(probs)
