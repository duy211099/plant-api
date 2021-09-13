import os 
import io
import torch
from torch._C import device
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.transforms import Resize


# 1. Load model
model = models.resnet18(pretrained=True)

in_features = model.fc.in_features
out_features = 104

model.fc = nn.Linear(in_features, out_features)

best_model_path = 'ResNet18.pth'
checkpoint = torch.load(best_model_path)

model.load_state_dict(checkpoint)
model.eval()

# img to tensor

def transform_image(image_bytes):
    preprocess = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(io.BytesIO(image_bytes))

    return preprocess(image).unsqueeze(0)

def get_prediction(input_batch):
    with torch.no_grad():
        output = model(input_batch)

    output = model(input_batch)

    probabilities = torch.nn.functional.softmax(
        output[0], dim=0).cpu().data.numpy()

    prediction = int(torch.max(output.data, 1)[1].numpy())
    best_accuracy = probabilities[prediction]

    result = [best_accuracy, prediction]

    return result