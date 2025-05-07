import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Define augmentations
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.Resize(224, 224),
    ToTensorV2()
])

def augment_image(image_np):
    augmented = augment(image=image_np)
    return augmented['image']
