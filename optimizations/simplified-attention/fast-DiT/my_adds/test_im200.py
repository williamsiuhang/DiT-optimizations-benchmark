from im200 import ImageFolder200

dataset = ImageFolder200("~/scratch/imagenet")
print("loaded")
print(f"{dataset.classes}")
