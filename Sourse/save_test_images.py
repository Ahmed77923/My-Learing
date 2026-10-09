
from pathlib import Path
from torchvision.datasets import CIFAR10

# Download or load the official CIFAR-10 test set
dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
)

output_dir = Path("test_images")
output_dir.mkdir(exist_ok=True)

class_names = dataset.classes
saved_classes = set()

for image, label in dataset:
    class_name = class_names[label]

    # Save one image per class
    if class_name not in saved_classes:
        image.save(output_dir / f"{class_name}.png")
        saved_classes.add(class_name)
        print(f"Saved: {class_name}.png")

    if len(saved_classes) == len(class_names):
        break

print(f"\nDone! Saved {len(saved_classes)} images in {output_dir}/")
