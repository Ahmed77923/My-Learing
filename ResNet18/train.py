
from pathlib import Path
import copy
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
IMAGE_SIZE = 224

DATA_DIR = Path("./data")
ARTIFACTS_DIR = Path("./artifacts")
OUTPUTS_DIR = Path("./outputs")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = ARTIFACTS_DIR / "best_resnet18.pth"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# 2. Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Image transformations
# ============================================================

# ImageNet normalization expected by pretrained ResNet18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(IMAGE_SIZE, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 4. Load CIFAR-10 with separate train and validation transforms
# ============================================================

train_full = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=train_transform,
)

val_full = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=eval_transform,
)

test_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=eval_transform,
)

# Create the split once, then reuse its indices.
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    len(train_full), generator=generator
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

# Keep num_workers=0 for straightforward debugging.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")


# ============================================================
# 5. Load pretrained ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all pretrained parameters.
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# Only the new classifier will be trained.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 6. Training and evaluation functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    # Keep the frozen backbone in evaluation mode.
    # This also prevents BatchNorm running statistics from changing.
    model.eval()
    model.fc.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
    )


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# ============================================================
# 7. MLflow setup
# ============================================================

mlflow.set_experiment("CIFAR10-ResNet18-Transfer-Learning")


# ============================================================
# 8. Train with early stopping and checkpointing
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_model_state = None

with mlflow.start_run(run_name="resnet18-feature-extraction"):

    mlflow.log_params({
        "architecture": "ResNet18",
        "pretrained": True,
        "training_strategy": "feature_extraction",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "test_size": len(test_dataset),
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "augmentation": "RandomHorizontalFlip, RandomCrop",
    })

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_results = evaluate(
            model, val_loader, criterion
        )
        val_loss = val_results["loss"]
        val_acc = val_results["accuracy"]

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
        }, step=epoch)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save({
                "model_state_dict": best_model_state,
                "class_names": CLASS_NAMES,
                "best_epoch": best_epoch,
                "validation_loss": best_val_loss,
                "image_size": IMAGE_SIZE,
            }, CHECKPOINT_PATH)

            print("  Best checkpoint saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_model_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    # Restore the best model before final evaluation.
    model.load_state_dict(best_model_state)

    mlflow.log_metric("best_epoch", best_epoch)
    mlflow.log_metric("best_validation_loss", best_val_loss)

    # Log the trained model.
    mlflow.pytorch.log_model(
        model,
        name="model",
    )

    # Save checkpoint as an MLflow artifact too.
    mlflow.log_artifact(str(CHECKPOINT_PATH))


# ============================================================
# 9. Final validation and test evaluation
# ============================================================

val_results = evaluate(model, val_loader, criterion)
test_results = evaluate(model, test_loader, criterion)

print("\nFinal results using the best checkpoint")
print(f"Best epoch:          {best_epoch}")
print(f"Validation accuracy: {val_results['accuracy']:.2f}%")
print(f"Test accuracy:       {test_results['accuracy']:.2f}%")

# Start a separate run for final evaluation results.
with mlflow.start_run(run_name="resnet18-final-evaluation"):
    mlflow.log_metrics({
        "best_epoch": best_epoch,
        "validation_accuracy": val_results["accuracy"],
        "validation_loss": val_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "test_loss": test_results["loss"],
    })


# ============================================================
# 10. Classification report
# ============================================================

print("\nClassification Report (Test Set):")

print(classification_report(
    test_results["labels"],
    test_results["predictions"],
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))


# ============================================================
# 11. Confusion matrix
# ============================================================

cm = confusion_matrix(
    test_results["labels"],
    test_results["predictions"],
    labels=list(range(NUM_CLASSES)),
)

fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)

ax.set_title("ResNet18 Transfer Learning - CIFAR-10")
fig.tight_layout()
fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close(fig)

with mlflow.start_run(run_name="resnet18-evaluation-artifacts"):
    mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))

print(f"\nCheckpoint: {CHECKPOINT_PATH}")
print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
print("Training and evaluation complete.")
