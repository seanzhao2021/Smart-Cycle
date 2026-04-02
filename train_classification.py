import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_transforms(img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tfms, val_tfms


class WasteClassificationDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "mobilenet":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnetv2_s":
        model = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif model_name == "efficientnetv2_m":
        model = timm.create_model(
            "tf_efficientnetv2_m.in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    model_name = model_name.lower()

    if model_name == "mobilenet":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_name in {"resnet50", "resnet101"}:
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name in {"efficientnetv2_s", "efficientnetv2_m"}:
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Could not find classifier/head to unfreeze for efficientnet model.")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def collect_samples(data_dir: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    samples = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] Missing class folder: {class_dir}")
            continue

        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((file_path, class_idx))

    return samples


def stratified_split(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    return train_samples, val_samples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": val_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def save_history(history: list, output_dir: Path) -> None:
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "history.csv", index=False)


def save_metrics(metrics: Dict, class_names: list, output_dir: Path) -> None:
    payload = {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision"],
        "recall_macro": metrics["recall"],
        "f1_macro": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": class_names,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to raw detection-style dataset root containing class folders."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["mobilenet", "resnet50", "resnet101", "efficientnetv2_s", "efficientnetv2_m"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    class_names = [
        "biological",
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms, val_tfms = get_transforms(args.img_size)

    data_dir = Path(args.data_dir)
    all_samples = collect_samples(data_dir, class_names)

    if not all_samples:
        raise RuntimeError(f"No images found under {data_dir}")

    train_samples, val_samples = stratified_split(
        samples=all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = WasteClassificationDataset(train_samples, transform=train_tfms)
    val_dataset = WasteClassificationDataset(val_samples, transform=val_tfms)

    print(f"Classes: {class_names}")
    print(f"Total images: {len(all_samples)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = get_model(args.model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    run_name = args.save_name if args.save_name else args.model_name
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_backbone(model, args.model_name)
    optimizer = get_optimizer(model, args.lr, args.weight_decay)

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            print(f"Unfreezing all layers at epoch {epoch + 1}")
            unfreeze_all(model)
            optimizer = get_optimizer(model, args.lr * 0.1, args.weight_decay)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision"],
            "val_recall_macro": val_metrics["recall"],
            "val_f1_macro": val_metrics["f1"],
        }
        history.append(epoch_result)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pt")
            save_metrics(val_metrics, class_names, output_dir)

    save_history(history, output_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    save_metrics(final_metrics, class_names, output_dir)

    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print("\nTraining complete.")
    print(f"Best macro F1: {final_metrics['f1']:.4f}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()