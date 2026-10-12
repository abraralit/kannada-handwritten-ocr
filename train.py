"""Train the Kannada word OCR model with CTC loss, mixed precision and warmup + cosine LR.

Usage (Colab, data on Google Drive - the defaults):
    python train.py

Usage (local):
    python train.py --data-dir data/preprocessed --out-dir models
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from labels import (KANNADA_UNICODE_CONSORTIUM, ConsortiumLabelConverter,
                    validate_vocabulary_with_consortium)
from model import EnhancedKannadaOCRModel

DRIVE_BASE = '/content/drive/MyDrive/kannada'


# ---------------------- Dataset with augmentation ----------------------
class EnhancedKannadaOCRDataset(Dataset):
    def __init__(self, images_npy, labels_npy, train_mode=False):
        self.images = images_npy
        self.labels = labels_npy
        self.train_mode = train_mode

        if train_mode:
            self.augment = transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05),
                                            scale=(0.9, 1.1), shear=3)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                ], p=0.3),
            ])
        else:
            self.augment = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Scale to [0, 1]
        img_arr = img.astype(np.float32)
        if img_arr.ndim == 2:
            img_arr = img_arr[..., None]
        if img_arr.max() > 1.0:
            img_arr = img_arr / 255.0

        img_tensor = torch.FloatTensor(img_arr).permute(2, 0, 1)

        # Normalise to [-1, 1]
        img_tensor = (img_tensor - 0.5) / 0.5

        if self.train_mode and self.augment:
            img_tensor = self.augment(img_tensor)

            # Random Gaussian noise
            if random.random() < 0.3:
                noise = torch.randn_like(img_tensor) * 0.02
                img_tensor = torch.clamp(img_tensor + noise, -1, 1)

            # Random brightness / contrast
            if random.random() < 0.5:
                img_tensor = TF.adjust_brightness(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = TF.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))

        label_clean = label[label != 0]  # index 0 is padding
        return img_tensor, label_clean


def collate_fn_dynamic(batch):
    """Stack images and pad label sequences to the longest label in the batch."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.LongTensor([len(l) for l in labels])
    max_len = max(label_lengths)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = torch.LongTensor(label)
    return images, padded_labels, label_lengths


# ---------------------- Training (mixed precision) ----------------------
def train_epoch_amp(model, device, loader, criterion, optimizer, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    batch_count = 0
    use_amp = device.type == 'cuda'

    for imgs, labels, label_lengths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        targets = []
        for i in range(labels.size(0)):
            targets.extend(labels[i, :label_lengths[i]].tolist())
        if len(targets) == 0:
            continue
        targets = torch.LongTensor(targets).to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(imgs)
            T, B, C = preds.size()
            input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
            loss = criterion(preds, targets, input_lengths, label_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠️  Invalid loss, skipping batch")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def validate_enhanced(model, device, loader, criterion, converter,
                      use_beam=False, beam_width=5, show_samples=False, num_samples=5):
    model.eval()
    total_loss = 0.0
    all_preds, all_gts = [], []
    sample_preds, sample_gts = [], []

    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            preds = model(imgs)
            T, B, C = preds.size()

            targets = []
            for i in range(labels.size(0)):
                targets.extend(labels[i, :label_lengths[i]].tolist())
            if len(targets) == 0:
                continue
            targets = torch.LongTensor(targets).to(device)

            input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
            loss = criterion(preds, targets, input_lengths, label_lengths)
            total_loss += loss.item()

            if use_beam:
                decoded = converter.beam_search_decode(preds.exp(), beam_width)
            else:
                decoded = converter.decode(preds.exp())
            all_preds.extend(decoded)

            for i in range(len(label_lengths)):
                gt_text = ''.join([converter.idx_to_char[int(idx)]
                                   for idx in labels[i, :label_lengths[i]]])
                all_gts.append(gt_text)

            if show_samples and len(sample_preds) < num_samples:
                for i in range(min(B, num_samples - len(sample_preds))):
                    sample_preds.append(decoded[i])
                    sample_gts.append(all_gts[len(all_gts) - B + i])

    # Word accuracy = exact match. Character accuracy = position-wise match
    # against the ground truth (not edit-distance based).
    correct_chars = total_chars = correct_words = 0
    for pred, gt in zip(all_preds, all_gts):
        if pred == gt:
            correct_words += 1
        for i, c in enumerate(gt):
            total_chars += 1
            if i < len(pred) and pred[i] == c:
                correct_chars += 1

    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    word_acc = correct_words / len(all_gts) if len(all_gts) > 0 else 0

    if show_samples:
        print("\n  📊 Sample Predictions:")
        for gt, pred in zip(sample_gts, sample_preds):
            mark = '✅' if gt == pred else '❌'
            print(f"    {mark} True: {gt}")
            print(f"       Pred: {pred}")

    return total_loss / max(len(loader), 1), char_acc, word_acc


# ---------------------- LR warmup + cosine annealing ----------------------
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def maybe_mount_drive(*paths):
    """Mount Google Drive when running on Colab with Drive paths."""
    if any(p.startswith('/content/drive') for p in paths):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            pass


# ---------------------- Main ----------------------
def main(args):
    maybe_mount_drive(args.data_dir, args.out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}\n")

    print("📂 Loading vocabulary...")
    with open(os.path.join(args.data_dir, 'vocabulary.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    print(f"📋 Kannada Unicode table: {len(KANNADA_UNICODE_CONSORTIUM)} characters")
    enhanced_vocab = validate_vocabulary_with_consortium(vocab)
    converter = ConsortiumLabelConverter(vocab)
    print(f"✅ Vocab size: {vocab['vocab_size']}")
    print(f"✅ Unicode check: {enhanced_vocab['valid_chars']} valid, "
          f"{enhanced_vocab['invalid_chars']} invalid\n")

    print("📂 Loading preprocessed data...")
    train_images = np.load(os.path.join(args.data_dir, 'train/images.npy'))
    train_labels = np.load(os.path.join(args.data_dir, 'train/labels.npy'))
    val_images = np.load(os.path.join(args.data_dir, 'val/images.npy'))
    val_labels = np.load(os.path.join(args.data_dir, 'val/labels.npy'))
    print(f"✅ Train: {len(train_images)} samples")
    print(f"✅ Val: {len(val_images)} samples\n")

    train_dataset = EnhancedKannadaOCRDataset(train_images, train_labels, train_mode=True)
    val_dataset = EnhancedKannadaOCRDataset(val_images, val_labels, train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_dynamic, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_dynamic, num_workers=args.workers)

    model_config = {'cnn_out': 512, 'rnn_hidden': 384, 'n_rnn_layers': 3, 'num_heads': 8}
    model = EnhancedKannadaOCRModel(num_classes=vocab['vocab_size'], dropout=0.3,
                                    **model_config).to(device)
    print("🏗️  Model: CNN-SE + BiLSTM + Multi-Head Attention")
    print(f"🔢  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CTCLoss(blank=vocab['vocab_size'], zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')  # no-op on CPU
    scheduler = WarmupScheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs,
                                base_lr=1e-5, max_lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'enhanced_kannada_ocr_best.pth')

    best_char_acc = 0.0
    patience_counter = 0

    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        current_lr = scheduler.step()
        train_loss = train_epoch_amp(model, device, train_loader, criterion, optimizer, scaler, None)

        val_loss, char_acc, word_acc = validate_enhanced(
            model, device, val_loader, criterion, converter,
            use_beam=epoch > 10, beam_width=5,           # beam search after epoch 10
            show_samples=epoch % 5 == 0, num_samples=5,
        )

        print(f"  LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Char Acc: {char_acc * 100:.2f}%")
        print(f"  Word Acc: {word_acc * 100:.2f}%")

        if char_acc > best_char_acc:
            best_char_acc = char_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char_acc': char_acc,
                'word_acc': word_acc,
                'model_config': model_config,
                'consortium_info': {
                    'total_consortium_chars': len(KANNADA_UNICODE_CONSORTIUM),
                    'valid_chars': enhanced_vocab['valid_chars'],
                },
            }, ckpt_path)
            print(f"  ✓ Model saved! (Best: {best_char_acc * 100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Character Accuracy: {best_char_acc * 100:.2f}%")
    print(f"Model saved at: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Train the Kannada handwritten word OCR model')
    p.add_argument('--data-dir', default=f'{DRIVE_BASE}/preprocessed',
                   help='folder with vocabulary.json and train/ val/ .npy files')
    p.add_argument('--out-dir', default=f'{DRIVE_BASE}/models', help='where to save the checkpoint')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3, help='peak learning rate')
    p.add_argument('--patience', type=int, default=20, help='early-stopping patience (epochs)')
    p.add_argument('--workers', type=int, default=2)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
