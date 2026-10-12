"""Run the trained Kannada OCR model on a single word image.

Usage:
    python predict.py path/to/word.jpg
    python predict.py path/to/word.jpg --model models/enhanced_kannada_ocr_best.pth \
                                       --vocab data/preprocessed/vocabulary.json
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from labels import ConsortiumLabelConverter
from model import EnhancedKannadaOCRModel

DRIVE_BASE = '/content/drive/MyDrive/kannada'


def load_model(model_path, vocab_dict, device):
    """Load the trained model from a checkpoint (only load checkpoints you trust)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('model_config',
                            {'cnn_out': 512, 'rnn_hidden': 384, 'n_rnn_layers': 3, 'num_heads': 8})
    print("  ✅ Model configuration:")
    print(f"     CNN Out: {config.get('cnn_out', 512)}")
    print(f"     RNN Hidden: {config.get('rnn_hidden', 384)}")
    print(f"     RNN Layers: {config.get('n_rnn_layers', 3)}")
    print(f"     Attention Heads: {config.get('num_heads', 8)}")

    model = EnhancedKannadaOCRModel(
        num_classes=vocab_dict['vocab_size'],
        cnn_out=config.get('cnn_out', 512),
        rnn_hidden=config.get('rnn_hidden', 384),
        n_rnn_layers=config.get('n_rnn_layers', 3),
        num_heads=config.get('num_heads', 8),
        dropout=0.3,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    print("  ✅ Model loaded")
    if 'epoch' in checkpoint:
        print(f"     Best epoch: {checkpoint['epoch']}")
    if 'char_acc' in checkpoint:
        print(f"     Val character accuracy: {checkpoint['char_acc'] * 100:.2f}%")
    if 'word_acc' in checkpoint:
        print(f"     Val word accuracy: {checkpoint['word_acc'] * 100:.2f}%")
    return model


def preprocess_image(image_path, target_height=64, target_width=256):
    """Grayscale, aspect-preserving resize, sharpen, pad onto a white 64x256 canvas."""
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')

    orig_width, orig_height = image.size
    print(f"  📏 Original size: {orig_width}x{orig_height}")

    aspect_ratio = orig_width / orig_height
    if aspect_ratio > target_width / target_height:
        new_width = target_width                       # wider image: fit to width
        new_height = max(1, int(target_width / aspect_ratio))
    else:
        new_height = target_height                     # taller image: fit to height
        new_width = max(1, int(target_height * aspect_ratio))

    image_resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    image_resized = image_resized.filter(ImageFilter.SHARPEN)
    image_resized = ImageEnhance.Contrast(image_resized).enhance(1.5)

    canvas = Image.new('L', (target_width, target_height), color=255)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas.paste(image_resized, (x_offset, y_offset))
    print(f"  📏 Resized to: {new_width}x{new_height} (padded to {target_width}x{target_height})")

    img_array = np.array(canvas, dtype=np.float32)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0

    # Optional: hard binarisation for very clean text
    # img_array = np.where(img_array > 0.5, 1.0, 0.0)

    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = (img_tensor - 0.5) / 0.5                                # same as training
    return canvas, img_tensor


def predict_text(model, image_tensor, converter, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)              # (T, B, C) log-probabilities
        probabilities = torch.exp(output)
        predicted = converter.decode(probabilities)
    return predicted[0] if predicted else ""


def visualize_prediction(image, predicted_text):
    """Show the preprocessed image next to the prediction.
    Kannada glyphs need a Kannada font installed for matplotlib (e.g. Noto Sans Kannada)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Preprocessed Image', fontsize=12)
    axes[1].text(0.5, 0.5, predicted_text, fontsize=24, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].axis('off')
    axes[1].set_title('Predicted Text', fontsize=12)
    plt.tight_layout()
    plt.show()


def test_kannada_ocr(model_path, vocab_path, image_path, show_plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}\n")

    print("📂 Loading vocabulary...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    print(f"  ✅ Vocabulary size: {vocab_dict['vocab_size']}\n")
    converter = ConsortiumLabelConverter(vocab_dict, verbose=False)

    print("🏗️  Loading model...")
    model = load_model(model_path, vocab_dict, device)
    print()

    print(f"🖼️  Processing image: {image_path}")
    image, image_tensor = preprocess_image(image_path)
    print(f"  ✅ Tensor shape: {tuple(image_tensor.shape)}\n")

    predicted_text = predict_text(model, image_tensor, converter, device)
    print("=" * 60)
    print(f"Predicted Text: {predicted_text}")
    print("=" * 60)

    if show_plot:
        visualize_prediction(image, predicted_text)
    return predicted_text


def parse_args():
    p = argparse.ArgumentParser(description='Recognise a handwritten Kannada word image')
    p.add_argument('image', nargs='?', help='path to the word image (asked interactively if omitted)')
    p.add_argument('--model', default=f'{DRIVE_BASE}/models/enhanced_kannada_ocr_best.pth')
    p.add_argument('--vocab', default=f'{DRIVE_BASE}/preprocessed/vocabulary.json')
    p.add_argument('--no-plot', action='store_true', help='print the result only')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_path = args.image or input("Enter the path to your test image: ")
    try:
        test_kannada_ocr(args.model, args.vocab, image_path, show_plot=not args.no_plot)
    except FileNotFoundError as e:
        print(f"\n❌ File not found - {e}")
        print(f"Check these paths:\n  Model: {args.model}\n  Vocabulary: {args.vocab}\n  Image: {image_path}")
