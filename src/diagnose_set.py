import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pre_process import get_dataloaders
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

def denormalize(tensor):
    """Reverse the normalization to display images properly"""
    # Reverse: (x - 0.5) / 0.5
    return tensor * 0.5 + 0.5

def visualize_predictions(model, test_loader, device, num_images=16):
    model.to(device)
    model.eval()
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images_device = images.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(images_device)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    
    preds = (probs >= 0.5).astype(int)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = "Crack" if labels[i] == 1 else "No Crack"
        pred_label = "Crack" if preds[i] == 1 else "No Crack"
        confidence = probs[i] if preds[i] == 1 else (1 - probs[i])
        
        color = 'green' if preds[i] == labels[i] else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", 
                         color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to test_predictions.png")
    plt.close()

def check_image_statistics(test_loader):
    """Check if there are obvious visual differences between classes"""
    print("\n=== CHECKING IMAGE STATISTICS ===")
    
    class_0_means = []
    class_1_means = []
    
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            mean_intensity = img.mean().item()
            if label == 0:
                class_0_means.append(mean_intensity)
            else:
                class_1_means.append(mean_intensity)
    
    print(f"Class 0 (No Crack) - Mean intensity: {np.mean(class_0_means):.4f} ± {np.std(class_0_means):.4f}")
    print(f"Class 1 (Crack) - Mean intensity: {np.mean(class_1_means):.4f} ± {np.std(class_1_means):.4f}")
    print(f"Difference: {abs(np.mean(class_0_means) - np.mean(class_1_means)):.4f}")
    
    if abs(np.mean(class_0_means) - np.mean(class_1_means)) > 0.1:
        print("⚠️ WARNING: Large intensity difference between classes!")
        print("   The model might be using brightness instead of crack patterns.")

def check_model_confidence(model, test_loader, device):
    """Check the distribution of prediction confidence"""
    model.to(device)
    model.eval()
    
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    
    print("\n=== MODEL CONFIDENCE DISTRIBUTION ===")
    print(f"Min probability: {all_probs.min():.4f}")
    print(f"Max probability: {all_probs.max():.4f}")
    print(f"Mean probability: {all_probs.mean():.4f}")
    print(f"Std probability: {all_probs.std():.4f}")
    
    # Count very confident predictions
    very_confident = np.sum((all_probs < 0.01) | (all_probs > 0.99))
    print(f"Very confident predictions (< 0.01 or > 0.99): {very_confident}/{len(all_probs)}")
    
    if very_confident == len(all_probs):
        print("⚠️ Model is extremely confident on ALL predictions!")
        print("   This suggests the task might be too easy or there's a shortcut feature.")
        
def check_image_dimensions_and_file_sizes():
    """Check if there are systematic differences in file properties"""
    from pathlib import Path
    import cv2
    
    test_root = Path("data/test")
    
    class_0_sizes = []
    class_1_sizes = []
    
    for img_path in (test_root / "0_no_crack").glob("*.jpg"):
        img = cv2.imread(str(img_path))
        class_0_sizes.append(img_path.stat().st_size)
    
    for img_path in (test_root / "1_crack").glob("*.jpg"):
        img = cv2.imread(str(img_path))
        class_1_sizes.append(img_path.stat().st_size)
    
    print("\n=== FILE SIZE CHECK ===")
    print(f"No Crack - Mean file size: {np.mean(class_0_sizes):.0f} bytes")
    print(f"Crack - Mean file size: {np.mean(class_1_sizes):.0f} bytes")
    print(f"Difference: {abs(np.mean(class_0_sizes) - np.mean(class_1_sizes)):.0f} bytes")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, _ = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=32,
        num_workers=2,
    )
    
    # Load trained model
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    
    # Run diagnostics
    check_image_statistics(test_loader)
    check_model_confidence(model, test_loader, device)
    visualize_predictions(model, test_loader, device)
    
    print("\n✓ Diagnostics complete!")
    print("Check 'test_predictions.png' to see what the model is seeing.")

if __name__ == "__main__":
    main()