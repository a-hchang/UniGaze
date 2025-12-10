import torch
import unigaze

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

# Load model (downloads weights from HF on first use)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nLoading model on {device}...")
model = unigaze.load("unigaze_h14_joint", device=device)
print("Model loaded successfully!")

# Input: normalized batch (B, 3, 224, 224)
image_normalized_batch = torch.ones((10, 3, 224, 224), device=device)
print(f"\nInput shape: {image_normalized_batch.shape}")

# Output: {'pred_gaze': (B, 2)} with (pitch, yaw)
pred_gaze = model(image_normalized_batch)['pred_gaze']
print(f"Output shape: {pred_gaze.shape}")
print(f"First prediction (pitch, yaw): {pred_gaze[0]}")
print("\n✅ Quick Start test completed successfully!")
