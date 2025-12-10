import torch, unigaze
model = unigaze.load('unigaze_h14_joint', device='cuda' if torch.cuda.is_available() else 'cpu')
image_normalized_batch = torch.ones((10, 3, 224, 224), device=next(model.parameters()).device)
pred_gaze = model(image_normalized_batch)['pred_gaze']
print('pred_gaze shape:', pred_gaze.shape)
