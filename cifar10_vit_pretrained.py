import torch
import timm

# Assuming you have timm installed. If not, you can install it with: pip install timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.size())
