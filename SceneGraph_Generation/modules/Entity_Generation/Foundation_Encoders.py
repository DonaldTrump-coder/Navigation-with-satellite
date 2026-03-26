import torch.nn as nn
import torch

class CLIPEncoder(nn.Module):
    def __init__(self, clipmodel):
        super(CLIPEncoder, self).__init__()
        self.model = clipmodel
        for param in self.model.parameters():
            param.requires_grad = False # freeze the model
        
    def forward(self, image):
        image_features = self.model.encode_image(image)
        return image_features
    
if __name__ == '__main__':
    import torch, open_clip
    from PIL import Image
    # test for clip encoder
    
    model_name = 'RN50'
    ckpt_path = 'D:/Projects/Navigation-with-satellite/SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-RN50.pt'
    img_path = 'C:/Users/10527/Pictures/original.png'
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_path)
    model = CLIPEncoder(model)
    model = model.cuda().eval()
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model(image.cuda())
        print(image_features.shape)