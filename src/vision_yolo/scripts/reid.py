import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import cv2

class ReID:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # 移除分类层
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def extract(self, image_bgr):
        """输入 BGR 图像（OpenCV 格式），返回归一化特征向量"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze().cpu()
        return feat / feat.norm(p=2)

