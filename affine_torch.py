from PIL import Image
from torch.nn import functional as F
import torch
from torchvision.transforms import ToTensor,ToPILImage

img=Image.open("./119.jpg").resize((224,224))
img.save("./ori.jpg")
img=ToTensor()(img).view(1,3,224,224)
theta=torch.autograd.Variable(torch.FloatTensor([[[0.7,0.0,0.2],[0,0.7,0.2]]]))
grid=F.affine_grid(theta,img.size())
img=F.grid_sample(img,grid).squeeze().data
img=ToPILImage()(img)
img.save("./new.jpg")
