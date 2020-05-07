import torch
import torch.nn.functional as F
from unet import UNet
from ptsemseg.loader.dataloader import data_loader 
from torch.utils import data
from PIL import Image
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)

optim = torch.optim.Adam(model.parameters())
# data_loader = get_loader('kitti','seg')
# data_path = "/home/sombit/kitti"
t_loader = data_loader(
        
        is_transform=True,
        img_norm=False,
        # version = cfg['data']['version'],
)
trainloader = data.DataLoader(t_loader,
                                  batch_size=1, 
                                  num_workers=2, 
                                  shuffle=False)

epochs = 1
filepath = "/media/disk2/sombit/kitti_test/"
counter =0
state_dict = torch.load( "/media/disk2/sombit/kitti_seg/ck.pth")
model.load_state_dict(state_dict)

for (X, y,image_path) in trainloader:
    X = X.to(device)  # [N, 1, H, W]
    # y = y.to(device)  # [N, H, W] with class indices (0, 1)
    prediction = model(X)  
    # n, c, h, w = prediction.size()
    # nt, ht, wt = y.size()

    # Handle inconsistent size between input and target
    # if h > ht and w > wt:  # upsample labels
    #     y = y.unsequeeze(1)
    #     y = F.upsample(y, size=(h, w), mode="nearest")
    #     y = prediction.sequeeze(1)
    # elif h < ht and w < wt:  # upsample images
    #     prediction = F.upsample(prediction, size=(ht, wt), mode="bilinear")
    # elif h != ht and w != wt:
    #     raise Exception("Only support upsampling")

    # loss = F.cross_entropy(
    #     prediction, y, ignore_index=250
    # )
    #print(y.shape)
    #if (i==0):
        #t = y.numpy()
        #print(image_path)
        ## print(t.type)
        #img = np.asarray(y,dtype=np.uint8)
        ## print(t[1,:,:])
        #img = Image.fromarray(np.uint8(img[1,:,:]))
    filename = {0}.format(counter)
    img.save(filepath + filename)
    counter = counter+1
    
    # print('[%d, %5d] loss: %.3f' %(i + 1, counter + 1, loss))
    # print("loss",loss.item()," epochs",epochs)
    # optim.zero_grad()
    # loss.backward()
    # optim.step()
    # # if(i==0):
    #     print(image_path)
    #     t = y.numpy()
    #     print(np.unique(t))
    #     break



# ck_path = "/media/disk2/sombit/kitti_seg/ck.pth"   

# torch.save(model.state_dict(),ck_path )
