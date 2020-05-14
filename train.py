import torch
import torch.nn.functional as F
from unet import UNet
from ptsemseg.loader.dataloader import data_loader 
from torch.utils import data
from PIL import Image
import numpy as np
def save_ckp(state):
    f_path = "/media/disk2/sombit/kitti_seg/checkpoint.pt"
    torch.save(state, f_path)
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    return model, optim, checkpoint['epoch']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
model = UNet( padding=True, up_mode='upsample').to(device)
print("Load Model")
optim = torch.optim.Adam(model.parameters())
# data_loader = get_loader('kitti','seg')
# data_path = "/home/sombit/kitti"
t_loader = data_loader(
        
        is_transform=True,
        img_norm=False,
        # version = cfg['data']['version'],
)

trainloader = data.DataLoader(t_loader,
                                  batch_size=2, 
                                  num_workers=2, 
                                  shuffle=True)

epochs = 70
resume = False
if(resume):
    model, optim, start_epoch = load_ckp("/media/disk2/sombit/kitti_seg/checkpoint.pt", model, optim)
    i = start_epoch
print("Started Training")
# import shutil


for i in range(epochs):
    counter =0
    running_loss = 0.0
    for (X, y,image_path) in trainloader:
        X = X.to(device)  # [N, 1, H, W]
        y = y.to(device)  # [N, H, W] with class indices (0, 1)
        model.train()
        prediction = model(X)  
        n, c, h, w = prediction.size()
        nt, ht, wt = y.size()

        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            y = y.unsequeeze(1)
            y = F.upsample(y, size=(h, w), mode="nearest")
            y = prediction.sequeeze(1)
        elif h < ht and w < wt:  # upsample images
            prediction = F.upsample(prediction, size=(ht, wt), mode="bilinear")
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        loss = F.cross_entropy(
            prediction, y, ignore_index=250
        )
        #print(y.shape)
        #if (i==0):
            #t = y.numpy()
            #print(image_path)
            ## print(t.type)
            #img = np.asarray(y,dtype=np.uint8)
            ## print(t[1,:,:])
            #img = Image.fromarray(np.uint8(img[1,:,:]))

            # img.save('test.png')
            # break
        # print('[%d, %5d] loss: %.3f' %(i + 1, counter + 1, loss))
     
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss +=loss.item()
        if counter%10==9:
            print("loss",running_loss/10," epochs",i+1,"counter",counter)
            running_loss =0.0
        counter +=1
        # if(i==0):
        #     print(image_path)
        #     t = y.numpy()
        #     print(np.unique(t))
        #     break



ck_path = "/media/disk2/sombit/kitti_seg/ck2.pth"   

state_curr = {
    'epoch': i+1,
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
}
save_ckp(state_curr)
