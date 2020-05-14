import torch
import torch.nn.functional as F
from unet import UNet
from ptsemseg.loader.dataloader import data_loader 
from torch.utils import data
from PIL import Image
import numpy as np
import scipy.misc as m
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device != 'cuda'):
    print("working on CPU, BYEBYE")
    return
print (device)
model = UNet(n_classes=13, padding=True, up_mode='upsample').to(device)

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
                                  shuffle=False)
filepath = "/media/disk2/sombit/kitti_test/"
counter =0
state_dict = torch.load( "/media/disk2/sombit/kitti_seg/checkpoint.pt")
model.load_state_dict(state_dict)
model.to(device)
model.eval()
   colors = [  # [  0,   0,   0],
        [128, 64, 128],
        # [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        # [190, 153, 153],
        # [153, 153, 153],
        # [250, 170, 30],
        # [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        # [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(13), colors))

def decode_segmap_tocolor(temp, n_classes=13):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
i =1
with torch.no_grad():
        for (X, y,image_path) in trainloader:
            X= X.to(device)
            outputs = model(X)
            pred = outputs.data.max(1)[1].cpu().numpy()
            if(i):
                print(pred)
                i=0
            
            # print(pred.shape)
            # gt = labels.numpy()
            # t = prediction.numpy()
            #print(image_path)
            ## print(t.type)
            # img = np.asarray(y,dtype=np.uint8)
            ## print(t[1,:,:])
            for j in (2):
                # img = Image.fromarray(np.uint8(pred[j,:,:]))
                decoded = decode_segmap_tocolor(pred[j,:,:], n_classes=13)
                filename = "{}.png".format(counter)
                m.imsave(filepath + filename, decoded)
                counter = counter+1

            
            # running_metrics.update(gt, pred)
# for (X, y,image_path) in trainloader:
#     X = X.to(device)  # [N, 1, H, W]
#     # y = y.to(device)  # [N, H, W] with class indices (0, 1)
#     prediction = model(X)  
#     # n, c, h, w = prediction.size()
    # nt, ht, wt = y.size()

    # Handle inconsistent size between input and target
    # if h > ht and w > wt:  # upsample labels
    #     y = y.unsequeeze(1)
    #     y = F.upsample(y, size=(h, w), mode="nearest")
    #     y = prediction.sequeeze(1)
    # elif h < ht and w < wt:# upsample images
    #     prediction = F.upsample(prediction, size=(ht, wt), mode="bilinear")
    # elif h != ht and w != wt:
    #     raise Exception("Only support upsampling")

    # loss = F.cross_e  ntropy(
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
    # for j in (4):
    #     t = prediction.numpy()
    #     #print(image_path)
    #     ## print(t.type)
    #     img = np.asarray(y,dtype=np.uint8)
    #     ## print(t[1,:,:])
    #     img = Image.fromarray(np.uint8(img[j,:,:]))
    #     filename = {0}.format(counter)
    #     img.save(filepath + filename)
    #     counter = counter+1
    
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
