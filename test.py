
from M2TR_model import create_model

import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
import re

import pandas as pd
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from my_dataset import MyDataSet
from utils import evaluate


to_tensor = ToTensor()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loadpath = 'model-7.pth' #modify here to change your model data
datapath = r'C:\Users\satomi ishihara\za\desktop\fakeface\深度伪造人脸检测数据集\image\test'
testpath = r'C:\Users\satomi ishihara\za\desktop\fakeface\test_fake'
savecsvpath = 'test.csv'
Resize = transforms.Resize(size=(224, 224))

def test1():
   model = create_model().to(device)
   model.load_state_dict(torch.load(loadpath))
   model.eval()
   testlist = os.listdir(datapath)
   testlist.sort(key = lambda x:int(re.match('\D+(\d+)\.jpg',x).group(1)))
   csv_list = []
   #print(testlist)
   for i in range(len(testlist)):
       testimg = Image.open(datapath +'/' + testlist[i])
       testimg = to_tensor(testimg)
       testimg = testimg.unsqueeze(0)
       testimg = Resize(testimg)
       pred = model(testimg.to(device))
       pred_classes = torch.max(pred, dim=1)[1]

       t = testlist[i] + '\t%d' % pred_classes
       print(t)
       csv_list.append(t)

   return csv_list


def test2(root):
    model = create_model().to(device)
    model.load_state_dict(torch.load(loadpath))
    model.eval()

    dataset = dset.ImageFolder(root,
                               transform=transforms.Compose([transforms.Resize((224, 224)),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                                  (0.5, 0.5, 0.5)),
                                                             ]))
    dataloader = DataLoader(dataset,
                                shuffle=False,
                                batch_size=8,
                                num_workers=0)


    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    #dataloader = tqdm(dataloader)

    for idx, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        sample_num += img.shape[0]
        pred = model(img)
        pred_classes = torch.max(pred, dim=1)[1]
        accu = torch.eq(pred_classes, label).sum()/img.shape[0]
        accu_num += accu
        print('step:%d, accu:%f'%(idx, accu))
        #dataloader.desc = "acc: {:.3f}".format(accu_num.item() / sample_num)


    print(accu_num)

def test3(root):
    model = create_model().to(device)
    model.load_state_dict(torch.load(loadpath))
    model.eval()
    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    val_images_path = []
    val_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in classes:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]
        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transforms.Compose([transforms.Resize(224),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)
    val_loss, val_acc = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=1)



csvlist = test1()
csvlist = pd.DataFrame(data=csvlist)
csvlist.to_csv('submion.csv', encoding='gbk',index=False,header=None)
#test3(root=testpath)