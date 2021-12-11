import os
import sys
import json

import random
#from M2TR_model import create_FocalLoss
import torch
from tqdm import tqdm




def read_split_data(root: str, val_rate: float = 0.2):  #modify the val_rate to change the rate of validation
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in classes:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))


    return train_images_path, train_images_label, val_images_path, val_images_label




def center_loss(x, label, center):
    b, c, h, w = x.size()
    losspos = 0
    posnum = 0

    lossneg = 0
    negnum = 0
    for i in range(b):
        if label[i] == 0:
            losspos = losspos + torch.sqrt(torch.sum((x[i] - center)**2))
            posnum = posnum + 1
        elif label[i] == 1:
            lossneg = lossneg + torch.sqrt(torch.sum((x[i] - center)**2))
            negnum = negnum + 1
    if posnum == 0:
        loss = -lossneg/negnum
    elif negnum == 0:
        loss = losspos/posnum
    else:
        loss = losspos/posnum - lossneg/negnum
    return loss


#train
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)

    if epoch%1 == 0:
        lr = lr * (0.5) ** (epoch/5)

    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        pred = model(images.to(device))



        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()




        #labels_t = torch.sparse.torch.eye(2).index_select(0, labels)
        loss = loss_function(pred, labels.to(device))


        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, lr

#validation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        #labels = torch.sparse.torch.eye(2).index_select(0, labels)
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
