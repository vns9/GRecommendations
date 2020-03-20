from model.agree import AGREE
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset
#import matplotlib.pyplot as plt

loss_list = list()
epoch_list = list()
# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user training
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[0]
    elif epoch_id >=20:
        lr = learning_rates[0]

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    total_loss=0
    counter=0
    for batch_id, (u, pi_ni, r) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        d_r = r.double()
        d_r = d_r/5
        #print(pos_prediction)
        loss = torch.sqrt(torch.mean((pos_prediction-d_r) **2))
        total_loss+=loss
        counter+=1
        # Backward
        loss.backward()
        optimizer.step()

    print("Training epoch id : "),
    print(epoch_id)
    print("Loss : "),
    print(total_loss.item()/counter)
    
    if(type_m=='group'):
        loss_list.append(total_loss.item()/counter)
        epoch_list.append(epoch_id)
    

# test the model
def testing(model, train_loader, epoch_id, config, type_m):
    
    total_loss=0
    counter=0
    losses = []
    for batch_id, (u, pi_ni, r) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
        d_r = r.double()
        d_r = d_r/5
        loss = torch.sqrt(torch.mean((pos_prediction-d_r) **2))
        total_loss+=loss
        counter+=1

    print("Testing epoch id : "),
    print(epoch_id)
    print("Loss : "),
    print(total_loss.item()/counter)
    



if __name__ == '__main__':

    config = Config()

    helper = Helper()

    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)

    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    genres = dataset.gdata

    # build model
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio, genres)

    
    print("Model training at embedding size %d, number of epochs:%d" %(config.embedding_size, config.epoch))
    for epoch in range(config.epoch):
        agree.train()
        t1 = time()
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')
        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("User and Group training time %.1f s\n" % (time()-t1))
    
    print(loss_list)

    print("Model testing at embedding size %d, number of epochs:%d" %(config.embedding_size, config.test_epoch))
    for epoch in range(config.test_epoch):
        t1 = time()
        testing(agree, dataset.get_group_test_dataloader(config.batch_size), epoch, config, 'group')
        print("Group testing time %.1f s\n" % (time()-t1))
    
    print("Done.")