import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time

from configuration import HyperParam

from util import Helper

from dataset import GDataset
# from dataset2 import GDataset2
# from dataset3 import GDataset3
# from dataset4 import GDataset4
# from dataset5 import GDataset5

from models.bahdanau import BAHDANAU
from models.bilinear import BILINEAR
from models.bahdanau2 import bahdanau2
from models.bahdanauplus import BAHDANAUplus

train_loss_list = []
test_loss_list = []

# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user training
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 20:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]

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
        loss = torch.sqrt(torch.mean((pos_prediction-d_r) **2))
        total_loss+=loss
        counter+=1
        # Backward
        loss.backward()
        optimizer.step()
    
    if(type_m=='group'):
        train_loss_list.append(total_loss.item()/counter)
    

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
        d_r = d_r/5 #ratings given in stars out of 5
        loss = torch.sqrt(torch.mean((pos_prediction-d_r) **2))
        total_loss+=loss
        counter+=1
    if(type_m=='group'):
        test_loss_list.append(total_loss.item()/counter)



if __name__ == '__main__':

    configuration = HyperParam()

    helper = Helper()

    g_m_d = helper.gen_group_member_dict(configuration.user_in_group_path)

    #---------------------------------------------------------------------------------------------------------------------------------

    dataset = GDataset(configuration.user_dataset, configuration.group_dataset)

    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    genres = dataset.gdata

    # BAHDANAU PLUS------------------------------------------------------------------------------------------------------------------
    '''
    bahdanau = BAHDANAUplus(num_users, num_items, num_group, configuration.embedding_size, g_m_d, configuration.drop_ratio, genres)
    t = time()

    for epoch in range(configuration.epoch):
        bahdanau.train()
        training(bahdanau, dataset.get_user_dataloader(configuration.batch_size), epoch, configuration, 'user')
        training(bahdanau, dataset.get_group_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    for epoch in range(configuration.test_epoch):
        testing(bahdanau, dataset.get_user_test_dataloader(configuration.batch_size), epoch, configuration, 'user')
        testing(bahdanau, dataset.get_group_test_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    print("Bahdanau+: %.1f s\n" % (time()-t))
        
    print(train_loss_list)
    print(test_loss_list)

    train_loss_list = []
    test_loss_list = []
    '''
    # BILINEAR MODEL-----------------------------------------------------------------------------------------------------------------
    '''
    bilinear = BILINEAR(num_users, num_items, num_group, configuration.embedding_size, g_m_d, configuration.drop_ratio, genres)
    t=time()
    
    for epoch in range(configuration.epoch):
        bilinear.train()
        training(bilinear, dataset.get_user_dataloader(configuration.batch_size), epoch, configuration, 'user')
        training(bilinear, dataset.get_group_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    for epoch in range(configuration.test_epoch):
        testing(bilinear, dataset.get_user_test_dataloader(configuration.batch_size), epoch, configuration, 'user')
        testing(bilinear, dataset.get_group_test_dataloader(configuration.batch_size), epoch, configuration, 'group')

    print("Bilinear: %.1f s\n" % (time()-t))
        
    print(train_loss_list)
    print(test_loss_list)

    train_loss_list = []
    test_loss_list = []
    '''
    # BENCHMARK MODEL-----------------------------------------------------------------------------------------------------------------
    '''
    bahdanau = BAHDANAU(num_users, num_items, num_group, configuration.embedding_size, g_m_d, configuration.drop_ratio, genres)
    t = time()

    for epoch in range(configuration.epoch):
        bahdanau.train()
        training(bahdanau, dataset.get_user_dataloader(configuration.batch_size), epoch, configuration, 'user')
        training(bahdanau, dataset.get_group_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    for epoch in range(configuration.test_epoch):
        testing(bahdanau, dataset.get_user_test_dataloader(configuration.batch_size), epoch, configuration, 'user')
        testing(bahdanau, dataset.get_group_test_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    print("Bahdanau: %.1f s\n" % (time()-t))
        
    print(train_loss_list)
    print(test_loss_list)

    train_loss_list = []
    test_loss_list = []
'''
    bahdanau2 = bahdanau2(num_users, num_items, num_group, configuration.embedding_size, g_m_d, configuration.drop_ratio, genres)
    t=time()
    
    for epoch in range(configuration.epoch):
        bahdanau2.train()
        #training(bahdanau2, dataset.get_user_dataloader(configuration.batch_size), epoch, configuration, 'user')
        training(bahdanau2, dataset.get_group_dataloader(configuration.batch_size), epoch, configuration, 'group')
        
    # for epoch in range(configuration.test_epoch):
    #     testing(bahdanau2, dataset.get_user_test_dataloader(configuration.batch_size), epoch, configuration, 'user')
    #     testing(bilinbahdanau2ear, dataset.get_group_test_dataloader(configuration.batch_size), epoch, configuration, 'group')

    print("Bahdanau2: %.1f s\n" % (time()-t))
        
    print(train_loss_list)
    print(test_loss_list)
 
print("Completed")