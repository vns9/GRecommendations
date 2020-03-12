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


# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    # # lr decay
    # if epoch_id % 5 == 0:
    #     lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

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
        # Zero_grad
        model.zero_grad()
        # Loss
        d_r = r.double()
        d_r = d_r/5
        #print(pos_prediction)
        loss = torch.sqrt(torch.mean((pos_prediction-d_r) **2))
        total_loss+=loss
        counter+=1
        # record loss history
        losses.append(loss)
        # Backward
        loss.backward()
        optimizer.step()

    print("Loss : "),
    print(total_loss.item()/counter)
    print("Epoch id : "),
    print(epoch_id)



# def evaluation(model, helper, testRatings, testNegatives, K, type_m):
#     model.eval()
#     (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
#     hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#     return hr, ndcg


if __name__ == '__main__':
    # initial parameter class
    config = Config()

    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)

    # initial dataSet class
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    # get group number
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    genres = dataset.gdata

    # build AGREE model
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio, genres)

    # config information
    print("Model training at embedding size %d, run Iteration:%d" %(config.embedding_size, config.epoch))
    # train the model
    for epoch in range(config.epoch):
        agree.train()
        t1 = time()
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')
        #print("User training complete.")
        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("User and Group training time is: %.1f s" % (time()-t1))
    print("Done!")
