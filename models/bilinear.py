import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import e

class BILINEAR(nn.Module):

    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio, genres):

        super(BILINEAR, self).__init__()
        self.genres = genres
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim, genres)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = BilinearAttentionLayer( embedding_dim, embedding_dim, 1)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Bilinear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                

    def forward(self, group_inputs, user_inputs, item_inputs):

        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):

        group_embeds = Variable(torch.Tensor())
        gm_embeddings = Variable(torch.Tensor())
        all_item_embeds = Variable(torch.Tensor())
        at_wt = []
        item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs)))
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[int(i)]
            members_embeds = self.userembeds(Variable(torch.LongTensor(members)))
            items_numb = []
            items_numb.append(j)
            item_embeds = self.itemembeds(Variable(torch.LongTensor(items_numb)))
            for member in members_embeds:
                xmember = member.view(1,member.shape[0])
                at_wt.append(self.attention(xmember, item_embeds))
            final_user = torch.zeros([32])
            val=0
            for member in members_embeds:
                final_user = torch.add(at_wt[val]*member, final_user)
                val+=1
            if all_item_embeds.dim() == 0:
                all_item_embeds = item_embeds
            else:
                all_item_embeds = torch.cat((all_item_embeds, item_embeds))
            if group_embeds.dim() == 0:
                group_embeds = final_user
            else:
                group_embeds = torch.cat((group_embeds, final_user))
        element_embeds = torch.mul(group_embeds, all_item_embeds)
        new_embeds = torch.cat((element_embeds, group_embeds, all_item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y
        
    # user forward
    def usr_forward(self, user_inputs, item_inputs):

        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs)
        user_embeds = self.userembeds(user_inputs_var)
        item_embeds = self.itemembeds(item_inputs_var)
        element_embeds = torch.mul(user_embeds, item_embeds)
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

class UserEmbeddingLayer(nn.Module):

    def __init__(self, num_users, embedding_dim):

        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):

        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):

    def __init__(self, num_items, embedding_dim, genres):

        super(ItemEmbeddingLayer, self).__init__()
        self.genres = genres
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):

        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):

    def __init__(self, number_group, embedding_dim):

        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):

        group_embeds = self.groupEmbedding(num_group)
        return group_embeds

class BilinearAttentionLayer(nn.Module):

    def __init__(self, embedding_dim_1, embedding_dim_2, embedding_dim_3, drop_ratio=0):

        super(BilinearAttentionLayer, self).__init__()
        self.bilinear =  nn.Bilinear(embedding_dim_1, embedding_dim_2, embedding_dim_3)
        


    def forward(self, x, y):

        out = self.bilinear(x,y)
        weight = F.softmax(out.view(1, -1), dim=1)
        return out


class PredictLayer(nn.Module):

    def __init__(self, embedding_dim, drop_ratio=0):

        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
			nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):

        out = self.linear(x)
        return out