import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BAHDANAUplus(nn.Module):

    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio, genres):

        super(BAHDANAUplus, self).__init__()
        self.genres = genres
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim, genres)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.attention = ConcatAttentionLayer( embedding_dim * 4)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)

        #initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                #m.weight.data.fill_(0.01)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                #m.weight.data.fill_(0.01)

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
        all_item_embeds = Variable(torch.Tensor())
        item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs)))
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[int(i)]
            members_embeds = self.userembeds(Variable(torch.LongTensor(members)))
            items_numb = []
            items_numb.append(j)
            item_embeds = self.itemembeds(Variable(torch.LongTensor(items_numb)))
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=0)
            group_item_embeds_a = torch.reshape(group_item_embeds, (1,4*32))
            at_wt = self.attention(group_item_embeds_a)
            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            if all_item_embeds.dim() == 0:
                all_item_embeds = item_embeds
            else:
                all_item_embeds = torch.cat((all_item_embeds, item_embeds))
            if group_embeds.dim() == 0:
                group_embeds = g_embeds_with_attention
            else:
                group_embeds = torch.cat((group_embeds, g_embeds_with_attention))
        element_embeds = torch.mul(group_embeds, all_item_embeds)
        new_embeds = torch.cat((element_embeds, group_embeds, all_item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y 

    # user forward
    def usr_forward(self, user_inputs, item_inputs):

        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs)
        user_embeds = self.userembeds(user_inputs_var)
        item_embeds = self.itemembeds(item_inputs_var)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
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
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim-18)

    def forward(self, item_inputs):

        itemGenres = torch.zeros([item_inputs.size()[0], 18])
        for i in range(item_inputs.size()[0]):
            for j in range(18):
                if str(item_inputs[i].item()) in self.genres:
                    itemGenres[i][j] = int(self.genres[str(item_inputs[i].item())][j])
                else:
                    itemGenres[i][j] = 0

        item_embeds = self.itemEmbedding(item_inputs)
        item_embedds = torch.cat([item_embeds, itemGenres], dim=1)
        return item_embedds


class GroupEmbeddingLayer(nn.Module):

    def __init__(self, number_group, embedding_dim):

        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):

        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class ConcatAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, drop_ratio=0):

        super(ConcatAttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 3),
            nn.Dropout(drop_ratio)
        )

    def forward(self, x):

        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return out

class PredictLayer(nn.Module):

    def __init__(self, embedding_dim, drop_ratio=0):

        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):

        out = self.linear(x)
        return out