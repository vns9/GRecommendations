import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "ctrain.txt")
        self.user_testMatrix = self.load_rating_file_as_matrix(user_path + "ctest.txt")
        self.user_Matrix = self.load_rating_file_as_matrix(user_path + "SortedMovies.txt")
        self.num_users, self.num_items = self.user_Matrix.shape
        self.gdata = self.load_genre_file_as_tensors(group_path+"Genre.txt")
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testMatrix = self.load_rating_file_as_matrix(group_path + "Test.txt")

    
    def load_genre_file_as_tensors(self, filename):
        genreSource = open(filename, 'r')
        genreText = genreSource.readlines()
        genres = []
        dict={}
        for liner in genreText:
            a = liner.split(" ")
            dict[str(a[0])] = a[1]
        return dict


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                negatives = []
                for x in arr[1:]: 
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])#*10
                    if (rating > 0):
                        mat[user, item] = rating
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        actual_ratings = []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            pos_item_input.append(i)
            user_input.append(u)
            actual_ratings.append(train[(u,i)])
        return {0:user_input, 1:pos_item_input, 2:actual_ratings }

    def get_user_dataloader(self, batch_size):
        user = self.get_train_instances(self.user_trainMatrix)[0]
        pos_item_input = self.get_train_instances(self.user_trainMatrix)[1]
        ratings = self.get_train_instances(self.user_trainMatrix)[2]
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(pos_item_input), torch.LongTensor(ratings))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_user_test_dataloader(self, batch_size):
        user = self.get_train_instances(self.user_testMatrix)[0]
        pos_item_input = self.get_train_instances(self.user_testMatrix)[1]
        ratings = self.get_train_instances(self.user_testMatrix)[2]
        test_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(pos_item_input), torch.LongTensor(ratings))
        user_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return user_test_loader

    def get_group_dataloader(self, batch_size):
        group = self.get_train_instances(self.group_trainMatrix)[0]
        pos_item_input = self.get_train_instances(self.group_trainMatrix)[1]
        ratings = self.get_train_instances(self.group_trainMatrix)[2]
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(pos_item_input), torch.LongTensor(ratings))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader

    def get_group_test_dataloader(self, batch_size):
        group = self.get_train_instances(self.group_testMatrix)[0]
        pos_item_input = self.get_train_instances(self.group_testMatrix)[1]
        ratings = self.get_train_instances(self.group_testMatrix)[2]
        test_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(pos_item_input), torch.LongTensor(ratings))
        group_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return group_test_loader