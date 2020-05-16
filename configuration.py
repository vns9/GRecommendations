class HyperParam(object):
    def __init__(self):
        self.num_negatives = 6
        self.path = './datasets/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./datasets/groupMember.txt"
        self.embedding_size = 32
        self.epoch = 30
        self.test_epoch = 5
        self.batch_size = 10036
        self.test_batch_size = 10036
        #self.lr = [0.005, 0.005, 0.005]
        #self.lr = [0.0001, 0.000005, 0.000005]
        self.lr = [0.0001, 0.0001, 0.0001]
        self.drop_ratio = 0.4