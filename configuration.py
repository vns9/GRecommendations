class HyperParam(object):
    def __init__(self):
        self.path = './datasets/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./datasets/groupMember.txt"
        self.embedding_size = 50
        self.epoch = 60
        self.test_epoch = 5
        self.batch_size = 256
        self.test_batch_size = 256
        self.lr = [0.00001, 0.00001, 0.00001]
        self.drop_ratio = 0.2
