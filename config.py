
class Config(object):
    def __init__(self):
        self.path = './data/CAMRa2011/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./data/CAMRa2011/groupMember.txt"
        self.embedding_size = 32
        self.epoch = 3
        self.test_epoch = 5
        self.num_negatives = 6
        self.batch_size = 10036
        self.test_batch_size = 10036
        self.lr = [0.00005, 0.0001, 0.00005]
        self.drop_ratio = 0.2
        self.topK = 5
