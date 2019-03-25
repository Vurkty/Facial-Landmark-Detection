class DefaultConfig(object):
    def __init__(self):
        self.scalesize = 240
        self.cropsize = 224

        self.train_batch_size = 8
        self.test_batch_size = 8

        self.train_path = '/home/vurkty/Dataset/3K-Faces/train/Pics/01/'
        self.train_csv_path = '/home/vurkty/Dataset/3K-Faces/train/Csvs/01.csv'

        self.test_path = '/home/vurkty/Dataset/3K-Faces/test/Pics/01/'
        self.test_csv_path = '/home/vurkty/Dataset/3K-Faces/test/Csvs/01.csv'

        # self.train_path = '/home/vurkty/Dataset/300VW-TRANS/Train/Pics/'
        # self.train_csv_path = '/home/vurkty/Dataset/300VW-TRANS/Train/train.csv'
        #
        # self.test_path = '/home/vurkty/Dataset/300VW-TRANS/Test/Pics/'
        # self.test_csv_path = '/home/vurkty/Dataset/300VW-TRANS/Test/test.csv'
