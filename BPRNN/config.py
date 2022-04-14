"""

"""


class Configs:
    def __init__(self):
        self.snr_l = 1
        self.snr_h = 10
        self.snr_step = 1
        self.iterations = 5
        self.code_file_name = 'bp-rnn-in-mind-spore/codes/BCH_63_45'   # code change
        self.H_file = self.code_file_name + '.alist'
        self.G_file = self.code_file_name + '.gmat'
        self.result_path = './result/'

        self.batch_size = 240
        self.epochs = 50

        self.train_data_num = 10000
        self.test_data_num = 10000

        self.all_data_num = int(10000 * int((self.snr_h - self.snr_l) / self.snr_step))
