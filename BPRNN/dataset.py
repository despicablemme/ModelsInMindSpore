import numpy as np
from mindspore.dataset import NumpySlicesDataset
from sklearn.utils import shuffle


def get_variance(snr, r):
    ebn0 = 10 ** (snr / 10)
    avg_e = 1
    var = avg_e / (1 * ebn0 * r)

    return var


class Dataset:
    def __init__(self, train_data_num, test_data_num, snr_low, snr_high, snr_step, iteration, G, n, m):
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.snr_step = snr_step
        self.iteration = iteration
        self.G = G
        self.n = n
        self.k = n - m
        self.r = 1 - m / n

        self.random_code_ind = 0

    def get_test_data(self, phase):

        inputs_all = {}
        labels_all = {}

        for snr in range(self.snr_low, self.snr_high, self.snr_step):
            var = get_variance(snr, self.r)

            code = self.get_code(phase)
            bpsk_symbol = np.where(code == 0, 1.0, -1.0)

            vars = np.array([var] * self.test_data_num, dtype=np.float64).reshape((self.test_data_num, 1))

            inputs = np.concatenate([bpsk_symbol, vars], axis=1)
            labels = code

            inputs_all[str(snr)] = inputs
            labels_all[str(snr)] = labels

        return inputs_all, labels_all

    def get_train_data(self, phase):

        inputs_all = []
        vars_all = []
        labels_all = []

        for snr in range(self.snr_low, self.snr_high, self.snr_step):
            var = get_variance(snr, self.r)

            code = self.get_code(phase)
            bpsk_symbol = np.where(code == 0.0, 1.0, -1.0).astype(np.float32)
            vars = np.array([var] * self.train_data_num, dtype=np.float32).reshape((self.train_data_num, 1))
            inputs = np.concatenate((bpsk_symbol, vars), axis=1)
            labels = code

            inputs_all.append(inputs)
            # vars_all.append(vars)
            labels_all.append(labels)  # (batch_size, n)
        inputs_all = np.concatenate(inputs_all, 0)
        # vars_all = np.concatenate(vars_all, 0)
        labels_all = np.concatenate(labels_all, 0)
        labels_all = np.tile(labels_all, (1, self.iteration))
        # todo 数据集建立
        inputs_all, labels_all = shuffle(inputs_all, labels_all)
        dataset = NumpySlicesDataset((inputs_all, labels_all), column_names=['codes', 'labels'])
        return dataset

    def get_code(self, phase):
        if phase == 'all_zero':
            code = np.zeros((self.train_data_num, self.n), dtype=np.float32)
        elif phase == 'random':
            # todo : get data from G
            messages = np.np.random.randint(0, 2, [self.train_data_num, self.k], dtype=np.float32)
            code = np.dot(self.G, messages) % 2
        else:
            raise ValueError
        return code
