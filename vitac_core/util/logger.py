# import matplotlib
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt
import numpy as np
# import scipy
import pickle
import os
import csv

class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0

    def log_kv(self, key, value):
        # logs the (key, value) pair

        # TODO: This implementation is error-prone:
        # it would be NOT aligned if some keys are missing during one iteration.
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path, extra_path='train'):
        # TODO: Validate all lengths are the same.
        pickle.dump(self.log, open(save_path + '/log_%s.pickle'%extra_path, 'wb'))
        with open(save_path + '/log_%s.csv'%extra_path, 'w') as csv_file:
            fieldnames = list(self.log.keys())
            if 'iteration' not in fieldnames:
                fieldnames = ['iteration'] + fieldnames

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {'iteration': row}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            # TODO: this is very error-prone (alignment is not guaranteed)
            row_dict[key] = self.log[key][-1]
        return row_dict

    def shrink_to(self, num_entries):
        for key in self.log.keys():
            self.log[key] = self.log[key][:num_entries]

        self.max_len = num_entries
        # assert min([len(series) for series in self.log.values()]) == \
        #     max([len(series) for series in self.log.values()])

    def read_log(self, log_path):
        # assert log_path.endswith('log_train.csv')

        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row, row_dict in enumerate(listr):
                for key in keys:
                    try:
                        data[key].append(eval(row_dict[key]))
                    except:
                        print("ERROR on reading key {}: {}".format(key, row_dict[key]))

                if 'iteration' in data and data['iteration'][-1] != row:
                    raise RuntimeError("Iteration %d mismatch -- possibly corrupted logfile?" % row)

        self.log = data
        self.max_len = max(len(v) for k, v in self.log.items())
        print("Log read from {}: had {} entries".format(log_path, self.max_len))


class LossLog(DataLog):
    def __init__(self):
        super().__init__()

    def log_epoch(
        self,
        epoch_data: dict,
        is_val: bool = False
    ) -> None:
        train_val_str = "val_" if is_val else "train_"
        for key, value in epoch_data.items():
            self.log_kv(train_val_str + key, float(value))

