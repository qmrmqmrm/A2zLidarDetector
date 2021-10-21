import os
import os.path as op
import numpy as np
import torch
import pandas as pd


class LogFile:
    def __init__(self, ckpt_path):
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)

    def save_log(self, epoch, train_log, val_log):
        history_summary = self.merge_logs(epoch, train_log.get_history_summary(), val_log.get_history_summary())
        # if val_log.exhuastive_logger is not None:
        #     exhaust_summary = val_log.get_exhuastive_summary()
        #     exhaust_filename = self.exhaust_path + f"/{epoch}.csv"
        #     exhaust = pd.DataFrame(exhaust_summary)
        #     exhaust.to_csv(exhaust_filename, encoding='utf-8', index=False, float_format='%.4f')

        if op.isfile(self.history_filename):
            history = pd.read_csv(self.history_filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(history_summary, ignore_index=True)
        else:
            history = pd.DataFrame([history_summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        history.to_csv(self.history_filename.replace("history.csv", "temp.csv"), encoding='utf-8', index=False, float_format='%.4f')

    def merge_logs(self, epoch, train_summary, val_summary):
        summary = dict()
        summary["epoch"] = epoch
        if train_summary is not None:
            train_summary = {"!" + key: val for key, val in train_summary.items()}
            summary.update(train_summary)
            summary["|"] = 0
        if "anchor" in val_summary:
            del val_summary["anchor"]
            del val_summary["category"]
        val_summary = {"`" + key: val for key, val in val_summary.items()}
        summary.update(val_summary)
        return summary

    def save_val_log(self, val_log):
        print("validation summary:", val_log.get_history_summary())
        history_summary = self.merge_logs(0, None, val_log.get_history_summary())
        val_filename = self.history_filename[:-4] + "_val.csv"
        if op.isfile(val_filename):
            history = pd.read_csv(val_filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(history_summary, ignore_index=True)
        else:
            history = pd.DataFrame([history_summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        print("val_filename", val_filename)
        history.to_csv(val_filename, encoding='utf-8', index=False, float_format='%.4f')
        exhaust_summary = val_log.get_exhuastive_summary()
        exhaust_filename = self.exhaust_path + f"/exhaust_val.csv"
        exhaust = pd.DataFrame(exhaust_summary)
        exhaust.to_csv(exhaust_filename, encoding='utf-8', index=False, float_format='%.4f')