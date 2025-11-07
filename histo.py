
import time
import threading
from collections import Counter
import shutil
import numpy as np

class CLIHistogram:
    def __init__(self, bins, minval, maxval, width=50, char='▇'):
        self.bins = bins
        self.minval = minval
        self.maxval = maxval
        self.width = width
        self.char = char
        self.counts = [0]*bins
        self.lock = threading.Lock()
        self.data = []

    def add(self, val):
        self.data.append(val)
        with self.lock:
            if val < self.minval:
                idx = 0
            elif val >= self.maxval:
                idx = self.bins-1
            else:
                idx = int((val - self.minval) / (self.maxval - self.minval) * self.bins)
            self.counts[idx] += 1

    def print(self):
        hist,be = np.histogram(self.data,bins = self.bins)
        intervals = list(zip(be[:-1],be[1:]))
        with self.lock:
            maxcount = max(hist) if self.counts else 1
            cols = shutil.get_terminal_size((80,20)).columns
            bar_max = min(self.width, cols - 20)
            for i,(c,(lo,hi)) in enumerate(zip(hist,intervals)):
                bar_len = int(c / maxcount * bar_max)
                print(f"{lo:6.2f}-{hi:6.2f} | {self.char * bar_len} ({c})")

