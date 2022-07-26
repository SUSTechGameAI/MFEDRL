"""
  @Time : 2022/3/10 11:15 
  @Author : Ziqi Wang
  @File : datastruct.py 
"""
import random


class RingQueue:
    def __init__(self, capacity):
        self.main = []
        self.p = 0
        self.capacity = capacity

    def push(self, item):
        if len(self.main) < self.capacity:
            self.main.append(item)
        else:
            self.main[self.p] = item
        self.p = (self.p + 1) % self.capacity

    def front(self):
        if len(self.main) < self.capacity:
            return self.main[-1]
        else:
            return self.main[(self.p - 1) % self.capacity]

    def rear(self):
        if len(self.main) < self.capacity:
            return self.main[0]
        else:
            return self.main[self.p]

    def to_list(self):
        return self.main[self.p:] + self.main[:self.p]

    def clear(self):
        self.main.clear()
        self.p = 0

    def __len__(self):
        return len(self.main)


class ConditionalDataBuffer:
    def __init__(self):
        self.main = []

    def push(self, *items):
        """
        Push however many items into the buffer
        :param items: Any objects
        :return: None
        """
        for item in items:
            self.main.append(item)

    def collect(self, condition):
        """
        Collect all items that meet the condition and remove them from the buffer
        :param condition: A function to describe the collecting condition
        :return: All items in the buffer that meet the condition
        """
        i, j = 0, 0
        roll_outs = []
        while j < len(self):
            item = self.main[j]
            if condition(item):
                roll_outs.append(item)
            else:
                if j != i:
                    self.main[i] = item
                i += 1
            j += 1
        del self.main[i:]
        return roll_outs
        pass

    def __str__(self):
        return f'ConditionalBuffer({len(self)}): {str(self.main)}'

    def __len__(self):
        return len(self.main)

def batched_iter(arr, n):
    for s in range(0, len(arr), n):
        e = min(s + n, len(arr))
        n = e - s
        yield arr[s:e], n

#
# if __name__ == '__main__':
#     seq = [random.randrange(0, 10) for _ in range(20)]
#     buffer = ConditionalDataBuffer()
#     for batch, _ in batched_iter(seq, 4):
#         buffer.push(*batch)
#         print('push', batch)
#         roll_out = buffer.collect(lambda x: x > 6)
#         print('collected', roll_out)
#         print(buffer)
