"""
  @Time : 2022/3/18 14:56 
  @Author : Ziqi Wang
  @File : parallel.py 
"""


from multiprocessing import Pool
from src.utils.datastruct import ConditionalDataBuffer


class MyAsyncPool:
    def __init__(self, n):
        self.n = n
        self.pool = Pool(n)
        self.res_buffer = ConditionalDataBuffer()

    def push(self, func, *args):
        """
        Push task(s) in to the process pool
        :param func: function to be executed
        :param args: arguments of the function in the task, any number larger than 1 of tuples is valid.
        :return: None
        """
        results = [self.pool.apply_async(func, arg) for arg in args]
        self.res_buffer.push(*results)

    def collect(self):
        """
        Collect resutls of all the finished tasks
        :return: A list of resutls of all the finished tasks
        """
        roll_outs = self.res_buffer.collect(lambda x:x.ready())
        return [item.get() for item in roll_outs]
        pass

    def wait_and_get(self):
        for res in self.res_buffer.main:
            res.wait()
        return self.collect()
        pass

    def run_all(self, func, *args):
        pass

    def close(self):
        self.pool.close()

    def get_num_waiting(self):
        return len(self.res_buffer)

    def terminate(self):
        self.pool.terminate()


class TaskLoad:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.info = kwargs
    pass


class AsyncPoolWithLoad(MyAsyncPool):
    def push(self, func, *loads):
        """
        Push task(s) with addtional dictionary information into the process pool
        :param func: function to be executed
        :param args: arguments of the function in the task, any number larger than 1 of tuples is valid.
        :return: None
        """
        results = [self.pool.apply_async(func, load.args) for load in loads]
        tmp = ((res, load.info) for res, load in zip(results, loads))
        self.res_buffer.push(*tmp)

    def collect(self):
        """
        Collect resutls of all the finished tasks
        :return: A list of resutls of all the finished tasks
        """
        roll_outs = self.res_buffer.collect(lambda x: x[0].ready())
        # print(res)
        # tmp =
        return [(res.get(), info) for res, info in roll_outs]
        pass

    def wait_and_get(self):
        for res, _ in self.res_buffer.main:
            res.wait()
        res = self.collect()
        # print(res)
        return res


