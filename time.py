from contextlib import contextmanager
import time

@contextmanager
def Time_decorator():
    start_time = time.time()
    yield
    end_time = time.time()
    t = end_time - start_time
    print("%f decorator seconds\n" % t)

class Time_class:
    def __enter__(self):
        self.time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.time
        print("%f class seconds\n" % t)

def main():
    with Time_decorator():
        time.sleep(2)

    with Time_class():
        time.sleep(2)

main()