import TensorboardTools
import time

if __name__ == '__main__':
    with TensorboardTools.TensorboardProcess(r'.\logs', 6006):
        while True:
            time.sleep(1)