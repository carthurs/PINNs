import subprocess
import os

class TensorboardProcess(object):

    def __init__(self, log_dir, local_webserver_port):
        self.log_dir = log_dir
        self.port = local_webserver_port

    def __enter__(self):
        self.process = subprocess.Popen(
                                    ['tensorboard', '--logdir={}'.format(self.log_dir),
                                     '--port', '{}'.format(self.port)])
        return self

    def __exit__(self, type, value, traceback):
        self.process.terminate()