import logging
import os

class Logger:
    def __init__(self,logfile):
        self._logfile = logfile
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # logging.basicConfig(filename=f'logs/{self._logfile}',
        # format='%(asctime)s %(levelname)s %(message)s',
        # datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)
        logging.basicConfig(handlers=[logging.FileHandler(f'logs/{self._logfile}', mode='w')],
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)

    def log(self, log_message,level='info'):
        getattr(logging, level)(log_message)


class DummyLogger:

    def __init__(self,logfile):
        pass
    def log(self, log_message,level='info'):
        print(log_message)