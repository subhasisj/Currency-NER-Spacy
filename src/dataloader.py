import sys
sys.path.append('./')
import os

class DataLoader:
    def __init__(self,logger,path):
        self.logger = logger
        self.parent_filepath = path

    def read_file(self,filename):
        filename = filename+'.txt'
        try:
            full_path = os.path.join(self.parent_filepath,filename)
            self.logger.log(f'Reading file {full_path}')
            with open(full_path,encoding="utf8") as reader:
                text = reader.read()
            return text
        except Exception as e:
            self.logger.log(f' Exception occured while loading file {full_path}. Exception : {str(e)}','critical')
            raise Exception

