import sys
sys.path.append('./')
import os
import pandas as pd
import numpy as np
from datetime import datetime
from logger import Logger
from dataloader import DataLoader
from text_preprocessor import TextPreprocessor
from spacy_trainer import SpacyMatcher,SpacyTrainingDataCreator,SpacyNerTrainer
import pickle
from tqdm import tqdm

class NerEngine:

    def __init__(self):
        self._time_created = datetime.now()
        self._training_labels_path = os.path.join('.','Data','train_labels.csv')
        self._training_files_path = os.path.join('.','Data','train')
        self._testing_files_path = os.path.join('Data','test')
        self._testing_labels_path = os.path.join('Data','test_labels.csv')
        self.logger = Logger(f'training_logs_{self._time_created.date()}_{self._time_created.strftime("%H%M%S")}.log')
        self.text_cleaner = TextPreprocessor(self.logger)
    
    def run(self):
        file_names = self._load_training_dataset()
        data_loader = DataLoader(self.logger,self._training_files_path)
        spacy_matcher = SpacyMatcher(self.logger)
        matcher,vocab = spacy_matcher.train_matcher()
        data_creator = SpacyTrainingDataCreator(self.logger,matcher,vocab)
        trainer = SpacyNerTrainer(self.logger)
        training_data = []
        print('Creating Training Data...')
        print('-------------------------')
        for file_name in tqdm(file_names):

            try:

            # load text
                text = data_loader.read_file(file_name)

                # clean text
                text = self.text_cleaner.clean_text(text)

                # create training data for spacy
                training_data.append(data_creator.create_from_text(text,file_name))
            except Exception as e:
                continue

        print('Training Model now...')
        print('----------------------')
        model = trainer.train(training_data)
        
   
    def _load_training_dataset(self):
        self.logger.log(f'Training: Loading files from {self._training_labels_path}')
        try:
            df = pd.read_csv( self._training_labels_path )
            unique_file_names = np.unique(df.doc_id)
            self.logger.log(f'Training: Files found: {len(unique_file_names)}')
            return unique_file_names
        except Exception as e:
            self.logger.log(f'Training: Exception occured, Exception : {str(e)}','critical')
      


if __name__ == "__main__":

    engine = NerEngine()
    engine.run()
