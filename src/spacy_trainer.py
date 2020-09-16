import pandas as pd 
import numpy as np
import spacy
import os
from spacy.matcher import Matcher
import random
from spacy.util import minibatch, compounding


class SpacyMatcher:

    def __init__(self,logger):
        self.logger = logger
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
    
    def _load_data_for_matcher(self):
        lexicon_data = os.path.join('.','Data','lexicon.csv')
        self.logger.log(f'Reading file {lexicon_data}')
        try:
            df = pd.read_csv(lexicon_data)
            df_all_phrases = df.groupby('ric')['phrase'].agg(lambda x: list(x)).reset_index()

            return df_all_phrases
        except Exception as e:
            self.logger.log(f' Exception occured while Reading file {lexicon_data}. Exception : {str(e)}','critical')

    
    def train_matcher(self):

        df_all_phrases = self._load_data_for_matcher()
        # Create matcher for each currency
        self.logger.log(f'Creating Matcher...')
        try:
            for index, row in df_all_phrases.iterrows():
                phrase_patterns = [ [ {'ORTH': phrase , 'DEP': {"NOT_IN": ["POBJ"]},  'POS':{"IN": ["PROPN", "NOUN"]} } ]  for phrase in row['phrase'] ]
                self.matcher.add(row['ric'],None,*phrase_patterns)
            
            return self.matcher,self.nlp
        
        except Exception as e:
            self.logger.log(f' Exception occured while creating Matcher object. Exception : {str(e)}','critical')

        

class SpacyTrainingDataCreator:

    def __init__(self,logger,matcher,vocab):
        self.logger = logger
        self.matcher = matcher
        self.nlp = vocab

    def create_from_text(self,text,file_name):

        try:
            doc = self.nlp(text)
            detections = []
            detections = [(doc[start:end].start_char, doc[start:end].end_char, self.nlp.vocab.strings[idx] ) for idx, start, end in self.matcher(doc)]
            return (doc.text, {'entities': detections})
        except Exception as e:
            self.logger.log(f' Exception occured while creating training data for file {file_name}. Exception : {str(e)}','critical')


class SpacyNerTrainer:

    def __init__(self,logger):
        self.logger = logger
        self.nlp = spacy.blank("en")
        self.model_save_path = os.path.join('.','model')

    def train(self,training_data):

        self.training_data = training_data
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(ner,last=True)
        else:
            ner = self.nlp.get_pipe('ner')

        for _,annotation in self.training_data:
            for ent in annotation.get('entities'):
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        optimizer = self.nlp.begin_training()
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(10):
               
                self.logger.log(f'Training for Iteration {itn+1}')
                random.shuffle(training_data)
                batches = minibatch(training_data, size=sizes)
                losses = {}
                for batch in batches:
                    try:
                        texts, annotations = zip(*batch)
                        self.nlp.update(texts, annotations, sgd=optimizer, drop=0.5, losses=losses)
                    except Exception as e:
                        self.logger.log(f' Exception occured while training model. Exception : {str(e)}','critical')
                # print("Losses", losses)
                self.logger.log(f'Losses for Iteration {itn+1} : {losses}')
                

        
        # Save model
        try:
            self.nlp.meta['name'] = 'CurrencyNERModel'  # rename model
            self.nlp.to_disk(self.model_save_path)
            self.logger.log(f'Model has been saved to {self.model_save_path}')
        
        except Exception as e:
            self.logger.log(f' Exception occured while saving model. Exception : {str(e)}','critical')