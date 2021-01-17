import pickle as pkl
import os
from collections import Counter

class Vocabulary(object):
    RAW_DATA_PATH = "src\data\sets\\raw\im2latex_train_filter.lst"
    FILE_PATH = "src\data\sets\\raw\latex_vocab.txt"
    PROCESSED_PATH = "src\\data\\sets\\processed\\vocab.pkl"

    def __init__(self):
        self.load()

    def _initialize(self, use_txt = False):

        # Build the dictionaries with default values
        self.token_id_dic = {
            'PAD': 0, 'GO': 1, 'EOS': 2, 'UNKNOWN': 3
        }

        # Uses the latex_vocab.txt file
        if use_txt :
            file_text = open(Vocabulary.FILE_PATH).readlines()
            for i,x in enumerate(file_text):
                
                # Updates tokens dictionaries
                token = x.split('\n')[0]
                self.add_item(token)           
        self.id_token_dic = dict((id, token) for token, id in self.token_id_dic.items())



    def __len__(self):
        return len(self.token_id_dic)

    def add_token(self, token):
        if token not in self.token_id_dic:
            length = self.__len__()
            self.token_id_dic[token] = length
            self.id_token_dic[length] = token

    def load(self):
        if self.is_already_created():
            with open(Vocabulary.PROCESSED_PATH, 'rb') as f:
                data_saved = pkl.load(f)
                self.token_id_dic = data_saved[0]
                self.id_token_dic = data_saved[1]

        else:
            self._initialize()

    def dispose(self):
        self.id_token_dic = {}
        self.token_id_dic = {}

    def save(self):
        #TODO for testing avoid saving
        #return True
        
        try:
            with open(Vocabulary.PROCESSED_PATH, 'wb') as w:
                pkl.dump([self.token_id_dic, self.id_token_dic], w)
        except Exception:
            return False
        return True

    def delete(self):
        if self.is_already_created():
            os.remove(Vocabulary.PROCESSED_PATH)

    def is_already_created(self):
        try:
            file = open(Vocabulary.PROCESSED_PATH)
        except IOError:
            return False
        return True

    def get_token(self, index):
        return self.id_token_dic[index]

    def get_index(self, token):
        return self.token_id_dic[token]