import pickle as pkl
import os
from collections import Counter

# Project
from utilities.system import apply_system_format

class Vocabulary(object):
    START_TOKEN_ID = 0
    PAD_TOKEN_ID = 1
    END_TOKEN_ID =2
    UNK_TOKEN_ID = 3
    RAW_DATA_PATH = "src\\data\\sets\\raw\\im2latex_train_filter.lst"
    FILE_PATH = "src\\data\\sets\\raw\latex_vocab.txt"
    PROCESSED_PATH = "src\\data\\sets\\processed\\vocab.pkl"

    def __init__(self):
        # Fix paths
        Vocabulary.RAW_DATA_PATH = apply_system_format(Vocabulary.RAW_DATA_PATH)
        Vocabulary.FILE_PATH = apply_system_format(Vocabulary.FILE_PATH)
        Vocabulary.PROCESSED_PATH = apply_system_format(Vocabulary.PROCESSED_PATH)
        self.max_token_len = 0
        self.load()
        
    def _initialize(self, use_txt = False):

        # Build the dictionaries with default values
        self.token_id_dic = {
            '<s>': Vocabulary.START_TOKEN_ID, 
            '</s>': Vocabulary.END_TOKEN_ID, 
            '<pad>': Vocabulary.PAD_TOKEN_ID,
            '<unk>': Vocabulary.UNK_TOKEN_ID
        }

        # Uses the latex_vocab.txt file
        if use_txt :
            file_text = open(Vocabulary.FILE_PATH).readlines()
            for i,x in enumerate(file_text):
                # Updates tokens dictionaries
                token = x.split('\n')[0]
                self.add_item(token) 
        
        self.id_token_dic = dict((id, token) for token, id in self.token_id_dic.items())

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

        self.max_token_len = max([len(form) for form in self.token_id_dic.keys()])

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
            self.max_token_len = 0
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
        
    def __len__(self):
        return len(self.token_id_dic)

    def print_info(self):
        print(f"Tokens: {len(self.token_id_dic)}")
        print(f"Max token len: {self.max_token_len}")