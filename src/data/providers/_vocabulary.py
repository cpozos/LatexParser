import pickle as pkl
import os

class Vocabulary(object):
    FILE_PATH = "src\data\sets\\raw\latex_vocab.txt"
    PROCESSED_PATH = "src\\data\\sets\\processed\\vocab.pkl"

    def __init__(self):
        self.load()

    def _initialize(self, use_txt = False):

        # Build the dictionaries with default values
        self.token_id_dic = {
            'PAD': 0, 'GO': 1, 'EOS': 2, 'UNKNOWN': 3
        }
        self.length = 4

        # Uses the latex_vocab.txt file
        if use_txt :
            file_text = open(Vocabulary.FILE_PATH).readlines()
            for i,x in enumerate(file_text):
                
                # Updates tokens dictionaries
                token = x.split('\n')[0]
                self.add_item(token)

        self.id_token_dic = dict((id, token) for token, id in self.token_id_dic.items())

    def __len__(self):
        return self.length

    def add_token(self, token):
        if token not in self.token_id_dic:
            self.token_id_dic[token] = self.length
            self.id_token_dic[self.length] = token
            self.length += 1

    def load(self):
        if self.is_already_created():
            with open(Vocabulary.PROCESSED_PATH, 'rb') as f:
                self = pkl.load(f)
        else:
            self._initialize()

    def dispose(self):
        self.id_token_dic = None
        self.token_id_dic = None
        self.length = 0

    def save(self):
        #TODO for testing avoid saving
        return True
        
        try:
            with open(Vocabulary.PROCESSED_PATH, 'wb') as w:
                pkl.dump(self, w)
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