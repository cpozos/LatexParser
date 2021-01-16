class Vocabulary(object):
    FILE_PATH = "src\data\sets\\raw\latex_vocab.txt"

    def __init__(self):

        # Build the dictionaries 
        self.tokens = {
            'PAD': 0, 'GO': 1, 'EOS': 2, 'UNKNOWN': 3
        }
        self.tokensIndexes = {
            0:'PAD', 1: 'GO', 2:'EOS', 3:'UNKNOWN'
        }

        file_text = open(Vocabulary.FILE_PATH).readlines()
        for i,x in enumerate(file_text):
            
            # Updates tokens dictionaries
            t = x.split('\n')[0]
            self.tokens[t] = i + 4
            self.tokensIndexes[i] = t

        self.size = len(self.tokens.keys())

    def get_token(self, index):
        return self.tokensIndexes[index]

    def get_index(self, token):
        return self.tokens[token]

    def get_tokens_dic(self):
        return self.tokens

    def get_indexes_dic(self):
        return self.tokensIndexes

    def build_custom_vocabulary(count = 10):
        # TODO : impement build_vocab from build_vocab.py
        return
