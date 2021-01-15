class Vocabulary(object):
    
    def __init__(self):

        # Build the dictionaries 
        self.tokens = {
            'PAD': 0, 'GO': 1, 'EOS': 2, 'UNKNOWN': 3
        }
        self.tokensIndexes = {
            0:'PAD', 1: 'GO', 2:'EOS', 3:'UNKNOWN'
        }

        file_text = open('latex_vocab.txt').readlines()
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
