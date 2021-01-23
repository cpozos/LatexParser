#PytTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(object):
    """
    docstring
    """
    def __init__(self, enc_out_dim, num_emb, emb_dim, hidden_size, out_size, dropout=0.):

        # ******************
        # Embeding
        # ******************
        self.embedding = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
        
        # ******************
        # LSTM
        # ******************
        self.lstm = nn.LSTMCell(emb_dim + hidden_size, hidden_size)

        # ******************
        # Attention
        # ******************

        # for initialization
        self.init_wh = nn.Linear(in_features=enc_out_dim, out_features=hidden_size)
        self.init_wc = nn.Linear(enc_out_dim, hidden_size)
        self.init_wo = nn.Linear(enc_out_dim, hidden_size)

        # for the algorithm
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(hidden_size, enc_out_dim, bias=False)
        self.W_3 = nn.Linear(hidden_size + enc_out_dim, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, out_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def init_decode(self, enc_out):
        """
        Returns the initial values for h0, c0 and o0 to be used in Attention mechanism

        args:
            enc_out: the output of row encoder [B, H*W, C]
        return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        # Init decode
        mean_enc_out = enc_out.mean(dim=1)
        h = torch.tanh(self.init_wh(mean_enc_out))
        c = torch.tanh(self.init_wc(mean_enc_out))
        init_o = torch.tanh(self.init_wo(mean_enc_out))
        return (h, c), init_o 

    def step_decoding(self, dec_states, o_t, enc_out, tgt, beta):
        """
        Runing one step decoding

        returns:
            h_t (ht)
            context_t (ct)
            o_t (ot)
            logits (pt) 
        """
        # Embedding
        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # [B, emb_size+dec_rnn_h]

        # LSTM
        h_t, c_t = self.lstm(inp, dec_states)  # h_t --> [B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # Attention
        context_t, attn_scores = self.attention(enc_out, h_t, beta) # context_t --> [B,C]

        # tanh
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t) # o_t -->[B, dec_rnn_h]
        
        # softmax
        logit = F.softmax(self.W_out(o_t), dim=1)  #[B, out_size]

        return (h_t, c_t), o_t, logit

    def attention(self, enc_out, h_t, beta):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, L=H*W, C]
            h_t: the current time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, C]
            attn_scores: Attention scores
        """
        # cal alpha
        W1_et = self.W_1(enc_out) # W1 x enc_out
        W2_ht = self.W_2(h_t).unsqueeze(1) # W2 x ht
        alpha = torch.tanh(W1_et + W2_ht) # at = tanh (W1 x enc_out + W2 x ht)
        alpha = torch.sum(beta*alpha, dim=-1)  #a = SUM (b * at)  --> [B, L]

        # softmax
        alpha = F.softmax(alpha, dim=-1)  #a_ = softmax(a) --> [B, L]

        # compute weigths: context: [B, C]
        context = torch.bmm(alpha.unsqueeze(1), enc_out) # c_t = a_ * e_t
        context = context.squeeze(1)
        return context, alpha
  