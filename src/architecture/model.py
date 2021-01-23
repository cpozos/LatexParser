#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform


INIT = 1e-2
class Model(nn.Module):

    def __init__(self, out_size, enc_out_dim=512, emb_size=80, dec_rnn_h=512, dropout=0.):
        super(Model, self).__init__()

        # ******************
        # Encoder
        # ******************
        self.cnn_encoder = nn.Sequential(

            # [BatchSize, NumberChannels, Height, Width]
            # Square kernels 3x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # [B, 128, H, W]
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),

            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1),0),

            nn.Conv2d(256, enc_out_dim,3,1,0),
            nn.ReLU()

            # [B, 512, H, W]
        )
        

        # ******************
        # Decoder
        # ******************
        self.rnn_decoder = nn.LSTMCell(emb_size + dec_rnn_h, dec_rnn_h)
        
        # Embeding
        self.embedding = nn.Embedding(num_embeddings=out_size, embedding_dim=emb_size)

        # ******************
        # Attention
        # ******************

        # for initialization
        self.init_wh = nn.Linear(in_features=enc_out_dim, out_features=dec_rnn_h)
        self.init_wc = nn.Linear(enc_out_dim, dec_rnn_h)
        self.init_wo = nn.Linear(enc_out_dim, dec_rnn_h)

        # for the algorithm
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)
        self.W_1 = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self.W_2 = nn.Linear(dec_rnn_h, enc_out_dim, bias=False)
        self.W_3 = nn.Linear(dec_rnn_h + enc_out_dim, dec_rnn_h, bias=False)
        self.W_4 = nn.Linear(dec_rnn_h, out_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.uniform = Uniform(0,1)

    def forward(self, imgs, formulas, epsilon=1.):
        """args:
        imgs: [B, C, H, W] where 
            B: Batch size
            C: Channels
            H: Height
            W: Width
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # Encoding
        encoded_imgs = self.encode(imgs)

        # Decoder's states
        dec_states, o_t = self.init_decoder(encoded_imgs)

        # Attention
        logits = []
        for t in range(formulas.size(1)):
            tgt = formulas[:,t:t+1]

            if logits and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)

            # ont step decoding
            dec_states, o_t, logit = self.step_decode(
                dec_states, o_t, encoded_imgs, tgt, self.beta)

            logits.append(logit)
        
        logits = torch.stack(logits, dim=1) #[B, MAX_LEN, out_size]
        return logits

    def encode(self, imgs):
        """
        Applies the CNN layer to encode the images
        args:
            imgs: tensors of dimension [B,3,H,W]
        """
        encoded_imgs = self.cnn_encoder(imgs)  #[B, 512, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  #[B, H', W', 512]

        # Unfold the image to get a sequence
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H*W, -1)
        return encoded_imgs # [B, H' * W']


    def init_decoder(self, enc_out):
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

    def step_decode(self, dec_states, o_t, enc_out, tgt, beta):
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
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t --> [B, dec_rnn_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # Attention
        context_t, attn_scores = self.attention(enc_out, h_t, beta) # context_t --> [B,C]

        # tanh
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t) # o_t -->[B, dec_rnn_h]
        
        # softmax
        logit = F.softmax(self.W_4(o_t), dim=1)  #[B, out_size]

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
  