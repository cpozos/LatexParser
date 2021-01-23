from architecture.encoder import CNNEncoder
from architecture.decoder import Decoder

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

        # Encoder
        self.cnn_encoder = CNNEncoder(enc_out_dim)

        # Decoder
        self.rnn_decoder = Decoder(
            enc_out_dim= enc_out_dim,
            num_emb=out_size,
            emb_dim=emb_size,
            hidden_size=dec_rnn_h,
            out_size=out_size,
            dropout=dropout
        )

        self.uniform = Uniform(0,1)

        # For Attention mechanism
        self.beta = nn.Parameter(torch.Tensor(enc_out_dim))
        init.uniform_(self.beta, -INIT, INIT)

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
        encoded_imgs = self.cnn_encoder.encode(imgs)

        # Decoder's states
        dec_states, o_t = self.rnn_decoder.init_decode(encoded_imgs)

        # ??
        logits = []
        for t in range(formulas.size(1)):
            tgt = formulas[:,t:t+1]

            if logits and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(torch.log(logits[-1]), dim=1, keepdim=True)

            # ont step decoding
            dec_states, o_t, logit = self.rnn_decoder.step_decoding(dec_states, o_t, encoded_imgs, tgt, self.beta)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=1) #[B, MAX_LEN, out_size]
        return logits