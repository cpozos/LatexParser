from data.vocabulary import Vocabulary

class LatexGenerator(object):
    """

    """

    def __init__(self, model, vocabulary, beam_size=5, max_len=64, device="cpu"):
        """
        """
        self.model = model
        self._sign2id = vocabulary.token_id_dic
        self._id2sign = vocabulary.id_token_dic
        self.device = device

        self.beam_size = beam_size
        self._beam_search = BeamSearch(Vocabulary.END_TOKEN_ID)

    def get(self):
        """
        """
        if self.beam_size == 1:
            results = self._greedy_decoding(imgs)
        else:
            results = self._batch_beam_search(imgs)

        return results

    def _greedy_decoding(self, imgs):
        """
        """
        imgs = imgs.to(self.device)
        self.model.eval()

        enc_outs = self.model.encode(imgs)
        dec_states, O_t = self.model.init_decoder(enc_outs)

        batch_size = imgs.size(0)
        # storing decoding results
        formulas_idx = torch.ones(
            batch_size, self.max_len, device=self.device).long() * PAD_TOKEN
        # first decoding step's input
        tgt = torch.ones(
            batch_size, 1, device=self.device).long() * START_TOKEN
        with torch.no_grad():
            for t in range(self.max_len):
                dec_states, O_t, logit = self.model.step_decoding(
                    dec_states, O_t, enc_outs, tgt)

                tgt = torch.argmax(logit, dim=1, keepdim=True)
                formulas_idx[:, t:t + 1] = tgt
        results = self._idx2formulas(formulas_idx)
        return results

    def _batch_beam_search(self, imgs):
        """
        """
        pass



    def _idx2formulas(self, formulas_idx):
        """convert formula id matrix to formulas list
        """
        results = []
        for id_ in formulas_idx:
            id_list = id_.tolist()
            result = []
            for sign_id in id_list:
                if sign_id != END_TOKEN:
                    result.append(self._id2sign[sign_id])
                else:
                    break
            results.append(" ".join(result))
        return results


class BeamSearch(object)
    """
    """
    
    def  __init__(self):
        """
        """
        pass