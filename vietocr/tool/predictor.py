import torch

from vietocr.tool.translate import build_model, process_input_batch, translate


class Predictor:
    def __init__(self, config):
        device = config['device']

        model, vocab = build_model(config)
        model.load_state_dict(torch.load(config['weights'], map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device

    def predict(self, img, return_prob=False):
        text, prob = self.predict_batch([img])[0]
        if return_prob:
            return text, prob
        return text

    def predict_batch(self, imgs):
        """Recognizes MULTIPLE crops in ONE encoder+decoder forward pass
        instead of predict()'s one-crop-at-a-time path.

        Each crop keeps its own aspect-ratio-correct resize (same as
        predict()/process_input) and is then right-padded to the widest
        crop in THIS batch -- process_input_batch also returns each crop's
        true (unpadded) sequence length, which translate() uses (via
        pack_padded_sequence + attention masking in Seq2Seq) so padding
        never contaminates a shorter crop's recognized text. Without that,
        naively padding+stacking would give genuinely WRONG results for
        every crop shorter than the batch's longest one, not just
        numerically-off ones -- see seq2seq.py's Encoder/Attention docs.

        Returns a list of (text, prob) tuples, one per input image, in the
        SAME ORDER as `imgs`.
        """
        if not imgs:
            return []

        img, src_lengths = process_input_batch(
            imgs,
            self.config['dataset']['image_height'],
            self.config['dataset']['image_min_width'],
            self.config['dataset']['image_max_width'],
            self.config['cnn']['ss'],
        )
        img = img.to(self.device)
        src_lengths = src_lengths.to(self.device)

        sent, prob = translate(img, self.model, src_lengths=src_lengths)

        results = []
        for i in range(len(imgs)):
            text = self.vocab.decode(sent[i].tolist())
            results.append((text, float(prob[i])))
        return results
