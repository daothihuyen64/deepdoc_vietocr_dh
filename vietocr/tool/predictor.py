import torch

from vietocr.tool.translate import build_model, process_input, translate


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
        img = process_input(
            img,
            self.config['dataset']['image_height'],
            self.config['dataset']['image_min_width'],
            self.config['dataset']['image_max_width'],
        )
        img = img.to(self.device)

        sent, prob = translate(img, self.model)
        text = self.vocab.decode(sent[0].tolist())

        if return_prob:
            return text, float(prob[0])
        return text
