import torch
import numpy as np
import cv2
from vietocr.model.vocab import Vocab
from vietocr.model.transformerocr import VietOCR
import math
from PIL import Image


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW

    Returns:
        (translated_sentence, char_probs) -- translated_sentence: token ids
        per sequence, same as before. char_probs[i]: confidence for sequence
        i -- geometric mean of the softmax probability of each emitted
        non-special token, exp(mean(log P(char_i))).
    """
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1.0] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = torch.softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 1)
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            values = values[:, -1, 0].tolist()

            translated_sentence.append(indices)
            char_probs.append(values)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        char_probs = np.asarray(char_probs).T

        # Special tokens (pad=0/sos=1/eos=2/mask=3, see Vocab) don't count
        # towards confidence -- only real decoded characters do, matching
        # what Vocab.decode() itself keeps.
        valid_mask = translated_sentence > 3
        log_probs = np.log(np.clip(char_probs, 1e-12, 1.0)) * valid_mask
        valid_counts = np.maximum(valid_mask.sum(axis=-1), 1)
        char_probs = np.exp(log_probs.sum(axis=-1) / valid_counts)

    return translated_sentence, char_probs


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img



