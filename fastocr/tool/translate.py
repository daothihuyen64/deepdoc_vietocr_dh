import torch
import numpy as np
import cv2
from fastocr.model.vocab import Vocab
from fastocr.model.transformerocr import FastOCR
import math
from PIL import Image


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2, src_lengths=None):
    """data: BxCxHxW

    src_lengths: batch_size -- each image's true (unpadded) CNN-output
        sequence length, only needed when `img` batches MULTIPLE
        variable-width images padded to a common width (see
        process_input_batch). None (default) matches the original
        single-image/unpadded-batch behavior exactly.

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
        memory = model.transformer.forward_encoder(src, src_lengths)

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
        # towards confidence -- only real decoded characters BEFORE each
        # sequence's own first EOS do. The decode loop above only stops
        # once EVERY sequence in the batch has emitted EOS, so with
        # batch_size > 1 a sequence that finishes early keeps getting fed
        # its own EOS as the next input and can keep emitting further
        # tokens purely as continuation noise -- occasionally a real
        # character id, which `> 3` alone would NOT filter out, silently
        # dragging that sequence's confidence down. Explicitly excluding
        # everything at/after each row's own first EOS (not just by token
        # identity) fixes this; for batch_size == 1 the loop already stops
        # the instant that one sequence hits EOS, so no such trailing
        # tokens ever exist there and this is a no-op (byte-identical to
        # before) for single-image calls.
        is_eos = translated_sentence == eos_token
        before_own_eos = np.cumsum(is_eos, axis=1) == 0
        valid_mask = (translated_sentence > 3) & before_own_eos
        log_probs = np.log(np.clip(char_probs, 1e-12, 1.0)) * valid_mask
        valid_counts = np.maximum(valid_mask.sum(axis=-1), 1)
        char_probs = np.exp(log_probs.sum(axis=-1) / valid_counts)

    return translated_sentence, char_probs


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = FastOCR(len(vocab),
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


def _cnn_output_length(pixel_width, pixel_height, cnn_ss):
    """Replicates the CNN backbone's pooling stages (kernel == stride at
    every stage -- non-overlapping pooling, see fastocr/model/backbone/vgg.py
    and the vgg-seq2seq.yml config's cnn.ss/cnn.ks, which are always equal)
    to compute the EXACT sequence length a single image of this pixel size
    produces, with NO padding involved. `cnn_ss` is config['cnn']['ss'], a
    list of [height_stride, width_stride] pairs, one per pooling stage.

    This is what lets process_input_batch() tell the encoder/attention
    exactly where each image's real content ends and its padding begins,
    once multiple different-width images are stacked into one tensor.
    """
    w, h = pixel_width, pixel_height
    for stride_h, stride_w in cnn_ss:
        h = h // stride_h
        w = w // stride_w
    return max(w * h, 1)


def process_input_batch(images, image_height, image_min_width, image_max_width, cnn_ss):
    """Batched counterpart of process_input() -- resizes EACH image to its
    own aspect-ratio-correct width exactly like process_input() does, then
    right-pads every one (in pixel space, with zeros) out to the widest
    image in this particular batch, so they can be stacked into one
    (B, C, H, W) tensor for a single CNN + encoder forward pass.

    Returns (batched_tensor, src_lengths): src_lengths[i] is image i's own
    true (unpadded) CNN-output sequence length (see _cnn_output_length),
    computed from its OWN resized width -- pass this straight through to
    translate()'s src_lengths so the encoder/attention ignore the padded
    tail instead of treating it as real image content.
    """
    resized = [
        process_image(img, image_height, image_min_width, image_max_width)
        for img in images
    ]
    widths = [arr.shape[2] for arr in resized]
    max_w = max(widths)
    channels = resized[0].shape[0]

    batched = np.zeros((len(resized), channels, image_height, max_w), dtype=np.float32)
    for i, arr in enumerate(resized):
        batched[i, :, :, : arr.shape[2]] = arr

    src_lengths = torch.LongTensor(
        [_cnn_output_length(w, image_height, cnn_ss) for w in widths]
    )

    return torch.FloatTensor(batched), src_lengths



