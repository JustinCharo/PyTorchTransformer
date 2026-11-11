import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config, get_weights_file_path, latest_weights_file_path
from train import get_model, get_device, greedy_decode, strip_special_tokens


def load_translator(config, epoch=None, device=None):
    device = device or get_device()

    tokenizer_paths = [Path(config['tokenizer_file'].format(lang)) for lang in (config['lang_src'], config['lang_tgt'])]
    for path in tokenizer_paths:
        if not path.exists():
            raise FileNotFoundError(f'{path} not found, train a model first')
    tokenizer_src, tokenizer_tgt = (Tokenizer.from_file(str(path)) for path in tokenizer_paths)

    model_filename = get_weights_file_path(config, epoch) if epoch else latest_weights_file_path(config)
    if not model_filename or not Path(model_filename).exists():
        raise FileNotFoundError('No checkpoint found, train a model first')

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tokenizer_src, tokenizer_tgt, device


def translate(sentences, model, tokenizer_src, tokenizer_tgt, seq_len, device):
    src_pad = tokenizer_src.token_to_id('[PAD]')
    sos = tokenizer_src.token_to_id('[SOS]')
    eos = tokenizer_src.token_to_id('[EOS]')

    # Truncate so <s> and </s> still fit within seq_len
    encoded = [[sos] + enc.ids[:seq_len - 2] + [eos] for enc in tokenizer_src.encode_batch(sentences)]
    source = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.int64) for ids in encoded],
                                             batch_first=True, padding_value=src_pad).to(device)
    source_mask = (source != src_pad).unsqueeze(1).unsqueeze(1)

    with torch.inference_mode():
        output = greedy_decode(model, source, source_mask, tokenizer_tgt, seq_len, device)

    tgt_ids = [tokenizer_tgt.token_to_id(t) for t in ('[SOS]', '[EOS]', '[PAD]')]
    return [tokenizer_tgt.decode(strip_special_tokens(ids, *tgt_ids)) for ids in output.tolist()]


def main():
    parser = argparse.ArgumentParser(description='Translate sentences with a trained checkpoint')
    parser.add_argument('sentences', nargs='*', help='sentences to translate (reads stdin, one per line, if omitted)')
    parser.add_argument('--epoch', help='checkpoint epoch to load, e.g. 07 (defaults to the latest)')
    args = parser.parse_args()

    config = get_config()
    sentences = args.sentences or [line.strip() for line in sys.stdin if line.strip()]
    if not sentences:
        parser.error('no sentences given')

    model, tokenizer_src, tokenizer_tgt, device = load_translator(config, args.epoch)
    for translation in translate(sentences, model, tokenizer_src, tokenizer_tgt, config['seq_len'], device):
        print(translation)


if __name__ == '__main__':
    main()
