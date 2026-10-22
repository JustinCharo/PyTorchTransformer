import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Special tokens for the source come from the source tokenizer, and for the target from the target tokenizer
        self.src_sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.src_eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Make sure the sentence fits (we add <s> and </s> to the source, and one of them to the target)
        # get_ds filters out pairs that are too long, so this should never trigger
        if len(enc_input_tokens) + 2 > self.seq_len or len(dec_input_tokens) + 1 > self.seq_len:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat([self.src_sos_token, torch.tensor(enc_input_tokens, dtype=torch.int64), self.src_eos_token])

        # Add only <s> token
        decoder_input = torch.cat([self.sos_token, torch.tensor(dec_input_tokens, dtype=torch.int64)])

        # Add only </s> token
        label = torch.cat([torch.tensor(dec_input_tokens, dtype=torch.int64), self.eos_token])

        # Padding and masks are added per batch in Collator, so each batch is only as long as its longest sentence
        return {
            "encoder_input": encoder_input,  # (src_len)
            "decoder_input": decoder_input,  # (tgt_len)
            "label": label,  # (tgt_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

class Collator:
    # Pads a batch to its longest sentence and builds boolean attention masks (True = attend)

    def __init__(self, src_pad_id: int, tgt_pad_id: int):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch):
        encoder_input = torch.nn.utils.rnn.pad_sequence([b["encoder_input"] for b in batch], batch_first=True, padding_value=self.src_pad_id)
        decoder_input = torch.nn.utils.rnn.pad_sequence([b["decoder_input"] for b in batch], batch_first=True, padding_value=self.tgt_pad_id)
        label = torch.nn.utils.rnn.pad_sequence([b["label"] for b in batch], batch_first=True, padding_value=self.tgt_pad_id)

        return {
            "encoder_input": encoder_input,  # (B, src_len)
            "decoder_input": decoder_input,  # (B, tgt_len)
            "encoder_mask": (encoder_input != self.src_pad_id).unsqueeze(1).unsqueeze(1),  # (B, 1, 1, src_len)
            "decoder_mask": (decoder_input != self.tgt_pad_id).unsqueeze(1).unsqueeze(1) & causal_mask(decoder_input.size(1)),  # (B, 1, 1, tgt_len) & (1, tgt_len, tgt_len)
            "label": label,  # (B, tgt_len)
            "src_text": [b["src_text"] for b in batch],
            "tgt_text": [b["tgt_text"] for b in batch],
        }

def causal_mask(size):
    # (1, size, size) boolean, True where a position may attend (itself and earlier positions)
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool))
