from model import build_transformer
from dataset import BilingualDataset, Collator, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import warnings
from contextlib import nullcontext
from tqdm import tqdm
import shutil
import math
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    # Translates a whole batch at once; source is (B, src_len), source_mask is (B, 1, 1, src_len)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.full((source.size(0), 1), sos_idx, dtype=source.dtype, device=device)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=device)
    while decoder_input.size(1) < max_len:
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token; sentences that already produced </s> just get padding
        next_word = model.project(out[:, -1]).argmax(dim=-1)
        next_word = next_word.masked_fill(finished, pad_idx)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)

        finished |= next_word == eos_idx
        if finished.all():
            break

    return decoder_input

def strip_special_tokens(ids, sos_idx, eos_idx, pad_idx):
    # Drop the leading <s> and cut at the first </s> (or padding)
    out = []
    for id in ids[1:]:
        if id in (eos_idx, pad_idx):
            break
        if id != sos_idx:
            out.append(id)
    return out

def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg, global_step, writer, loss_fn, autocast, num_examples=500, num_print_examples=2):
    model.eval()

    expected = []
    predicted = []
    total_loss = 0.0
    total_tokens = 0

    # get the console window width (falls back to 80 if it can't be determined)
    console_width = shutil.get_terminal_size((80, 20)).columns

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')

    with torch.inference_mode(), autocast:
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device, non_blocking=True) # (B, src_len)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True) # (B, 1, 1, src_len)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True) # (B, tgt_len)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True) # (B, 1, tgt_len, tgt_len)
            label = batch["label"].to(device, non_blocking=True) # (B, tgt_len)

            # Compute validation loss on the whole validation set, weighted by the number of real (non-pad) tokens
            proj_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
            n_tokens = (label != pad_idx).sum().item()
            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)).float(), label.view(-1))
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            # Generate translations for the first num_examples sentences (greedy decoding is the slow part)
            if len(predicted) < num_examples:
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
                for i, ids in enumerate(model_out.tolist()):
                    if len(predicted) >= num_examples:
                        break
                    predicted.append(tokenizer_tgt.decode(strip_special_tokens(ids, sos_idx, eos_idx, pad_idx)))
                    expected.append(batch["tgt_text"][i])

                    # Print the source, target and model output (only for first few examples)
                    if len(predicted) <= num_print_examples:
                        print_msg('-'*console_width)
                        print_msg(f"{f'SOURCE: ':>12}{batch['src_text'][i]}")
                        print_msg(f"{f'TARGET: ':>12}{batch['tgt_text'][i]}")
                        print_msg(f"{f'PREDICTED: ':>12}{predicted[-1]}")
                        if len(predicted) == num_print_examples:
                            print_msg('-'*console_width)

    if total_tokens == 0:
        return None, None, None, None

    avg_val_loss = total_loss / total_tokens

    # Evaluate the character error rate
    cer = torchmetrics.text.CharErrorRate()(predicted, expected).item()

    # Compute the word error rate
    wer = torchmetrics.text.WordErrorRate()(predicted, expected).item()

    # Compute the BLEU metric
    # Format references as list of lists (each reference is a list)
    bleu = torchmetrics.text.BLEUScore()(predicted, [[ref] for ref in expected]).item()

    if writer:
        writer.add_scalar('validation loss', avg_val_loss, global_step)
        writer.add_scalar('validation cer', cer, global_step)
        writer.add_scalar('validation wer', wer, global_step)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

    return avg_val_loss, cer, wer, bleu

def get_or_build_tokenizer(config, ds, lang):
    # Path to the tokenizer file, i.e. '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def filter_long_pairs(ds, tokenizer_src, tokenizer_tgt, config):
    keep = []
    batch_size = 1024
    for start in range(0, len(ds), batch_size):
        pairs = [ds[i]['translation'] for i in range(start, min(start + batch_size, len(ds)))]
        src_lengths = [len(item.ids) for item in tokenizer_src.encode_batch([pair[config['lang_src']] for pair in pairs])]
        tgt_lengths = [len(item.ids) for item in tokenizer_tgt.encode_batch([pair[config['lang_tgt']] for pair in pairs])]
        keep.extend(start + i for i, (src_len, tgt_len) in enumerate(zip(src_lengths, tgt_lengths))
                    if src_len + 2 <= config['seq_len'] and tgt_len + 1 <= config['seq_len'])
    if len(keep) != len(ds):
        print(f'Dropped {len(ds) - len(keep)} sentence pairs longer than seq_len={config["seq_len"]}')
    return Subset(ds, keep)

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Keep 90% for training, 10% for validation
    # Split before fitting tokenizers so validation sentences do not affect the vocabulary.
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    if train_ds_size == 0 or val_ds_size == 0:
        raise ValueError('The dataset needs at least two sentence pairs')
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(config['seed']))

    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])

    train_ds_raw = filter_long_pairs(train_ds_raw, tokenizer_src, tokenizer_tgt, config)
    val_ds_raw = filter_long_pairs(val_ds_raw, tokenizer_src, tokenizer_tgt, config)
    if not train_ds_raw or not val_ds_raw:
        raise ValueError('No sentence pairs remain in the training or validation split; increase seq_len')

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    collate_fn = Collator(tokenizer_src.token_to_id('[PAD]'), tokenizer_tgt.token_to_id('[PAD]'))
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
    )
    train_generator = torch.Generator().manual_seed(config['seed'])
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, generator=train_generator, **loader_kwargs)
    # Don't shuffle validation set for consistent metrics across epochs
    val_dataloader = DataLoader(val_ds, batch_size=config['val_batch_size'], shuffle=False, **loader_kwargs)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'], dropout=config['dropout'])
    return model

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_model(config):
    device = get_device()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        # Allow TF32 matmuls on Ampere+ GPUs (large speedup, negligible accuracy cost)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(config['seed'])

    # Mixed precision: bf16 needs no loss scaling; fp16 does
    use_amp = config['use_amp'] and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    autocast = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Betas from the original paper; fused kernel on CUDA
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9, weight_decay=config['weight_decay'], fused=device.type == 'cuda')

    # Learning rate scheduler: linear warmup for the first warmup_frac of steps, then cosine decay
    total_steps = config['num_epochs'] * len(train_dataloader)
    warmup_steps = max(1, int(config['warmup_frac'] * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # load preloaded model if specified
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'scaler_state_dict' in state:
            scaler.load_state_dict(state['scaler_state_dict'])
        if 'train_generator_state' in state:
            train_dataloader.generator.set_state(state['train_generator_state'])
        if 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    else:
        print('No model to preload, starting from scratch')

    # Compiled module is used for the forward pass; checkpoints are saved from the original model so they load either way
    train_forward = torch.compile(model) if config['compile'] else model

    # Use tokenizer_tgt for PAD token since loss is computed on target tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=config['label_smoothing']).to(device)
    # Validation loss without label smoothing, so it's the true cross entropy (exp(loss) = perplexity)
    val_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device, non_blocking=True) # (B, src_len)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True) # (B, tgt_len)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True) # (B, 1, 1, src_len)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True) # (B, 1, tgt_len, tgt_len)
            label = batch['label'].to(device, non_blocking=True) # (B, tgt_len)

            # Run the tensors through the encoder, decoder and the projection layer
            with autocast:
                proj_output = train_forward(encoder_input, encoder_mask, decoder_input, decoder_mask) # (B, tgt_len, vocab_size)

            # Compute the loss using cross entropy (in fp32 for numerical stability)
            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)).float(), label.view(-1))

            # Backpropagate the loss
            scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients (unscale first so the threshold applies to the real gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler skips the optimizer update when fp16 gradients overflow.
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
                global_step += 1
            optimizer.zero_grad(set_to_none=True)

            # Log the loss (only every log_every steps, since .item() waits for the GPU)
            if global_step % config['log_every'] == 0:
                loss_value = loss.item()
                batch_iterator.set_postfix({"loss": f"{loss_value:6.3f}"})
                writer.add_scalar('train loss', loss_value, global_step)
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)

        # Run validation at the end of every epoch
        val_loss, cer, wer, bleu = run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, val_loss_fn, autocast, num_examples=config['num_val_examples'], num_print_examples=config['num_print_examples'])

        if val_loss is not None:
            batch_iterator.write(f"Epoch {epoch:02d} - Val Loss: {val_loss:.4f} (ppl {math.exp(val_loss):.1f}), CER: {cer * 100:.2f}%, WER: {wer * 100:.2f}%, BLEU: {bleu:.4f}")

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'global_step': global_step,
            'train_generator_state': train_dataloader.generator.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
        }, model_filename)

    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
