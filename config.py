from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 10**-4,
        "weight_decay": 0.0,
        "warmup_frac": 0.1, # fraction of total steps spent warming up the learning rate
        "label_smoothing": 0.1,
        "dropout": 0.1,
        "seq_len": 350, # sentence pairs longer than this (after special tokens) are dropped
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "keep_last": 3, # number of epoch checkpoints to keep on disk (None keeps all of them)
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "seed": 42, # keeps the train/validation split identical across runs and resumes
        "num_workers": 2,
        "use_amp": True, # mixed precision on CUDA (bf16 if supported, otherwise fp16)
        "compile": False, # torch.compile the model (faster on recent GPUs, slow first steps)
        "log_every": 50, # steps between loss logs; logging every step forces a GPU sync
        "val_batch_size": 64,
        "num_val_examples": 500, # sentences translated each epoch for BLEU / CER / WER
        "num_print_examples": 2, # translations printed to the console each epoch
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    # Sort by epoch number rather than as strings, so epoch 100 comes after epoch 99
    weights_files.sort(key=lambda p: int(p.stem[len(config['model_basename']):]))
    return str(weights_files[-1])
