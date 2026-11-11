# PyTorch Transformer

An encoder-decoder Transformer for English-to-Italian translation with the OPUS Books dataset.

## Train

Use Python 3.9 or newer and install the dependencies:

```sh
python -m pip install -r requirements.txt
python train.py
```

Edit `config.py` to change the model size, batch size, training schedule, or device features. Training writes checkpoints to `opus_books_weights/` and TensorBoard logs to `runs/tmodel/`. Set `preload` to `None` to start a new run; `latest` resumes the newest checkpoint. Only the newest `keep_last` checkpoints are kept on disk.

The training and validation split is seeded. New tokenizers are fitted on the training split only. Existing tokenizer files are reused so that checkpoints retain their original vocabulary. When starting a completely new experiment, remove the old tokenizer files along with its checkpoints.

BLEU, WER and CER are computed after splitting the reference sentences the same way the tokenizer does, so punctuation spacing in the model output is not counted as an error.

## Translate

Once a checkpoint exists, translate sentences with the latest one:

```sh
python translate.py "I am not sure what you mean." "Where is the station?"
```

Pass `--epoch 07` to use a specific checkpoint, or pipe sentences in on stdin, one per line.

## Tests

Run the focused tests with:

```sh
python -m unittest discover -s tests -v
```
