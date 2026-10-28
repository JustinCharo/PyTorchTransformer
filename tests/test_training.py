import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from dataset import Collator
from model import build_transformer
from train import get_ds, run_validation


class TrainingTests(unittest.TestCase):
    def test_masked_tokens_do_not_affect_real_positions(self):
        torch.manual_seed(0)
        model = build_transformer(16, 16, 8, 8, d_model=16, N=1, h=4, dropout=0, d_ff=32)
        model.eval()
        collate = Collator(src_pad_id=0, tgt_pad_id=0)
        batch = collate([
            {'encoder_input': torch.tensor([1, 4, 2]), 'decoder_input': torch.tensor([1, 5]),
             'label': torch.tensor([5, 2]), 'src_text': '', 'tgt_text': ''},
            {'encoder_input': torch.tensor([1, 6, 7, 2]), 'decoder_input': torch.tensor([1, 8, 9]),
             'label': torch.tensor([8, 9, 2]), 'src_text': '', 'tgt_text': ''},
        ])
        with torch.no_grad():
            batched = model(batch['encoder_input'], batch['encoder_mask'],
                            batch['decoder_input'], batch['decoder_mask'])[0, :2]
            single = model(batch['encoder_input'][:1, :3], batch['encoder_mask'][:1, :, :, :3],
                           batch['decoder_input'][:1, :2],
                           batch['decoder_mask'][:1, :, :2, :2])[0]
        torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)

    def test_backward_updates_tied_embedding(self):
        torch.manual_seed(0)
        model = build_transformer(16, 16, 8, 8, d_model=16, N=1, h=4, dropout=0, d_ff=32)
        self.assertIs(model.tgt_embed.embedding.weight, model.projection_layer.proj.weight)
        src = torch.tensor([[1, 4, 2]])
        tgt = torch.tensor([[1, 5]])
        logits = model(src, torch.ones(1, 1, 1, 3, dtype=torch.bool),
                       tgt, torch.ones(1, 1, 2, 2, dtype=torch.bool).tril())
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([5, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.tgt_embed.embedding.weight.grad.abs().sum().item(), 0)

    def test_invalid_model_size(self):
        with self.assertRaisesRegex(ValueError, 'positive even'):
            build_transformer(16, 16, 8, 8, d_model=15)

    def test_tokenizers_fit_only_training_split(self):
        seed = 42
        validation_indices = torch.randperm(20, generator=torch.Generator().manual_seed(seed))[-2:]
        records = [{'translation': {'en': 'trainingword', 'it': 'parolatraining'}} for _ in range(20)]
        for index in validation_indices.tolist():
            records[index] = {'translation': {'en': 'validationword', 'it': 'parolavalidation'}}
        with tempfile.TemporaryDirectory() as folder:
            config = {
                'datasource': 'local', 'lang_src': 'en', 'lang_tgt': 'it',
                'tokenizer_file': str(Path(folder) / 'tokenizer_{0}.json'),
                'seq_len': 8, 'seed': seed, 'batch_size': 4, 'val_batch_size': 2,
                'num_workers': 0,
            }
            with patch('train.load_dataset', return_value=records):
                train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
            self.assertEqual(len(train_loader.dataset), 18)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertIsNone(tokenizer_src.token_to_id('validationword'))
            self.assertIsNone(tokenizer_tgt.token_to_id('parolavalidation'))
            model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                                      8, 8, d_model=16, N=1, h=4, dropout=0, d_ff=32)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'))
            metrics = run_validation(model, val_loader, tokenizer_tgt, 8, torch.device('cpu'),
                                     lambda message: None, 0, None, loss_fn, nullcontext(), num_examples=2)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics))


if __name__ == '__main__':
    unittest.main()
