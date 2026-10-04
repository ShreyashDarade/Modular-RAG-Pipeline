"""Tiny randomly initialised BERT models saved to disk, so the local-model code paths (loading,
batching, pooling, errors) run for real without downloading weights. They discriminate nothing -
quality is checked with real weights in tests/integration/test_local_models.py."""

from __future__ import annotations

from pathlib import Path

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    *"revenue growth cloud paris france capital tower bread".split(),
]


def save_tiny_bert(directory: Path, *, labels: int | None, hidden: int = 16) -> Path:
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    backend = Tokenizer(models.WordPiece({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", VOCAB.index("[CLS]")), ("[SEP]", VOCAB.index("[SEP]"))],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    kwargs = dict(
        vocab_size=len(VOCAB),
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = (
        BertModel(BertConfig(**kwargs))
        if labels is None
        else BertForSequenceClassification(BertConfig(num_labels=labels, **kwargs))
    )
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
