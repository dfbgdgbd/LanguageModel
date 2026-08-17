"""Decoder-only transformer defined and trained from random initialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_CHECKPOINT_PATH, DEFAULT_TOKENIZER_PATH


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1


@dataclass
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.75
    top_k: int = 40
    repetition_penalty: float = 1.18
    no_repeat_ngram_size: int = 4


def build_model(config: TransformerConfig):
    import torch
    from torch import nn

    class DecoderOnlyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                layer, config.n_layers, enable_nested_tensor=False
            )
            self.final_norm = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, input_ids, attention_mask=None):
            _, sequence_length = input_ids.shape
            if sequence_length > config.max_seq_len:
                raise ValueError("sequence exceeds configured context length")
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                diagonal=1,
            )
            padding_mask = attention_mask == 0 if attention_mask is not None else None
            hidden = self.blocks(
                hidden,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
            return self.lm_head(self.final_norm(hidden))

    return DecoderOnlyTransformer()


class ScratchTokenizer:
    def __init__(self, path: str | Path = DEFAULT_TOKENIZER_PATH) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))
        self.pad_id = self.token_to_id("<pad>")
        self.unk_id = self.token_to_id("<unk>")
        self.bos_id = self.token_to_id("<bos>")
        self.eos_id = self.token_to_id("<eos>")
        self.control_ids = {
            self.pad_id,
            self.unk_id,
            self.bos_id,
            self.token_to_id("<system>"),
            self.token_to_id("<user>"),
            self.token_to_id("<assistant>"),
        }

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"tokenizer is missing special token {token!r}")
        return int(value)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


class ScratchTransformerBackend:
    """Inference for the project's own from-scratch transformer checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        settings: GenerationSettings | None = None,
        seed: int = 42,
    ) -> None:
        import torch

        self.torch = torch
        self.tokenizer = ScratchTokenizer(tokenizer_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        self.config = TransformerConfig(**payload["config"])
        if self.config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")
        self.model = build_model(self.config)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.training_state = payload.get("training_state", {})
        self.settings = settings or GenerationSettings()
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def generate(self, prompt: str) -> str:
        torch = self.torch
        reserve = min(self.settings.max_new_tokens, self.config.max_seq_len // 2)
        token_ids = self.tokenizer.encode(prompt)[-(self.config.max_seq_len - reserve) :]
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]
        generated: list[int] = []
        with torch.inference_mode():
            for _ in range(self.settings.max_new_tokens):
                context = token_ids[-self.config.max_seq_len :]
                inputs = torch.tensor([context], dtype=torch.long)
                logits = self.model(inputs)[0, -1].float()
                logits[list(self.tokenizer.control_ids)] = float("-inf")
                for token_id in set(generated[-64:]):
                    if logits[token_id] > 0:
                        logits[token_id] /= self.settings.repetition_penalty
                    else:
                        logits[token_id] *= self.settings.repetition_penalty
                ngram_size = self.settings.no_repeat_ngram_size
                if ngram_size > 1 and len(generated) >= ngram_size - 1:
                    prefix = tuple(generated[-(ngram_size - 1) :])
                    for index in range(len(generated) - ngram_size + 1):
                        if tuple(generated[index : index + ngram_size - 1]) == prefix:
                            logits[generated[index + ngram_size - 1]] = float("-inf")
                if self.settings.temperature <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    logits /= self.settings.temperature
                    top_k = min(self.settings.top_k, logits.numel())
                    values, indices = torch.topk(logits, top_k)
                    probabilities = torch.softmax(values, dim=-1)
                    sampled = torch.multinomial(
                        probabilities, 1, generator=self.generator
                    )
                    next_id = int(indices[sampled].item())
                if next_id == self.tokenizer.eos_id:
                    break
                token_ids.append(next_id)
                generated.append(next_id)
        return self.tokenizer.decode(generated)


def save_checkpoint(
    path: str | Path,
    model,
    config: TransformerConfig,
    training_state: dict[str, Any],
) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "training_state": training_state,
        },
        destination,
    )
