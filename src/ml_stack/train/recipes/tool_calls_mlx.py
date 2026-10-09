"""MLX adapters for conversation fine-tuning and adapter-only checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml_stack.files import sha256_file, versioned, write_json
from ml_stack.train import lora
from ml_stack.train.checkpoint import CheckpointError
from ml_stack.train.recipes.base import resolve_base
from ml_stack.train.recipes.built import Built
from ml_stack.train.recipes.conversations import conversation_batches, read_conversations, render
from ml_stack.train.step import MLXStep


def local_base(base: str) -> Path:
    """Resolve an installed model directory without downloading weights."""
    path = Path(base).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base, local_files_only=True))


def base_identity(base: Path) -> dict[str, Any]:
    files = sorted([*base.glob('*.safetensors'), base / 'config.json',
                    base / 'tokenizer.json', base / 'tokenizer_config.json'])
    return {'directory': str(base.resolve()),
            'files': {p.name: sha256_file(p) for p in files if p.is_file()}}


class AdapterStep(MLXStep):
    def __init__(self, model: Any, optimizer: Any, loss: Any, *, identity: dict[str, Any]) -> None:
        super().__init__(model, optimizer, loss)
        self.identity = identity

    def validate_checkpoint(self, config: dict[str, Any]) -> None:
        if config.get('mlx_base_identity') != self.identity:
            raise CheckpointError('MLX checkpoint belongs to a different frozen base or tokenizer')

    def __call__(self, batch: Any) -> tuple[float, bool]:
        self.model.train()
        return super().__call__(batch)

    def eval_loss(self, batch: Any) -> float:
        self.model.eval()
        return super().eval_loss(batch)

    def parameters(self) -> dict[str, Any]:
        from mlx.utils import tree_flatten

        return dict(tree_flatten(self.model.trainable_parameters()))

    def restore(self, tensors: dict[str, Any], optimizer: dict[str, Any] | None) -> None:
        from mlx.utils import tree_unflatten
        here = self.parameters()
        if set(here) != set(tensors) or any(here[k].shape != tensors[k].shape for k in here):
            raise CheckpointError('MLX adapter checkpoint does not match its rank and projections')
        self.model.update(tree_unflatten(list(tensors.items())))
        self.mx.eval(self.model.trainable_parameters())
        if optimizer:
            self.opt.state = tree_unflatten(list(optimizer.items()))


def masked_loss(model: Any, batch: dict[str, np.ndarray]) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    ids = mx.array(batch['input_ids'])
    targets = mx.array(batch['labels'][:, 1:])
    valid = targets != -100
    safe = mx.where(valid, targets, 0)
    logits = model(ids[:, :-1])
    loss = nn.losses.cross_entropy(logits, safe)
    return mx.sum(mx.where(valid, loss, 0)) / mx.maximum(mx.sum(valid), 1)


def attach(model: Any, config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    settings = lora.Lora.of(config)
    suffixes = set(settings.targets)
    keys = sorted({name for layer in model.layers for name, module in layer.named_modules()
                   if name.rsplit('.', 1)[-1] in suffixes
                   and hasattr(module, 'weight')})
    unmatched = suffixes - {name.rsplit('.', 1)[-1] for name in keys}
    if unmatched:
        raise ValueError(f'MLX model has no adapter projections: {", ".join(sorted(unmatched))}')
    model.freeze()
    params = {'rank': settings.rank, 'scale': settings.alpha / settings.rank,
              'dropout': settings.dropout, 'keys': keys}
    linear_to_lora_layers(model, len(model.layers), params)
    if not tree_flatten(model.trainable_parameters()):
        raise ValueError('MLX adapter has no trainable parameters')
    return keys, params


def save_adapter(model: Any, config: dict[str, Any], out: Path) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / 'adapters.safetensors'),
                       dict(tree_flatten(model.trainable_parameters())))
    write_json(out / 'adapter_config.json', {
        'model': config['base'], 'fine_tune_type': 'lora',
        'num_layers': config['lora_layers'], 'lora_parameters': config['mlx_lora_parameters'],
        'base_config': config['base_config'], 'base_identity': config['mlx_base_identity'], 'framework': 'mlx'})
    return out


def build(spec: dict[str, Any], config: dict[str, Any], data: Path | None) -> Built:
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import get_total_parameters

    if not config.get('lora'):
        raise ValueError('MLX conversation fine-tuning requires the LoRA adapter strategy')
    if data is None:
        raise ValueError('Conversation fine-tuning requires a dataset')
    train, holdout, manifest = read_conversations(data)
    if not train:
        raise ValueError('No training conversations with messages were found')
    selected, _ = resolve_base(spec, config, manifest)
    base = local_base(selected)
    identity = base_identity(base)
    model, wrapped = load(str(base))
    tokenizer = wrapped._tokenizer
    if not tokenizer.chat_template:
        raise ValueError('The selected MLX base has no chat template')
    context = int(config['context'])
    training = [row for sample in train if
                (row := render(tokenizer, sample['messages'], sample.get('tools'), context=context))]
    evaluation = [row for sample in holdout if
                  (row := render(tokenizer, sample['messages'], sample.get('tools'), context=context))]
    if not training:
        raise ValueError(f'Every assistant answer was cut off at context {context}; raise it')
    keys, params = attach(model, config)
    optimizer = optim.AdamW(learning_rate=float(config['learning_rate']), weight_decay=0.0)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seed, batch = int(config.get('seed') or 0), int(config['batch_size'])
    effective = {**config, 'framework': 'mlx', 'base': str(base), 'base_id': selected,
                 'base_config': json.loads((base / 'config.json').read_text()),
                 'mlx_base_identity': identity,
                 'device': 'mlx', 'recipe': 'tool-calls', 'rows': len(train) + len(holdout),
                 'train_rows': len(training), 'holdout_rows': len(evaluation),
                 'lora_targets_used': keys, 'lora_layers': len(model.layers),
                 'mlx_lora_parameters': params, 'total_parameters': get_total_parameters(model),
                 'trainable_parameters': sum(p.size for _, p in tree_flatten(model.trainable_parameters()))}
    return Built(model=model, optimizer=optimizer, loss=masked_loss,
                 step=AdapterStep(model, optimizer, masked_loss, identity=identity), config=effective,
                 batches=conversation_batches(training, batch_size=batch, pad_id=pad, seed=seed),
                 eval_batches=conversation_batches(evaluation or training, batch_size=batch,
                                                   pad_id=pad, seed=seed + 1))


class Plan(lora.Fit):
    @property
    def resident_gb(self) -> float:
        return sum(p.stat().st_size for p in local_base(self.base).glob('*.safetensors')) / 1e9

    def as_dict(self) -> dict[str, Any]:
        got = super().as_dict()
        got.pop('resident_gb')
        got['cached_base_gb'] = self.resident_gb
        got['total_memory_measured'] = False
        if not self.measured:
            got['seconds_per_step'] = got['estimated_seconds'] = None
        return got

    def lines(self) -> list[str]:
        return [f'MLX LoRA: {self.base}; batch {self.batch}, context {self.context}, {self.steps} steps',
                f'Cached base weights: {self.resident_gb:.2f} GB; adapters, optimizer and activations add memory.',
                f'Measured {self.seconds_per_step:.3f} s/step' if self.measured else
                'Runtime and total memory are unmeasured; run a short training smoke first.']


def plan(config: dict[str, Any], base: str, examples: int, *,
         ceiling_min: float | None = None, seconds_per_step: float = 0.0) -> Plan:
    return Plan(str(local_base(base)), 'mlx', 0, 0, lora.Lora.of(config),
                int(config['batch_size']), int(config['context']), int(config['steps']), examples,
                ceiling_min=lora.CEILING_MIN if ceiling_min is None else ceiling_min,
                seconds_per_step=seconds_per_step, measured=bool(seconds_per_step))


@dataclass(frozen=True)
class Outcome:
    """What the last step left behind, and where the adapter and manifest go."""

    report: Any
    fit: Any
    data: Path
    out: Path
    dry: bool
    version: int


def finish(built: Built, config: dict[str, Any], outcome: Outcome, talk: Any) -> dict[str, Any]:
    report, fit, data, out, dry = outcome.report, outcome.fit, outcome.data, outcome.out, outcome.dry
    measured = report.seconds / report.steps if report.steps else 0.0
    effective = plan(config, built.config['base'], int(built.config['rows']),
                     ceiling_min=fit.ceiling_min, seconds_per_step=measured)
    got = {'settings': lora.Lora.of(config).as_dict(), 'framework': 'mlx',
           'trainable_parameters': built.config['trainable_parameters'],
           'seconds_per_step': round(measured, 3), 'plan': effective.as_dict(), 'adapter': ''}
    talk(f'Measured: {measured:.3f} s/step over {report.steps} steps')
    if dry:
        return got
    adapter = save_adapter(built.model, built.config, out / 'adapter')
    got['adapter'] = str(adapter)
    manifest = {'recipe': 'tool-calls', 'framework': 'mlx', 'base': built.config['base'],
                'base_id': built.config['base_id'], 'base_config': built.config['base_config'],
                'base_identity': built.config['mlx_base_identity'],
                'config': config, 'data': lora.fingerprint(data), 'lora': got,
                'steps': report.steps, 'final_loss': report.final_loss,
                'seconds': round(report.seconds, 1), 'stop_reason': report.stop_reason}
    write_json(out / 'manifest.json', versioned(manifest, outcome.version), default=str)
    talk(f'Adapter: {adapter}')
    return got
