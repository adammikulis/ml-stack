"""Conversation adapters train, cancel and save without copying frozen MLX bases."""

import json

import numpy as np
import pytest


@pytest.fixture
def cpu():
    pytest.importorskip('mlx.core', reason='ml-stack[train-mlx]')
    pytest.importorskip('mlx_lm', reason='ml-stack[train-mlx]')
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(previous)


@pytest.fixture
def base(tmp_path, cpu):
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    path = tmp_path / 'invented-llama'
    path.mkdir()
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, 'user': 3, 'assistant': 4,
             'red': 5, 'blue': 6, 'paint': 7, ':': 8}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>',
                                  unk_token='<unk>', eos_token='<eos>')
    fast.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    fast.save_pretrained(path)
    config = {'model_type': 'llama', 'hidden_size': 32, 'num_hidden_layers': 1,
              'intermediate_size': 64, 'num_attention_heads': 2, 'num_key_value_heads': 2,
              'rms_norm_eps': 1e-5, 'vocab_size': len(vocab)}
    model = Model(ModelArgs.from_dict(config))
    cpu.eval(model.parameters())
    cpu.save_safetensors(str(path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    (path / 'config.json').write_text(json.dumps(config))
    return path


@pytest.fixture
def data(tmp_path):
    root = tmp_path / 'conversations'
    root.mkdir()
    rows = [{'messages': [{'role': 'user', 'content': 'paint red'},
                          {'role': 'assistant', 'content': 'blue'}]} for _ in range(4)]
    for name in ['train', 'holdout']:
        (root / f'{name}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return root


def settings(base):
    return {'framework': 'mlx', 'base': str(base), 'lora': True, 'lora_rank': 2,
            'lora_alpha': 4, 'lora_targets': 'q_proj,v_proj', 'lora_dropout': 0,
            'steps': 20, 'batch_size': 1, 'context': 64, 'learning_rate': .001}


def test_assistant_only_loss_ignores_prompt_targets(base, data, cpu):
    from ml_stack.train.recipes import build
    from ml_stack.train.recipes.tool_calls_mlx import masked_loss

    built = build('tool-calls', settings(base), data)
    batch = built.batches(0)
    masked = batch['labels'] == -100
    assert masked.any() and (~masked).any()
    logits = np.array(built.model(cpu.array(batch['input_ids'][:, :-1])))
    assert batch['labels'][0, -1] == 6 and np.sum(~masked) == 1
    scores = logits[0, -1]
    expected = np.log(np.exp(scores - np.max(scores)).sum()) + np.max(scores) - scores[6]
    assert float(masked_loss(built.model, batch)) == pytest.approx(float(expected), rel=1e-5)


def test_trains_saves_adapter_and_reloads_without_frozen_weights(base, data, cpu, tmp_path):
    from mlx_lm import load

    from ml_stack.train.run import run

    updates = []
    out = tmp_path / 'run'
    result = run('tool-calls', settings(base), data, out, on_step=lambda *a: updates.append(a))
    assert result['steps'] == 20 and result['stop_reason'] == 'completed'
    assert updates and np.isfinite(result['final_loss'])
    weights = cpu.load(str(out / 'adapter/adapters.safetensors'))
    assert weights and all(key.endswith(('lora_a', 'lora_b')) for key in weights)
    assert any(np.any(np.array(value) != 0) for key, value in weights.items() if key.endswith('lora_b'))
    saved = json.loads((out / 'manifest.json').read_text())
    assert saved['framework'] == 'mlx' and saved['base'] == str(base)
    restored, _ = load(str(base), adapter_path=str(out / 'adapter'))
    cpu.eval(restored.parameters())
    checkpoint = cpu.load(str(out / 'latest/model.safetensors'))
    assert checkpoint.keys() == weights.keys()
    assert sum(v.nbytes for v in checkpoint.values()) < (base / 'model.safetensors').stat().st_size


def test_cancellation_records_partial_adapter(base, data, cpu, tmp_path):
    from ml_stack.train.run import run

    progress = []
    result = run('tool-calls', settings(base), data, tmp_path / 'cancelled',
                 on_step=lambda *a: progress.append(a), should_stop=lambda: len(progress) >= 3)
    assert result['steps'] == 3 and result['stop_reason'] == 'stopped'
    assert result['lora']['adapter']


def test_projection_mismatch_fails_before_training(base, data, cpu):
    from ml_stack.train.recipes import build

    with pytest.raises(ValueError, match='no adapter projections'):
        build('tool-calls', {**settings(base), 'lora_targets': 'missing_projection'}, data)


def test_resume_refuses_different_same_shape_frozen_base(base, data, cpu, tmp_path):
    import shutil

    from ml_stack.train.checkpoint import CheckpointError
    from ml_stack.train.run import run

    out = tmp_path / 'same-output'
    run('tool-calls', settings(base), data, out)
    other = tmp_path / 'different-base'
    shutil.copytree(base, other)
    tensors = cpu.load(str(other / 'model.safetensors'))
    key = next(iter(tensors))
    tensors[key] = tensors[key] + .25
    cpu.eval(tensors)
    cpu.save_safetensors(str(other / 'model.safetensors'), tensors)
    with pytest.raises(CheckpointError, match='different frozen base'):
        run('tool-calls', settings(other), data, out)


def test_resume_refuses_replaced_weights_at_same_base_path(base, data, cpu, tmp_path):
    from ml_stack.train.checkpoint import CheckpointError
    from ml_stack.train.run import run

    out = tmp_path / 'same-path-output'
    run('tool-calls', settings(base), data, out)
    tensors = cpu.load(str(base / 'model.safetensors'))
    key = next(iter(tensors))
    tensors[key] = tensors[key] + .25
    cpu.eval(tensors)
    cpu.save_safetensors(str(base / 'model.safetensors'), tensors)
    with pytest.raises(CheckpointError, match='different frozen base'):
        run('tool-calls', settings(base), data, out)


def test_unmeasured_mlx_plan_exposes_unknown_cost(base, data, cpu):
    from ml_stack.train.recipes import validate
    from ml_stack.train.run import plan_for

    got = plan_for('tool-calls', validate('tool-calls', settings(base)), data).as_dict()
    assert got['seconds_per_step'] is None and got['estimated_seconds'] is None
    assert got['cached_base_gb'] > 0 and got['total_memory_measured'] is False


def test_quantized_base_trains_adapter_without_changing_frozen_weights(base, data, cpu, tmp_path):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.utils import get_total_parameters

    from ml_stack.files import sha256_file
    from ml_stack.train.run import run

    model, _ = load(str(base))
    nn.quantize(model, group_size=32, bits=4)
    cpu.eval(model.parameters())
    cpu.save_safetensors(str(base / 'quantized.safetensors'), dict(tree_flatten(model.parameters())))
    (base / 'model.safetensors').unlink()
    (base / 'quantized.safetensors').rename(base / 'model.safetensors')
    config = json.loads((base / 'config.json').read_text())
    config['quantization'] = {'group_size': 32, 'bits': 4}
    (base / 'config.json').write_text(json.dumps(config))
    before = sha256_file(base / 'model.safetensors')
    result = run('tool-calls', settings(base), data, tmp_path / 'quantized-adapter')
    assert result['steps'] == 20 and np.isfinite(result['final_loss'])
    assert result['parameters'] >= get_total_parameters(model)
    assert sha256_file(base / 'model.safetensors') == before


def test_recipe_size_backend_default_preserves_explicit_choice():
    from ml_stack.train.recipes import validate

    assert validate('tool-calls', {'size': 'qwen27b'})['framework'] == 'mlx'
    assert validate('tool-calls', {'size': 'qwen27b', 'framework': 'torch'})['framework'] == 'torch'


def test_same_base_resumes_adapter_and_optimizer(base, data, cpu, tmp_path):
    from ml_stack.train.run import run

    out = tmp_path / 'resume'
    first = run('tool-calls', settings(base), data, out)
    resumed = run('tool-calls', {**settings(base), 'steps': 40}, data, out)
    assert first['steps'] == 20 and resumed['steps'] == 40
    assert resumed['stop_reason'] == 'completed' and np.isfinite(resumed['final_loss'])


def test_training_capability_does_not_redefine_existing_mlx_installation():
    from ml_stack.fleet.environment import CATALOG

    existing = next(lib for lib in CATALOG if lib.name == 'mlx')
    training = next(lib for lib in CATALOG if lib.name == 'train-mlx')
    assert existing.packages == ('mlx>=0.18',)
    assert training.packages == ('ml-stack[train-mlx]',)
    assert training.platforms == ('darwin',) and training.vendors == ('apple',)
    assert training.default is False
