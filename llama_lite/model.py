#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model.py

This file contains the definition of a LLaMA transformer model.
Mostly ported from [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
and [facebookresearch/llama](https://github.com/facebookresearch/llama)

Example usage:
```python
    ckpt = '../_models/stories15M.pt'
    model = get_model_from_ckpt(ckpt)
    model.summary()
    res = model.generate("Once upon a time ", max_new_tokens=50, top_k=40, seed=1234)
    print(res)
```
"""

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2023, "

import math
import os
from pathlib import Path

# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "jax"

# Force cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras_core as keras
import keras_core.ops as ops
from dataclasses import dataclass
from typing import Optional, Tuple

from llama_lite.tokenizer import Tokenizer


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10_000.0, dtype: str = 'float32') -> Tuple[
    keras.KerasTensor, keras.KerasTensor]:
    """
    Precomputes the cos, sin frequencies for positional embeddings

    :param dim: Dimension of the freq tensor
    :param max_seq_len: max sequence length
    :param theta: Base of the freqs
    :param dtype: type of the tensor
    :return: Tuple of (cos, sin) freqs
    """
    idx = ops.arange(0, dim, 2, dtype=dtype)[:dim // 2] / dim
    freqs = 1.0 / (theta ** (idx))
    t = ops.arange(max_seq_len, dtype=dtype)
    freqs = ops.outer(t, freqs)
    freqs_cos = ops.cos(freqs)
    freqs_sin = ops.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freq: keras.KerasTensor, x: keras.KerasTensor) -> keras.KerasTensor:
    """
     Helper function to reshape the freq tensor

    :param freq: frequency tensor
    :param x: tensor target
    :return: the reshaped freq tensor
    """
    ndim = len(x.shape)
    assert 0 <= 1 < ndim
    assert freq.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return ops.reshape(freq, shape)


def apply_rotary_emb(xq: keras.KerasTensor, xk: keras.KerasTensor, freqs_cos: keras.KerasTensor,
                     freqs_sin: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Apply rotary embeddings to q,k tensors
    :param xq: the query tensor
    :param xk: the key tensor
    :param freqs_cos: cos freq
    :param freqs_sin: sin freq
    :return: Tuple of the new (xq,xk) tensors with freqs applied
    """
    xq = ops.reshape(xq, new_shape=xq.shape[:-1] + (xq.shape[-1] // 2, 2))
    xk = ops.reshape(xk, xk.shape[:-1] + (xk.shape[-1] // 2, 2))
    xq_r, xq_i = ops.unstack(xq, 2, axis=-1)
    xk_r, xk_i = ops.unstack(xk, 2, axis=-1)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = ops.stack([xq_out_r, xq_out_i], axis=-1)
    new_shape = tuple(xq_out.shape[:-2]) + (-1,)
    xq_out = ops.reshape(xq_out, new_shape)
    xk_out = ops.stack([xk_out_r, xk_out_i], axis=-1)
    xk_out = ops.reshape(xk_out, new_shape)
    return ops.cast(xq_out, xq.dtype), ops.cast(xk_out, xk.dtype)


def repeat_kv(x: keras.KerasTensor, n_rep: int) -> keras.KerasTensor:
    """
     Repeat the x tensor `n_rep` times
    :param x: input tensor
    :param n_rep: number of repetitions
    :return: The modified x tensor
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep > 1:
        x = x[..., None, :]
        x = ops.repeat(x, n_rep, 3)
        x = ops.reshape(x, (bs, slen, n_kv_heads * n_rep, head_dim))
    return x


class Attention(keras.Model):
    """
    Multi-head attention
    """

    def __init__(self, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = keras.layers.Dense(args.n_heads * self.head_dim, use_bias=False, name='wq')
        self.wk = keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name='wk')
        self.wv = keras.layers.Dense(self.n_kv_heads * self.head_dim, use_bias=False, name='wv')
        self.wo = keras.layers.Dense(args.dim, use_bias=False, name='wo')
        self.attn_dropout = keras.layers.Dropout(args.dropout)
        self.resid_dropout = keras.layers.Dropout(args.dropout)
        self.dropout = args.dropout

        # flash attn not supported on all backends
        mask = ops.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"), dtype=self.dtype)
        self.mask = ops.triu(mask, k=1)

    def build(self, input_shape):
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        self.wo.build(input_shape)
        self.built = True

    def call(self, x, freqs_cos, freqs_sin, batch_size=1, training=False, **kwargs):
        bsz, seqlen, _ = x.shape
        if bsz is None:
            bsz = batch_size
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = ops.reshape(xq, new_shape=(bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = ops.reshape(xk, new_shape=(bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        xv = ops.reshape(xv, new_shape=(bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # heads into batch dim
        xq = ops.transpose(xq, (0, 2, 1, 3))
        xk = ops.transpose(xk, (0, 2, 1, 3))
        xv = ops.transpose(xv, (0, 2, 1, 3))

        # attention
        scores = ops.matmul(xq, ops.transpose(xk, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        scores += self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = ops.cast(scores, 'float32')
        scores = ops.softmax(scores, axis=-1)
        scores = ops.cast(scores, xq.dtype)
        scores = self.attn_dropout(scores)
        output = ops.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dim and concat heads
        output = ops.transpose(output, (0, 2, 1, 3))
        output = ops.reshape(output, new_shape=(bsz, seqlen, -1))

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


class FeedForward(keras.Model):
    """
    Feed Forward MLP
    """

    def __init__(self, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        if args.hidden_dim is None:
            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
            args.hidden_dim = hidden_dim
        self.w1 = keras.layers.Dense(args.hidden_dim, use_bias=False, name='w1')
        self.w2 = keras.layers.Dense(args.dim, use_bias=False, name='w2')
        self.w3 = keras.layers.Dense(args.hidden_dim, use_bias=False, name='w3')
        self.dropout = keras.layers.Dropout(args.dropout)

    def build(self, input_shape):
        self.w1.build(input_shape)
        self.w3.build(input_shape)
        self.w2.build(self.w1.compute_output_shape(input_shape))
        self.built = True

    def call(self, x, training=False):
        return self.dropout(self.w2(ops.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(keras.layers.Layer):
    """
    RMS norm layer
    """
    def __init__(self, dim: int, eps: float, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.dim,),
            initializer=keras.initializers.Ones(),
            trainable=True,
            dtype=self.dtype,
            name='rmsnorm_weight'
        )

        super().build(input_shape)

    def _norm(self, x):
        return x * ops.reciprocal(ops.sqrt(ops.mean(ops.power(x, 2), axis=-1, keepdims=True) + self.eps))

    def call(self, x):
        output = self._norm(ops.cast(x, 'float32'))
        output = ops.cast(output, x.dtype)
        return output * self.weight


class TransformerBlock(keras.Model):
    """
    Transformer block
    """

    def __init__(self, layer_id: int, args: ModelArgs, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def build(self, input_shape):
        self.attention.build(input_shape)
        self.feed_forward.build(input_shape)
        self.attention_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.built = True

    def call(self, x, freqs_cos, freqs_sin, batch_size=1, training=False, **kwargs):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, batch_size=batch_size)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLaMATransformer(keras.Model):
    """LLaMA transformer Model """

    def __init__(self, params: ModelArgs, tokenizer_model: str = None, jit_generate=False, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = keras.layers.Embedding(params.vocab_size, params.dim)
        self.dropout = keras.layers.Dropout(params.dropout)
        self.t_layers = [TransformerBlock(layer_id, params, name=f'layer.{layer_id}') for layer_id in
                         range(params.n_layers)]
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.embed_out = keras.layers.Dense(params.vocab_size, use_bias=False, name='output')

        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

        if tokenizer_model is not None:
            self.tokenizer = Tokenizer(tokenizer_model)
        else:
            self.tokenizer = Tokenizer(str(Path(__file__).parent.resolve() / "tokenizer.model"))

        self.jit_generate = jit_generate
        # workaround
        self.batch_size = 1

    def build(self, input_shape):
        self.tok_embeddings.build(input_shape)
        embeddings_output_shape = self.tok_embeddings.compute_output_shape(input_shape)
        for layer in self.t_layers:
            layer.build(embeddings_output_shape)
        self.norm.build(embeddings_output_shape)
        self.embed_out.build(embeddings_output_shape)
        self.built = True

    def call(self, tokens: keras.KerasTensor, targets: keras.KerasTensor = None, batch_size=1, training=False):
        if not self.jit_generate:
            _bsz, seqlen = tokens.shape
        else:
            _bsz, seqlen = tokens.shape
            if _bsz is None:
                _bsz = self.batch_size
            # I found that, for the generate function to be tensorflow/jax jit compatible, the seqlen should always be fixed!
            # set it to max_seq_len
            assert seqlen == self.params.max_seq_len

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.t_layers:
            h = layer(h, freqs_cos, freqs_sin, batch_size=_bsz)

        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.embed_out(h)
            self.last_loss = ops.categorical_crossentropy(ops.reshape(logits, new_shape=(-1, logits.shape[-1])),
                                                          ops.reshape(targets, new_shape=(-1)), from_logits=True)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            if not self.jit_generate:
                logits = self.embed_out(h[None, :, -1, :])
            else:
                logits = self.embed_out(h)
            self.last_loss = None

        return logits

    def load_weights(self, filepath: str, skip_mismatch=False, **kwargs):
        if str(filepath).endswith('pt') or str(filepath).endswith('bin'):
            # load from torch llama2.c models
            import torch
            ckpt = torch.load(filepath)

            self.tok_embeddings.set_weights([ckpt['model']['tok_embeddings.weight']])

            def set_transformer_layer(tblock, i):
                tblock.attention.wq.set_weights([ckpt['model'][f'layers.{i}.attention.wq.weight'].T])
                tblock.attention.wk.set_weights([ckpt['model'][f'layers.{i}.attention.wk.weight'].T])
                tblock.attention.wv.set_weights([ckpt['model'][f'layers.{i}.attention.wv.weight'].T])
                tblock.attention.wo.set_weights([ckpt['model'][f'layers.{i}.attention.wo.weight'].T])
                # load ffn
                tblock.feed_forward.w1.set_weights([ckpt['model'][f'layers.{i}.feed_forward.w1.weight'].T])
                tblock.feed_forward.w2.set_weights([ckpt['model'][f'layers.{i}.feed_forward.w2.weight'].T])
                tblock.feed_forward.w3.set_weights([ckpt['model'][f'layers.{i}.feed_forward.w3.weight'].T])
                # load norms
                tblock.attention_norm.set_weights([ckpt['model'][f'layers.{i}.attention_norm.weight']])
                tblock.ffn_norm.set_weights([ckpt['model'][f'layers.{i}.ffn_norm.weight']])

            for i, layer in enumerate(self.t_layers):
                set_transformer_layer(layer, i)

            self.norm.set_weights([ckpt['model']['norm.weight']])
            self.embed_out.set_weights([ckpt['model']['output.weight'].T])

        else:
            # keras/tensorflow weights
            super().load_weights(filepath, skip_mismatch, **kwargs)

    def _sample_next_token(self, current_logits, temp, top_k, seed):
        # jax does not work this way the simple way!!

        if keras.backend.backend() != 'jax':
            if temp == 0.:
                _, next_token = ops.top_k(current_logits, k=1)
            else:
                # scale by temp
                current_logits = current_logits / temp
                if top_k is not None:
                    v, _ = ops.top_k(current_logits, k=top_k)
                    current_logits = ops.where(current_logits < v[:, -1], -float("Inf"), current_logits)

                next_token = keras.random.categorical(logits=current_logits, num_samples=1, dtype='int32')

            return ops.cast(next_token, dtype='int32')
        else:
            # jax again!!
            def if_true():
                return ops.top_k(current_logits, k=1)[1]

            def if_false():
                nlogits = current_logits / temp
                if top_k is not None:
                    v, _ = ops.top_k(nlogits, min(top_k, nlogits.shape[-1]))
                    nlogits = ops.where(nlogits < v[:, -1], -float("Inf"), nlogits)
                    # pass

                # no random keys in this world again!!
                # keras_core api throw error with jax/tracer seed generator
                # Couldn't find a way to make it random without breaking the XLA
                import jax
                key = jax.random.PRNGKey(0)
                output_shape = list(nlogits.shape)
                output_shape[1] = 1
                output_shape = tuple(output_shape)
                output = jax.random.categorical(
                    key=key, logits=nlogits[..., None], shape=output_shape, axis=1
                )
                return output.astype('int32')

            return ops.cond(temp == 0, if_true, if_false)
    def _get_generate(self):
        def _eager_generate(tokens, max_new_tokens, temp=0.0, top_k=None, seed=None):
            for _ in range(max_new_tokens):
                # crop if too long
                idx_cond = tokens if tokens.shape[1] <= self.params.max_seq_len else tokens[:, -self.params.max_seq_len:]
                # forward
                logits = self.call(idx_cond, training=False)
                logits = logits[:, -1, :]
                if temp == 0.:
                    _, next_token = ops.top_k(logits, k=1)
                else:
                    # scale by temp
                    logits = logits / temp
                    if top_k is not None:
                        v, _ = ops.top_k(logits, min(top_k, logits.shape[-1]))
                        logits = ops.where(logits < v[:, -1], -float("Inf"), logits)
                    next_token = keras.random.categorical(logits=logits, num_samples=1, seed=seed)
                # append sampled idx to the running seq and continue
                tokens = ops.concatenate([tokens, next_token], axis=1)

            return tokens

        def _jit_able_generate(tokens, max_new_tokens, temp=0.0, top_k=None, seed=None, start_at_token=None):
            """
            A helper function to be able to jit the generate function, in the hope of gaining a performance boost

            :param tokens:
            :param max_new_tokens:
            :param temp:
            :param top_k:
            :param seed:
            :return:
            """
            bsz, seq_len = tokens.shape
            if start_at_token is not None:
                seq_len = start_at_token
                output = tokens
            else:
                # pad tokens to max_seq_len
                output = ops.zeros((1, self.params.max_seq_len - seq_len), dtype='int32')
                output = ops.concatenate([tokens, output], axis=-1)

            def loop(i, output):
                logits = self.call(output, training=False)
                # get current token logits
                current = seq_len + i - 1
                logits = logits[:, current, :]
                next_token = self._sample_next_token(logits, temp, top_k, seed)
                # update output, here as well!!!
                if keras.backend.backend() != 'jax':
                    output = ops.scatter_update(output, indices=[[0, seq_len + i]], updates=next_token[0])
                else:
                    output = output.at[0, seq_len + i].set(next_token[0][0])
                return output

            # run fori
            output = ops.fori_loop(0, max_new_tokens, loop, output)
            return output[:, :max_new_tokens]

        if self.jit_generate:
            return _jit_able_generate
        else:
            return _eager_generate
    def generate(self, prompt: str, max_new_tokens, temp=0.0, top_k=None, seed=None):

        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        tokens = ops.convert_to_tensor(tokens, dtype='int32')
        tokens = ops.expand_dims(tokens, axis=0)  # add batch dim

        _fnc = self._get_generate()
        if self.jit_generate:
            # print(f"JIT generate: {self.jit_generate}")
            # torch is very fast even without jit

            if keras.backend.backend() == 'tensorflow':
                import tensorflow as tf
                _fnc = tf.function(_fnc)

            elif keras.backend.backend() == 'jax':
                import jax
                _fnc = jax.jit(_fnc, static_argnames=['max_new_tokens', 'top_k'])

        res = _fnc(tokens, max_new_tokens, temp, top_k, seed)
        res = self.tokenizer.decode(ops.convert_to_numpy(res).tolist())
        return res[0]


def get_model_from_ckpt(filepath: str, tokenizer_model: str = None, jit_generate=True) -> LLaMATransformer:
    """
     Build a keras model from llama2.c Pytorch checkpoint

    :param ckpt: checkpoint file
    :param tokenizer_model: the tokenizer.model file
    :return: LLaMA transformer model
    """
    import torch
    ckpt = torch.load(filepath)
    args = ModelArgs(**ckpt['model_args'])
    model = LLaMATransformer(args, tokenizer_model=tokenizer_model, jit_generate=jit_generate)
    if jit_generate:
        model.build((1, args.max_seq_len))
    else:
        model.build((None, None))
    model.load_weights(filepath)
    return model


if __name__ == '__main__':
    ckpt = '../_models/stories15M.pt'
    model = get_model_from_ckpt(ckpt, jit_generate=True)
    model.summary()
    res = model.generate("Once upon a time ", max_new_tokens=10, temp=0.8, top_k=15, seed=1234)
    print(res)
