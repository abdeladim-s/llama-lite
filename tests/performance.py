#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple script to compare the performance between the three backends
"""
import subprocess

ckpt = '../_models/stories15M.pt'
prompt = "Once upon a time"
max_new_tokens = 50
top_k = 40
seed = 1234
backends = ['torch', 'tensorflow', 'jax']
num_iterations = 1
jit_generate = True
use_cpu = False


def test():
    # A simple program to load the model and run a simple generate of `max_new_tokens` new tokens
    import os
    if use_cpu:
        # Force cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from llama_lite.model import get_model_from_ckpt
    model = get_model_from_ckpt(ckpt, jit_generate=jit_generate)
    model.generate("Once upon a time ", max_new_tokens=max_new_tokens, top_k=top_k, seed=seed)


def run(ckpt=ckpt,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
        backends=backends,
        num_iterations=num_iterations,
        use_cpu=use_cpu
        ):
    if use_cpu:
        print("Using CPU ...")
    for backend in backends:
        print(f"Running {backend} backend in isolated subprocess ...")
        stm = f"'test()', setup='from performance import test', number={num_iterations}"
        cmd = f'KERAS_BACKEND={backend} python3 -c "import timeit; print(timeit.timeit({stm}))"'
        res = subprocess.check_output([cmd], shell=True)
        time = float(str(res).split('\\n')[1])
        print(f"Total inference time of generating {max_new_tokens} tokens is {time:.2f} s")


if __name__ == '__main__':
    run()
    # test()
