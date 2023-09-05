#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple script to compare the performance between the three backends
"""
import subprocess


ckpt = '../_models/stories15M.pt'
prompt = "Once upon a time"
max_new_tokens = 10
top_k = 40
seed = 1234
backends = ['torch', 'tensorflow', 'jax']
number_iteration = 1
use_cpu = True


def test():
    # A simple program to load the model and run a simple generate of `max_new_tokens` new tokens
    import os
    if use_cpu:
        # Force cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from mbllama.model import get_model_from_ckpt
    model = get_model_from_ckpt(ckpt)
    model.generate("Once upon a time ", max_new_tokens=max_new_tokens, top_k=top_k, seed=seed)



def run():
    if use_cpu:
        print("Using CPU ...")
    for backend in backends:
        print(f"Running {backend} backend in isolated subprocess ...")
        stm = f"'test()', setup='from performance import test', number={number_iteration}"
        cmd = f'KERAS_BACKEND={backend} python3 -c "import timeit; print(timeit.timeit({stm}))"'
        res = subprocess.check_output([cmd], shell=True)
        time = float(str(res).split('\\n')[1])
        print(f"Total inference time of generating {max_new_tokens} tokens is {time:.2f} s")


if __name__ == '__main__':
    run()
