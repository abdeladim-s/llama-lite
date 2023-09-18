#!/usr/bin/env python
# -*- coding: utf-8 -*-


from unittest import TestCase
import os

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestModelLoad(TestCase):

    def _test_load(self):
        from llama_lite.model import get_model_from_ckpt
        ckpt = '../_models/stories15M.pt'
        model = get_model_from_ckpt(ckpt)
        self.assertIsNotNone(model)

    def test_load_torch(self):
        os.environ["KERAS_BACKEND"] = "torch"
        return self._test_load()

    def test_load_tf(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        return self._test_load()

    def test_load_jax(self):
        os.environ["KERAS_BACKEND"] = "jax"
        return self._test_load()


class TestGenerate(TestCase):

    def _test_generate(self):
        from llama_lite.model import get_model_from_ckpt
        ckpt = '../_models/stories15M.pt'
        model = get_model_from_ckpt(ckpt)
        res = model.generate("Once upon a time,", max_new_tokens=50, temp=0.8, top_k=40, seed=1234)
        print(res)
        self.assertIsInstance(res, str)

    def test_generate_torch(self):
        os.environ["KERAS_BACKEND"] = "torch"
        return self._test_generate()

    def test_generate_tf(self):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        return self._test_generate()

    def test_generate_jax(self):
        os.environ["KERAS_BACKEND"] = "jax"
        return self._test_generate()