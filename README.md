# LLaMA-Lite
A Multi-Backend (Pytorch, Tensorflow, Jax) implementation of [LLaMA](https://github.com/facebookresearch/llama) using [keras-core](https://github.com/keras-team/keras-core).

<a target="_blank" href="https://colab.research.google.com/github/abdeladim-s/mbllama/blob/main/notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Installation
* Install your backend of choice  (Pytorch, Tensorflow or Jax)
* Then install `llama_lite`
```shell
git clone https://github.com/abdeladim-s/llama-lite && cd llama-lite
pip install -e .
```

# Inference usage

* Get the `tinyllama` model weights from [HF](https://huggingface.co/karpathy/tinyllamas/tree/main).

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"

from llama_lite.model import get_model_from_ckpt

model = get_model_from_ckpt('stories15M.pt')

prompt = "Once upon a time,"
max_new_tokens = 50
res = model.generate(prompt=prompt, max_new_tokens=max_new_tokens)
print(res)
```

# License
MIT