from pathlib import Path

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')


setup(
    name="mbllama",
    version="0.0.1",
    author="Abdeladim S.",
    description="A multi-backend (Pytorch, Tensorflow, Jax) implementation of the `Large Language Model Meta AI` (aka LLaMA) using keras-core",
    long_description=long_description,
    ext_modules=[],
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages('.'),
    package_dir={'': '.'},
    long_description_content_type="text/markdown",
    license='MIT',
    # project_urls={
    #     'Documentation': 'https://abdeladim-s.github.io/mbllama/',
    #     'Source': 'https://github.com/abdeladim-s/mbllama',
    #     'Tracker': 'https://github.com/abdeladim-s/mbllama/issues',
    # },
    install_requires=["keras-core==0.1.5", "sentencepiece"],
)
