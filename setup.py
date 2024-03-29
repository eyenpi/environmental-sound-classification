
from setuptools import setup, find_packages

setup(
    name='audio-classifier',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchaudio',
        'torchvision',
        'pytorch_lightning',
        'pyaudio',
        'scikit-learn',
    ]
)
