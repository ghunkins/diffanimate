#
#    ___  _______________   _  ________  ______ __________
#   / _ \/  _/ __/ __/ _ | / |/ /  _/  |/  / _ /_  __/ __/
#  / // // // _// _// __ |/    // // /|_/ / __ |/ / / _/  
# /____/___/_/ /_/ /_/ |_/_/|_/___/_/  /_/_/ |_/_/ /___/  
                                                        

from setuptools import setup, find_packages

setup(
    name='diffanimate',
    version='0.1.0',
    url='https://github.com/ghunkins/diffanimate',
    author='Gregory D. Hunkins',
    author_email='greg@mage.space',
    description='Diffusers Animation',
    packages=find_packages(),    
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "diffusers",
        "transformers",
        "accelerate",
        "xformers",
        "safetensors",
        "compel",
        "einops",
        "omegaconf",
    ],
)
