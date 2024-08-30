from setuptools import setup, find_packages

setup(
    name='policy_finetune',
    version='0.1.0',    
    url='url',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'matplotlib',
        'dataclasses',
    ],
)
