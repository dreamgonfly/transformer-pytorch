import setuptools
from setuptools import setup

__version__ = "0.1.0"


setup(
    name="transformer-pytorch",
    version=__version__,
    description='PyTorch implementation of Transformer from "Attention is All You Need".',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dreamgonfly/transformer-pytorch",
    author="Yongrae Jo",
    author_email="dreamgonfly@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="transformer pytorch translation",
    python_requires=">=3.7",
    tests_require=["pytest"],
)
