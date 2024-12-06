from setuptools import setup, find_packages

setup(
    name="torchprint",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "lovely_tensors==0.1.15",
        "numpy==2.1.3",
        "rich==13.9.4",
        "setuptools==75.1.0",
        "torch==2.4.1",
],
    description="A Python module for pretty printing PyTorch tensors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alberto-rota/torchprint",
    author="Alberto Rota",
    author_email="alberto1.rota@polimi.it",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
