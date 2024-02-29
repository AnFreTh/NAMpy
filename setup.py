from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


# Read the contents of your README file
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nampy",  # Replace with your package's name
    version="0.1.0",  # The current version of your package
    author="Anton Thielmann",  # Replace with your name
    author_email="anton.thielmann@tu-clausthal.de",  # Replace with your email
    description="A python package for explainable deep learning models for tabular data with a focus on additive and distributional regression.",  # A short description of your package
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnFreTh/NAMpy",  # Replace with the URL to your package's repository
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "*.ipynb", "*.ipynb.*"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
