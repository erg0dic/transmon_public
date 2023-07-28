from setuptools import setup, find_packages

setup(
    name="transmonrl",
    version="1.0.0",
    url="",
    author="Irtaza Khalid",
    author_email="khalidmi@cardiff.ac.uk",
    description="Robust RL pulse sequence development for the transmon",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.11.1",
        "matplotlib >= 1.5.1",
        "pandas",
        "qutip",
        "hyperopt",
        "ray[rllib]",
        "tabulate",
        "torch",
    ],
)
