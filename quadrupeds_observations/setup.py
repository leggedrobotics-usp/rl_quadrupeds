from setuptools import setup, find_packages

setup(
    name="quadrupeds_observations",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Lucas Toschi",
    description="A Python package containing quadruped's robots observations to train Reinforcement Learning policies.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)