from setuptools import setup, find_packages

setup(
    name="isaaclab_extensions",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Lucas Toschi",
    description="A Python package that implements new features for IsaacLab.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)