# coding:utf-8
from setuptools import setup, find_packages

setup(
    name="default_database_pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="Pipeline for corporate default analysis using WRDS",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pkuSapphire/default_database_pipeline",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "scikit-learn", "statsmodels", "wrds", "requests"
    ],
    python_requires=">=3.7",
)
