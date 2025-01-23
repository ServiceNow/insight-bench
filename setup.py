from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="insightbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Issam Laradji",
    author_email="issam.laradji@gmail.com",
    description="A benchmark for evaluating agents' ability to generate insights",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
