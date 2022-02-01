import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ConsensusDocking",
    version="0.0.1",
    author="Laura Malo",
    author_email="laumaro95@gmail.com",
    description="Consensus Docking",
    url="https://github.com/laumalo/ConsensusDocking",
    packages=[
        'consensus_docking',
        'consensus_docking/tests',
        'consensus_docking/preprocessing',
        'consensus_docking/encoding',
        'consensus_docking/clustering',
        'consensus_docking/analysis'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
