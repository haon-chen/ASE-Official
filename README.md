# ASE4Review

This anonymous repository contains the source code for the CIKM 2022 submission "Auto-Session-Encoder: Utilizing Invisible Sequences to Model Session Context".

To ensure anonymity, we will cite the source of some codes later.

## Requirements
- Python 3.7.13 <br>
- Pytorch 1.11.0 (with GPU support) <br>
- Transformers 4.18.0 <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- data 
  - To ensure anonymity, we will provide the preprocessed dataset later.

## Run
- For correlation with user satisfaction on FSD:
```
python ./run_fsd_experiments.py
```
- For intuitiveness on NST:
```
python ./run_nst_experiments.py
```
