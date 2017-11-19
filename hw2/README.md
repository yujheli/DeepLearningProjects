ADL x MLDS 2017 Fall HW2 - Video Captioning
========================

Used Library:
------------------------

- Tensorflow 1.3.0
- Numpy
- pandas

User guide:
------------------------
## Training

If you want to run the training code:

```python

python hw2_seq2seq_attention.py '--train' <data repository> 

```
or
```python

python model_seq2seq.py '--train' <data repository> 

```
The file `hw2_seq2seq_attention.py` and `model_seq2seq.py` are identical

## Testing
If you want to run the testing code

```bash

sh hw2_seq2seq.sh <data repository> <testing output name> <peer review output name>

```

PS: the `hw2_seq2seq.sh` will automatically download the model from other url, and then run the testing code


## Generating special
If you want to run the special code

```bash

sh hw2_special.sh <data repository> <special output name>

```
PS: the `hw2_special.sh` will automatically download the model from other url, and then run the testing code


