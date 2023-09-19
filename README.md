# knowledge_tracing

Knowledge Tracing model - is transformer encoder-decoder based model that allows you to track student's academic results.
For more information see [paper](https://github.com/ivankot88/knowledge_tracing/blob/main/Paper.pdf).

## Preparing 
Use `prepare_data.py` to collect all data from folders.

## Installing dependencies
```
pip install requirements.txt
```

## Train model
```
python main.py --wandb online/offline/disabled --test
```
Args:

* `wandb` - wandb mode

* `test` - test model training on small dataset (use wandb disabled)
