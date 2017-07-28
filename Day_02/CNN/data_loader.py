from torchtext.vocab import Vocab
from torchtext import data
from utils import tokenizer
import os
import pickle


def filter_pred(example):
    if example.label in ['0', '1']:
        if len(example.text) > 1:
            return True
    return False


def get_loader(batch_size=100, max_size=20000, is_train=True, data_dir=None):

    text_field = data.Field(tokenize=tokenizer, sequential=True)
    label_field = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(int))

    train_file_path = os.path.join(data_dir, 'naver_train.txt')
    test_file_path = os.path.join(data_dir, 'naver_test.txt')

    train_dataset = data.TabularDataset(
        path=train_file_path,
        format='tsv',
        fields=[
            ('id', None),
            ('text', text_field),
            ('label', label_field)
        ],
        filter_pred=filter_pred)

    text_field.build_vocab(train_dataset, max_size=max_size - 2)

    print(len(text_field.vocab))

    test_dataset = data.TabularDataset(
        path=test_file_path,
        format='tsv',
        fields=[
            ('id', None),
            ('text', text_field),
            ('label', label_field)
        ],
        filter_pred=filter_pred)

    if is_train:
        loader = iter(data.Iterator(
            dataset=train_dataset,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            train=True,  # if training set => repeat and shuffle : True
            device=-1  # CPU: -1
        ))
    else:
        loader = iter(data.Iterator(
            dataset=test_dataset,
            batch_size=batch_size,
            sort=False,
            train=False,
            device=-1))

    return loader
