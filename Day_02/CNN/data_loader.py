from torchtext.vocab import Vocab
from torchtext import data
from utils import tokenizer
from pathlib import Path
import pickle


def postprocess(x, train=True):
    x = int(x)
    return x


def filter_pred(example):
    if example.label in ['0', '1']:
        if len(example.text) > 1:
            return True
    return False


def get_loader(batch_size=100, max_size=20000, is_train=True, data_dir=None):

    text_field = data.Field(tokenize=tokenizer, sequential=True)
    label_field = data.Field(sequential=False, use_vocab=False,
                             postprocessing=data.Pipeline(postprocess))

    train_file_path = Path(data_dir).joinpath('naver_train.txt')
    test_file_path = Path(data_dir).joinpath('naver_test.txt')

    train_dataset = data.TabularDataset(
        path=train_file_path,
        format='tsv',
        fields=[
            ('id', None),
            ('text', text_field),
            ('label', label_field)
        ],
        filter_pred=filter_pred)

    print('Building Vocabulary \n')
    text_field.build_vocab(train_dataset, max_size=max_size - 2)

    if is_train:
        loader = data.Iterator(
            dataset=train_dataset,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            train=True,  # if training set => repeat and shuffle : True
            repeat=False,
            device=-1  # CPU: -1
        )
        # vocab = text_field.vocab
        # with open('./vocab.pkl', 'wb') as f:
        #     pickle.dump(vocab, f)

    else:
        test_dataset = data.TabularDataset(
            path=test_file_path,
            format='tsv',
            fields=[
                ('id', None),
                ('text', text_field),
                ('label', label_field)
            ],
            filter_pred=filter_pred)

        loader = data.Iterator(
            dataset=test_dataset,
            batch_size=batch_size,
            sort=False,
            train=False,
            device=-1)

    return loader


def get_vocab():
    with open('./vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab
