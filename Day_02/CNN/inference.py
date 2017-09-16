from solver import Solver
from data_loader import get_loader, get_vocab
from configs import get_config
from utils import tokenizer

if __name__ == '__main__':
    config = get_config(batch_size=1)
    print(config)

    data_loader = get_loader(
        batch_size=config.batch_size,
        max_size=config.vocab_size,
        is_train=False,
        data_dir=config.data_dir)

    solver = Solver(config, data_loader)
    solver.build(is_train=False)
    solver.load(epoch=2)
    vocab = get_vocab()

    while True:
        text = input('Insert Sentence: ')
        text = tokenizer(text)
        text = [vocab.stoi[word] for word in text]

        prediction = solver.inference(text)

        if prediction == 0:
            print('Positive!')
        else:
            print('Negative')
