from konlpy.tag import Mecab
import re

print('Loading Mecab')
mecab = Mecab()

# hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
hangul = re.compile('[^ ㅋㅎ가-힣]+')


def clean(sentence):
    clean_sentence = hangul.sub('', sentence)
    return clean_sentence


def mecab_tokenizer(sentence):
    out_list = []
    for word, pos in mecab.pos(sentence):
        out_list.append(word)
    return out_list


def tokenizer(sentence):
    clean_sentence = clean(sentence)
    tokens = mecab_tokenizer(clean_sentence)
    return tokens
