from sklearn.externals import joblib

################################################
# joblib cannot fully serialize external modules..
from konlpy.tag import Mecab
mecab = Mecab()
def mecab_tokenizer(text):
    tokens = mecab.morphs(text)
    return tokens
#################################################

class SentimentEngine(object):
    def __init__(self):
        # 저장된 모델 불러오기
        self.model = joblib.load('sentiment_engine.pkl')

    def score(self, text):
        # 고객의 감정 (0~1)
        # 0: 부정 / 1: 긍정 
        score = self.model.predict_proba([text])[:,1][0]
        return score

if __name__ == '__main__':
    scorer = Scorer()
