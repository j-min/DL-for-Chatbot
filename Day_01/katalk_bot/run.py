# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify
import random
from sentiment import SentimentEngine, mecab_tokenizer

app = Flask(__name__)

@app.route('/')
def Welcome():
    return app.send_static_file('index.html')

@app.route('/myapp')
def WelcomeToMyapp():
    return '반갑습니다 :)'

@app.route('/keyboard')
def Keyboard():
    message = {
            'type': 'buttons',
            'buttons': ["선택1", "선택2"],
    }
    return jsonify(message)

@app.route('/message', methods=['POST'])
def GetMessage():
    # 고객이 보낸 메시지 정보 (dict)
    received_data = request.get_json()

    # 고객 메시지 중 텍스트 부분 (string)
    text = received_data['content']

    # 감정 분석 점수 (0 ~ 1)
    score = sentiment.score(text)

    # 긍정문
    if score > 0.5:
        answer = random.choice(
                ['저도 그 영화 좋아해요', '저도 좋아요']
        )
    # 부정문
    else:
        answer = random.choice(
                ['저도 그 영화 싫어해요', '별로에요']
        )

    message = {
        "message": {
            "text": answer,
        }
    }
    return jsonify(message)

@app.errorhandler(404)
def page_not_found(e):
    error_message = {
        "message": {
            "text": '잘못된 접근입니다!'
        }
     }
    return jsonify(error_message)

if __name__ == "__main__":
    # 감정 분석 엔진 불러오기
    sentiment = SentimentEngine()

    # 미리 설정된 포트가 없으면 5000번 이용
    port = os.getenv('PORT', '5000')

    # 플라스크 서버 실행
    app.run(host='0.0.0.0', port=int(port))

    # ./ngrok http 5000
