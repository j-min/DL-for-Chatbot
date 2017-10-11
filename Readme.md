# Deep leaning for Chatbot Developers
- Course Materials of [Deep leaning for Chatbot Developers](http://www.fastcampus.co.kr/data_seminar_chatbot/) (Sep. 2017)
- Author: [Jaemin Cho](mailto:heythisischo@gmail.com)
- Pull Requests welcome :)

## Contents

### Day 01 Introduction to Chatbot ([slideshare](https://www.slideshare.net/JaeminCho6/dl-chatbot-seminar-day-01-80593691))
- Introduction to NLP/Chatbot
- Overview of Korean/English NLP Toolkits/Datasets
- Tutorial (code)
    - Introduction to spaCy / gensim / konlpy / other Korean toolkits
    - Sentiment classification via TF-IDF (scikit-learn)
    - Chatbot Pipelining / Serving via Kakaotalk (flask) / Slack (slacker)

### Day 02 Text Classification with CNN/RNN ([slideshare](https://www.slideshare.net/JaeminCho6/dl-chatbot-seminar-day-02))
- CNN for text classification
    - Word CNN / Dynamic CNN / Char CNN / Very Deep CNN
- RNN for text classification
    - Bidirectional RNN / Recursive NN / Tree LSTM / Dual Encoder LSTM
- Advanced CNN/RNN architectures
    - QRNN / SRU / ByteNet / SliceNet / LSTM-CNNs-CRF
- Tutorial (code)
    - Word-CNN for sentiment analysis
    - PyTorch Style Guide
    - TorchText Tutorial

### Day 03 Conversation Modeling with Seq2Seq / Attention ([slideshare](https://www.slideshare.net/JaeminCho6/dl-chatbot-seminar-day-03))
- Seq2Seq models for conversation modeling
    - Seq2Seq / Neural Conversation model / Diversity-prompting objective: MMI
- Advanced Seq2Seq architectures
    - Show and Tell / HRED / VHRED / Personal based Neural Conversation model / Contextualized Word Vectors (CoVe)
- Attention mechanism
    - Bahdanau / Luong
    - Global / Local
- Advanced Attention architectures
    - Show, Attend and Tell / Pointer Networks / CopyNet / BiDAF / Transformer
- Tutorial (code)
    - Seq2Seq with Attention for Machine Translation

### Day 04 QA with External Memory ([slideshare](https://www.slideshare.net/JaeminCho6/dl-chatbot-seminar-day-04))
- QA with External Memory
    - Memory Networks / End-to-End Memory Networks / Key-value Memory Networks / Neural Turing Machines
- Advanced Memory architectures
    - DNC / Life-long memory Modules / Context-Sequence Memory Networks
- Advanced Dialogue Architectures
    - MILABOT / Dialog based language learning / End-to-End Goal Oriented Dialog / Deep RL / Adversarial
- Tutorial (code)
    - End-to-End Memory Networks for Question Answering (bAbI)

## Dependencies

### Python 3
- Codes are written in Anacodna Python 3.6.
- Package management via Conda or virtualenv is recommended.

### ML / NLP
- PyTorch
- TorchText
- spaCy
- sckit-learn
- gensim
- konlpy (requires Jpype3)

### Interactive / DataFrame / Plot
- jupyter
- pandas
- matplotlib

### Kakaotalk / Slack Bot
- flask
- websocket-client
- beautifulsoup4
- slacker
