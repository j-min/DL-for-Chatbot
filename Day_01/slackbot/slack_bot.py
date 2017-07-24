# -*- coding: utf-8 -*-
from slacker import Slacker
import websocket
import json
import logging
from abstract_reader import detect_url, parse_abstract

joined_text = '안녕하세요 반갑습니다 :)'


class slackbot:
    def __init__(self, token):
        self._token = token
        self.slack = Slacker(token)

        response = self.slack.rtm.start()
        endpoint = response.body['url']
        self.socket = websocket.create_connection(endpoint)

    def recv(self):
        data = self.socket.recv()
        return json.loads(data)

    def send_text(self, message):
        """Send text message"""
        # https://api.slack.com/docs/message-attachments
        attachments_dict = dict()
        attachments_dict['pretext'] = ""  # attachments 블록 전에 나타나는 text
        attachments_dict['title'] = ""  # 다른 텍스트 보다 크고 볼드되어서 보이는 title
        attachments_dict['title_link'] = ""
        attachments_dict['fallback'] = ""  # 클라이언트에서 노티피케이션에 보이는 텍스트 입니다. attachment 블록에는 나타나지 않습니다
        attachments_dict['text'] = message  # 본문 텍스트! 5줄이 넘어가면 *show more*로 보이게 됩니다.
        attachments_dict['mrkdwn_in'] = []  # 마크다운을 적용시킬 인자들을 선택합니다.
        attachments = [attachments_dict]

        self.slack.chat.post_message(channel='#general', text=None,
                                     attachments=attachments, as_user=True)

    def read_arxiv(self, urls, channel):
        """Parse Arxiv urls and send abstract of the paper in pretty format"""
        contents = parse_abstract(urls)
        for content in contents:
            attachments = [
                {
                    "color": "#36a64f",
                    "title": content['title'],
                    "title_link": content['url'],
                    "author_name": content['authors'],
                    "text": content['abstract'],
                }
            ]
            self.slack.chat.post_message(
                channel=channel,
                text='Here is Summary :)',
                attachments=attachments,
                as_user=True)


if __name__ == '__main__':
    # Read token for access
    token = open('./token').read()[:-1]

    # Create Log
    logging.basicConfig(filename='./log', level=logging.DEBUG)

    # Launch slack bot!
    bot = slackbot(token)

    while True:
        data = bot.recv()
        logging.info(data)

        if data['type'] == 'member_joined_channel' and data['channel'] == '#general':
            bot.send_text(joined_text)

        # Arxiv reader
        try:
            text = data['text']
            # only check when text is long enough
            if len(text) > 20:
                urls = detect_url(text)
                if len(urls) > 0:
                    bot.read_arxiv(urls, data['channel'])
        except KeyError:
            continue
