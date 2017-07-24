# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import re

p_arxiv_url = re.compile(r'(http[s]?://arxiv.org/((abs)|(pdf))/[\d]+\.[\d]+v?\d?)(\.pdf)?')


def detect_url(text):
    text = text.replace("<", "")
    text = text.replace(">", "")
    urls = p_arxiv_url.findall(text)
    urls = [url[0].replace('pdf', 'abs') for url in urls]
    return urls


def parse_abstract(urls):
    contents = []

    for url in urls:
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'html.parser')

        title = soup.title.get_text().split(' ', 1)[1]
        authors = soup.find_all(attrs={'class': 'authors'})[
            0].get_text().split('\n', 1)[1].replace('\n', '')
        abstract = soup.blockquote.get_text(" ", strip=True).replace(
            '\n', ' ').replace('Abstract: ', '')

        content = {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'url': url,
        }
        contents.append(content)

    return contents


if __name__ == '__main__':

    # sample_text = "sdfaf afs"
    sample_text = "이것 좀 보세요 \\n <https://arxiv.org/abs/1703.08132v1> 이거랑 이거 <https://arxiv.org/abs/1703.08132> 대박이네요. <https://arxiv.org/pdf/1703.08132v1.pdf>"
    # sample_text = "https://arxiv.org/abs/1703.08132v1"

    print(sample_text)

    urls = detect_url(sample_text)
    print(urls)

    if len(urls) > 0:
        contents = parse_abstract(urls)
        print("\n\n".join(contents))
