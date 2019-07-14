import csv
import urllib
import urllib.request

import gensim
from bs4 import BeautifulSoup


# for UnicodeEncodeError
def savefile(content, filename):
    f = open("wikiData/" + filename, "a")
    f.write(str(content) + "\n")
    f.close()


def spideWiki(words):
    all_words = []
    erroor_number = 0
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    for i in range(len(words)):
        try:
            url = "https://en.wikipedia.org/wiki/" + words[i]
            request = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(request)
            wikiHtml = response.read().decode('utf-8')
            html = BeautifulSoup(str(wikiHtml), "lxml")
            firstHead = html.find(name='h1', id='firstHeading')
            div = html.find(name='div', id='mw-content-text')
            ps = div.find_all(name='p')  # only direct children
            print(words[i], len(ps))
            for p in ps:
                pText = p.get_text()
                pText = str(pText)
                pText_words = pText.split(" ")
                all_words.append(pText_words)
                # saveFile(pText, words[i])
            print(words[i], "process over...", "==" * 20)
        except:
            print('爬取失败', words[i])
            erroor_number += 1
            all_words.append(str(words[i]).split(" "))
    return all_words, erroor_number


path = '/home/luchixiang/Desktop/laji.csv'
train_data = []
reader = csv.reader(open(path, 'r'))
data = list(reader)
train_data_inside = []
for i in range(1, len(data)):
    temp = data[i][0].lower()
    train_data_inside.append(temp)
train_data.append(train_data_inside)
all_words, error_number = spideWiki(train_data_inside)
fp = open('all_words.txt', 'w')
for words in all_words:
    for word in words:
        fp.write(word)
        fp.write(" ")
    fp.write('\n')
fp.close()
model = gensim.models.Word2Vec(all_words, min_count=1)
print(error_number)
print(model['Ai ye'])

pass