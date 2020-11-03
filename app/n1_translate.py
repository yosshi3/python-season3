# coding: utf-8
import sys
sys.path.append('.')
import csv
from janome.tokenizer import Tokenizer
import jaconv

##### 対話文の取得
path = "./"  # 保存場所を指定

j_tk = Tokenizer("userdic.csv", udic_enc="utf8")
def tokenizer_hira(text): 
    tk = j_tk.tokenize(text)
    return [jaconv.kata2hira(token.reading) for token in tk]

with open(path + "dialogues.csv", encoding='utf8') as f:
    reader = csv.reader(f)
    lst = [[inp[0], ' '.join(tokenizer_hira(inp[0]))] for inp in reader]

with open(path + "dialogues_translate.csv", 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(lst)
