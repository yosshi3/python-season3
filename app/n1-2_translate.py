# coding: utf-8
import sys
sys.path.append('.')
import csv
from janome.tokenizer import Tokenizer
import jaconv
import itertools

##### 対話文の取得
path = "./"  # 保存場所を指定

j_tk = Tokenizer("userdic.csv", udic_enc="utf8")

def tokenizer(text): 
    tk = j_tk.tokenize(text, wakati=True)
    return tk

def tokenizer_hira(text): 
    tk = j_tk.tokenize(text)
    return [jaconv.kata2hira(token.reading) for token in tk]

def get2pair(xs):
    return  [x1 + x2 for x1, x2 in zip(xs, xs[1:])]


with open(path + "dialogues.csv", encoding='utf8') as f:
    reader = csv.reader(f)
    lst = [inp[0] for inp in reader]

lst2 = [[x] + get2pair(list(tokenizer(x))) for x in lst]
lst3 = list(set(itertools.chain.from_iterable(lst2)))
lst4 = [[x, ' '.join(tokenizer_hira(x))] for x in lst3]

with open(path + "dialogues_translate.csv", 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(lst4)


if __name__ == "__main__":
    xs = "吾輩は猫である"
    xs2 = get2pair(list(tokenizer(xs)))
