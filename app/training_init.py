# coding: utf-8
import sys
sys.path.append('.')
import torch
import torchtext
from janome.tokenizer import Tokenizer
import jaconv
import dill

##### 対話文の取得
path = "./"  # 保存場所を指定

j_tk = Tokenizer()
def tokenizer(text): 
    return [tok for tok in j_tk.tokenize(text, wakati=True)]  # 内包表記
 
def tokenizer_hira(text): 
    tk = j_tk.tokenize(text)
    return [jaconv.kata2hira(token.reading) for token in tk]  # 内包表記

# データセットの列を定義
input_field = torchtext.data.Field(  # 入力文
    sequential=True,  # データ長さが可変かどうか
    tokenize=tokenizer,  # 前処理や単語分割などのための関数
    batch_first=True,  # バッチの次元を先頭に
    lower=True  # アルファベットを小文字に変換
    )

reply_field = torchtext.data.Field(  # 応答文
    sequential=True,  # データ長さが可変かどうか
    tokenize=tokenizer_hira,  # 前処理や単語分割などのための関数
    init_token = "<sos>",  # 文章開始のトークン
    eos_token = "<eos>",  # 文章終了のトークン
    batch_first=True,  # バッチの次元を先頭に
    lower=True  # アルファベットを小文字に変換
    )

##### csvファイルからデータセットを作成
train_data, test_data = torchtext.data.TabularDataset.splits(
    path=path,
    train="dialogues_train.csv",
    validation="dialogues_test.csv",
    format="csv",
    fields=[("inp_text", input_field), ("rep_text", reply_field)]  # 列の設定
    )

# 単語とインデックスの対応
input_field.build_vocab(train_data, min_freq=1)
reply_field.build_vocab(train_data, min_freq=1)

print(input_field.vocab.freqs)  # 各単語の出現頻度
print(input_field.vocab.stoi)
print(input_field.vocab.itos)
print()
print(reply_field.vocab.freqs)  # 各単語の出現頻度
print(reply_field.vocab.stoi)
print(reply_field.vocab.itos)
print()
for example in train_data.examples[:10]:
    print(example.inp_text, example.rep_text)

##### データセットの保存
torch.save(train_data.examples, path + "train_examples.pkl", pickle_module=dill)
torch.save(test_data.examples, path + "test_examples.pkl", pickle_module=dill)

torch.save(input_field, path + "input_field.pkl", pickle_module=dill)
torch.save(reply_field, path + "reply_field.pkl", pickle_module=dill)

