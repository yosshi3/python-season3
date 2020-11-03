# coding: utf-8
import sys,os
sys.path.append('.')
import torch
import torchtext
from janome.tokenizer import Tokenizer
import dill

path = "./"

j_tk = Tokenizer("userdic.csv", udic_enc="utf8")
def tokenizer(text): 
    return [tok for tok in j_tk.tokenize(text, wakati=True)]
 
def tokenizer_space(text): 
    return text.split(' ')

input_field = torchtext.data.Field(
    sequential=True,  # データ長さが可変かどうか
    tokenize=tokenizer,  # 前処理や単語分割などのための関数
    batch_first=True,  # バッチの次元を先頭に
    lower=True  # アルファベットを小文字に変換
    )

reply_field = torchtext.data.Field(
    sequential=True,  # データ長さが可変かどうか
    tokenize=tokenizer_space,  # 前処理や単語分割などのための関数
    init_token = "<sos>",  # 文章開始のトークン
    eos_token = "<eos>",  # 文章終了のトークン
    batch_first=True,  # バッチの次元を先頭に
    lower=True  # アルファベットを小文字に変換
    )

train_data, test_data = torchtext.data.TabularDataset.splits(
    path=path,
    train="dialogues_translate.csv",
    validation="dialogues_translate.csv",
    format="csv",
    fields=[("inp_text", input_field), ("rep_text", reply_field)]  # 列の設定
    )

for example in train_data.examples:
    print(example.inp_text, example.rep_text)

inp_n_time = max([len(x.inp_text) for x in train_data.examples])
rep_n_time = max([len(x.rep_text) for x in train_data.examples]) + 2  #<sos><eos>を足す

input_field.build_vocab(train_data, min_freq=1)  # 辞書作成
reply_field.build_vocab(train_data, min_freq=1)

print(input_field.vocab.freqs)
print(input_field.vocab.stoi)
print(input_field.vocab.itos)
print()
print(reply_field.vocab.freqs)
print(reply_field.vocab.stoi)
print(reply_field.vocab.itos)
print()

dic = {"train": train_data.examples, "test": test_data.examples,
       "input": input_field, "reply": reply_field,
       "inp_n_time": inp_n_time, "rep_n_time": rep_n_time}
torch.save(dic, path + "dic.pkl", pickle_module=dill)

try:
    os.remove(path + "model_seq2seq.pth")
except FileNotFoundError:
    print("model_seq2seq.pthは削除済みでした。")
