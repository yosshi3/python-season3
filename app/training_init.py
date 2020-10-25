# coding: utf-8
import sys
sys.path.append('.')

import torch
import torchtext
from janome.tokenizer import Tokenizer
import jaconv
import dill

from seq2seq import evaluate_model,create_seq2seq

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
input_field.build_vocab(
    train_data,
    min_freq=1,
    )
reply_field.build_vocab(
    train_data,
    min_freq=1,
    )

print(input_field.vocab.freqs)  # 各単語の出現頻度
print(input_field.vocab.stoi)
print(input_field.vocab.itos)
print(reply_field.vocab.stoi)
print(reply_field.vocab.itos)

##### データセットの保存
torch.save(train_data.examples, path + "train_examples.pkl", pickle_module=dill)
torch.save(test_data.examples, path + "test_examples.pkl", pickle_module=dill)

torch.save(input_field, path + "input_field.pkl", pickle_module=dill)
torch.save(reply_field, path + "reply_field.pkl", pickle_module=dill)

##### データセットの読み込み

train_examples = torch.load(path + "train_examples.pkl", pickle_module=dill)
test_examples = torch.load(path+"test_examples.pkl", pickle_module=dill)

input_field = torch.load(path + "input_field.pkl", pickle_module=dill)
reply_field = torch.load(path + "reply_field.pkl", pickle_module=dill)

train_data = torchtext.data.Dataset(
    train_examples, 
    [("inp_text", input_field), ("rep_text", reply_field)]
    )
test_data = torchtext.data.Dataset(
    test_examples, 
    [("inp_text", input_field), ("rep_text", reply_field)]
    )

##### Iteratorの設定

batch_size = 32

train_iterator = torchtext.data.Iterator(
    train_data,
    batch_size=batch_size, 
    train=True  # シャッフルして取り出す
)

test_iterator = torchtext.data.Iterator(
    test_data,
    batch_size=batch_size, 
    train=False,
    sort=False
)

# ミニバッチを取り出して、内容を表示します。
# ミニバッチには、単語をインデックスに置き換えた文章が格納されます。
batch = next(iter(train_iterator))  # ミニバッチを取り出す
print(batch.inp_text.size())  # ミニバッチにおける入力のサイズ
print(batch.inp_text[0])  # 最初の要素
print(batch.rep_text.size())  # ミニバッチにおける応答のサイズ
print(batch.rep_text[0])  # 最初の要素

##### 学習

from torch import optim
import torch.nn as nn

#is_gpu = True  # GPUを使用するかどうか
is_gpu = False

early_stop_patience = 5  # 早期終了のタイミング（何回連続で誤差が上昇したら終了か）
clip = 100.0

# Seq2Seqのモデルを構築
encoder, decoder, seq2seq = create_seq2seq(input_field, reply_field, is_gpu)

# 誤差関数
loss_fnc = nn.CrossEntropyLoss(ignore_index=reply_field.vocab.stoi["<pad>"])

# 最適化アルゴリズム
optimizer_enc = optim.Adam(seq2seq.parameters(), lr=0.0001)
optimizer_dec = optim.Adam(seq2seq.parameters(), lr=0.0005)

# 損失のログ
record_loss_train = []
record_loss_test = []
min_losss_test = 0.0

# 学習
for i in range(100):
    # 訓練モード
    seq2seq.train()

    loss_train = 0
    for j, batch in enumerate(train_iterator):
        inp, rep = batch.inp_text, batch.rep_text
        x_enc = inp
        x_dec = torch.ones(rep.size(), dtype=torch.long) * reply_field.vocab.stoi["<sos>"]
        x_dec[:, 1:] = rep[:, :-1]
        y_dec = seq2seq(x_enc, x_dec)

        t_dec = rep.cuda() if is_gpu else rep
        loss = loss_fnc(
            y_dec.view(-1, y_dec.size()[2]),
            t_dec.view(-1)
            )
        loss_train += loss.item()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        optimizer_enc.step()
        optimizer_dec.step()

        if j%1000==0:
            print("batch:", str(j)+"/"+str(len(train_data)//batch_size+1), "loss:", loss.item())
    loss_train /= j+1
    record_loss_train.append(loss_train)

    # 評価モード
    seq2seq.eval()

    loss_test = 0
    for j, batch in enumerate(test_iterator):
        inp, rep = batch.inp_text, batch.rep_text
        x_enc = inp
        x_dec = torch.ones(rep.size(), dtype=torch.long) * reply_field.vocab.stoi["<sos>"]
        x_dec[:, 1:] = rep[:, :-1]
        y_dec = seq2seq(x_enc, x_dec)

        t_dec = rep.cuda() if is_gpu else rep
        loss = loss_fnc(
            y_dec.view(-1, y_dec.size()[2]),
            t_dec.view(-1)
            )
        loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if i%1 == 0:
        print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)
        print()

    evaluate_model(seq2seq, test_iterator, input_field, reply_field)

    # ----- 早期終了 -----
#     latest_min = min(record_loss_test[-(early_stop_patience):])  # 直近の最小値
#     if len(record_loss_test) >= early_stop_patience:
#         if latest_min > min_loss_test:  # 直近で最小値が更新されていなければ
#             print("Early stopping!")
#             break
#         min_loss_test = latest_min
#     else:
#         min_loss_test = latest_min

##### 誤差の推移

import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

##### データの保存

# state_dict()の表示
for key in seq2seq.state_dict():
    print(key, ": ", seq2seq.state_dict()[key].size())
# print(seq2seq.state_dict()["encoder.embedding.weight"][0])  # 　パラメータの一部を表示

torch.save(seq2seq.to('cpu').state_dict(), path + "model_seq2seq.pth")
