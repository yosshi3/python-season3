# coding: utf-8
import sys
sys.path.append('.')
import torch
from torch import optim
import torch.nn as nn
import torchtext
import dill
import matplotlib.pyplot as plt
from seq2seq import evaluate_model,create_seq2seq,accuracy_rate

path = "./"  # 保存場所を指定

dic = torch.load(path + "dic.pkl", pickle_module=dill)
train_examples = dic["train"]
test_examples = dic["test"]
input_field = dic["input"]
reply_field = dic["reply"]
rep_n_time = dic["rep_n_time"]

train_data = torchtext.data.Dataset(
    train_examples, 
    [("inp_text", input_field), ("rep_text", reply_field)]
    )
test_data = torchtext.data.Dataset(
    test_examples, 
    [("inp_text", input_field), ("rep_text", reply_field)]
    )

batch_size = 32
train_iterator = torchtext.data.Iterator(
    train_data, batch_size=batch_size, train=True  # シャッフルして取り出す
    )

test_iterator = torchtext.data.Iterator(
    test_data, batch_size=batch_size, train=False, sort=False
    )

batch = next(iter(train_iterator))  # ミニバッチでは、単語をインデックスに置換
print(batch.inp_text.size())
print(batch.inp_text[0])
print(batch.rep_text.size())
print(batch.rep_text[0])

is_gpu = torch.cuda.is_available()
clip = 100.0
encoder, decoder, seq2seq = create_seq2seq(input_field, reply_field, is_gpu)
try:
    seq2seq.load_state_dict(torch.load(path + "model_seq2seq.pth",
                                       map_location=torch.device("cpu")))
except FileNotFoundError:
    print("modelファイルが無いので初期パラメータから学習します。")

for key in seq2seq.state_dict():
    print(key, ": ", seq2seq.state_dict()[key].size())
# print(seq2seq.state_dict()["encoder.embedding.weight"][0])  # パラメータの一部を表示

loss_fnc = nn.CrossEntropyLoss(ignore_index=reply_field.vocab.stoi["<pad>"])
optimizer_enc = optim.Adam(seq2seq.parameters(), lr=0.0001)
optimizer_dec = optim.Adam(seq2seq.parameters(), lr=0.0005)

record_loss_train = []
record_loss_test = []
min_losss_test = 0.0
epoch = 1000 if is_gpu else 200

for i in range(epoch):
    seq2seq.train()    # 訓練モード
    loss_train = 0
    for j, batch in enumerate(train_iterator):
        inp, rep = batch.inp_text, batch.rep_text
        x_enc = inp
        sos_id = reply_field.vocab.stoi["<sos>"]
        x_dec = torch.ones(rep.size(), dtype=torch.long) * sos_id
        x_dec[:, 1:] = rep[:, :-1]
        y_dec = seq2seq(x_enc, x_dec)
        t_dec = rep.cuda() if is_gpu else rep
        loss = loss_fnc(y_dec.view(-1, y_dec.size()[2]),
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

        if i % 20 == 0:
            print("TRAIN Epoch:",i," Batch:", 
                  str(j)+"/"+str(len(train_data)//batch_size+1), "loss:", loss.item())
    loss_train /= j+1
    record_loss_train.append(loss_train)
    
    seq2seq.eval()    # 評価モード
    # loss_test = 0
    # for j, batch in enumerate(test_iterator):
    #     inp, rep = batch.inp_text, batch.rep_text
    #     x_enc = inp
    #     sos_id = reply_field.vocab.stoi["<sos>"]
    #     x_dec = torch.ones(rep.size(), dtype=torch.long) * sos_id  # 正解テンソルを<sos>で初期化 
    #     # print(rep[:, :-1])           # 最後の１列を削除
    #     # print(x_dec[:, 1:])          # 2列目から最終列まで指定
    #     x_dec[:, 1:] = rep[:, :-1]     # x_decの1列目以降にrepの最終列を削ったものを代入
    #     y_dec = seq2seq(x_enc, x_dec)  # y_decは、batchsize × 時系列 × rep単語数
    #     t_dec = rep.cuda() if is_gpu else rep
    #     loss = loss_fnc(y_dec.view(-1, y_dec.size()[2]),  # y_dec.size()[2]はrep単語数
    #                     t_dec.view(-1)                    # 正解idxを１行にreshape
    #                     )
    #     loss_test += loss.item()
    # loss_test /= j+1
    # record_loss_test.append(loss_test)

    if i % 20 == 0:
        # print("EVAL Epoch:", i, " Loss_Train:", loss_train, "Loss_Test:", loss_test)
        accuracy_rate(reply_field, y_dec, rep, is_gpu)
        print()
        evaluate_model(seq2seq, test_iterator, input_field, reply_field, rep_n_time)

##### 誤差の推移
plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
#plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

torch.save(seq2seq.state_dict(), path + "model_seq2seq.pth")  
