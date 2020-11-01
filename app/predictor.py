# coding: utf-8
import sys
sys.path.append('.')
import re
import torch
from janome.tokenizer import Tokenizer
import dill
from seq2seq import evaluate_model,create_seq2seq

j_tk = Tokenizer()
def tokenizer(text): 
    return [tok for tok in j_tk.tokenize(text, wakati=True)]

class Predictor:
    def __init__(self, path='./'):
        dic = torch.load(path+"dic.pkl", pickle_module=dill)
        self.input_field = dic["input"]
        self.reply_field = dic["reply"]
        self.rep_n_time = dic["rep_n_time"]
        is_gpu = False
        encoder, decoder, seq2seq = create_seq2seq(self.input_field, self.reply_field, is_gpu)
        self.seq2seq = seq2seq
        self.seq2seq.load_state_dict(torch.load(path + "model_seq2seq.pth",
                                                map_location=torch.device("cpu")))
        self.seq2seq.eval()  # 評価モード

    def predict(self, text):
        texts = tokenizer(text)
        text_id = [[self.input_field.vocab.stoi[x] for x in texts]]  #idの配列に変換
        text_id = torch.tensor(text_id)  #テンソル型に変換
        obj = type('', (), {})()         # 空のオブジェクト作成
        obj.inp_text = text_id
        _ , rep_text = evaluate_model(self.seq2seq, [obj],  #予測する
                                      self.input_field, 
                                      self.reply_field, 
                                      self.rep_n_time, 
                                      silent=True)
        return re.sub('( <sos> |<sos> | <eos>)','',rep_text)

if __name__ == "__main__":
    pre = Predictor()
    text = '決済区分'
    rep_text = pre.predict(text)
    print("input:", text)
    print("reply:", rep_text)
