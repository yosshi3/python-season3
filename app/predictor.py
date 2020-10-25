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
    return [tok for tok in j_tk.tokenize(text, wakati=True)]  # 内包表記

class Predictor:
    def __init__(self, path='./'):
        self.input_field = torch.load(path+"input_field.pkl", pickle_module=dill)
        self.reply_field = torch.load(path+"reply_field.pkl", pickle_module=dill)
        #is_gpu = True  # GPUを使用するかどうか
        is_gpu = False
        encoder, decoder, seq2seq = create_seq2seq(self.input_field, self.reply_field, is_gpu)
        self.seq2seq = seq2seq
        self.seq2seq.load_state_dict(torch.load(path+"model_seq2seq.pth", 
                                           map_location=torch.device("cpu")))  #CPU対応
        self.seq2seq.eval()  # 評価モード

    def predict(self, text):
        #分ち書きした配列作成
        texts = tokenizer(text)
        #idの配列に変換
        text_id = [[self.input_field.vocab.stoi[x] for x in texts]]
        #テンソル型に変換
        text_id = torch.tensor(text_id)
        # 空のオブジェクト作成
        obj = type('', (), {})()
        obj.inp_text = text_id
        #予測する
        _ , rep_text = evaluate_model(self.seq2seq, [obj], 
                                      self.input_field, self.reply_field, silent=True)
        return re.sub('(<sos>|<eos>)','',rep_text)

if __name__ == "__main__":

    pre = Predictor()
    text = '個別郵便番号'
    rep_text = pre.predict(text)
    print("input:", text)
    print("reply:", rep_text)

