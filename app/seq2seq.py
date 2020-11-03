import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_h, n_vocab, n_emb, input_field, 
                 num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.n_h = n_h
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.embedding = nn.Embedding(n_vocab, n_emb)      # 埋め込み層
        self.embedding_dropout = nn.Dropout(self.dropout)  # ドロップアウト層
        self.gru = nn.GRU(                                 # GRU層
            input_size=n_emb,  # 入力サイズ
            hidden_size=n_h,  # ニューロン数
            batch_first=True,  # 入力を (バッチサイズ, 時系列の数, 入力の数) にする
            num_layers=num_layers,  # RNN層の数（層を重ねることも可能）
            bidirectional=bidirectional,  # Bidrectional RNN
        )
        self.input_field = input_field

    def forward(self, x):
        idx_pad = self.input_field.vocab.stoi["<pad>"]  # 文章の長さを取得
        sentence_lengths = x.size()[1] - (x == idx_pad).sum(dim=1)
        y = self.embedding(x)  # 単語をベクトルに変換
        y = self.embedding_dropout(y)
        y = nn.utils.rnn.pack_padded_sequence(  # 入力のパッキング
            y,
            sentence_lengths,
            batch_first=True,
            enforce_sorted=False
            )
        y, h = self.gru(y)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)  # テンソルに戻す
        if self.bidirectional:  # 双方向の値を足し合わせる
            y = y[:, :, :self.n_h] + y[:, :, self.n_h:]
            h = h[:self.num_layers] + h[self.num_layers:]
        return y, h

class Decoder(nn.Module):
    def __init__(self, n_h, n_out, n_vocab, n_emb, num_layers=1, dropout=0.0):
        super().__init__()
        self.n_h = n_h
        self.n_out = n_out
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(n_vocab, n_emb)      # 埋め込み層
        self.embedding_dropout = nn.Dropout(self.dropout)  # ドロップアウト層
        self.gru = nn.GRU(  # GRU層
            input_size=n_emb,  # 入力サイズ
            hidden_size=n_h,  # ニューロン数
            batch_first=True,  # 入力を (バッチサイズ, 時系列の数, 入力の数) にする
            num_layers=num_layers,  # RNN層の数（層を重ねることも可能）
        )
        self.fc = nn.Linear(n_h*2, self.n_out)  # コンテキストベクトルが合流するので2倍のサイズ

    def forward(self, x, h_encoder, y_encoder):
        y = self.embedding(x)  # 単語をベクトルに変換
        y = self.embedding_dropout(y)
        y, h = self.gru(y, h_encoder)
        # ----- Attension -----
        y_tr = torch.transpose(y, 1, 2)  # 次元1と次元2を入れ替える
        ed_mat = torch.bmm(y_encoder, y_tr)  # バッチごとに行列積
        attn_weight = F.softmax(ed_mat, dim=1)  # attension weightの計算
        attn_weight_tr = torch.transpose(attn_weight, 1, 2)  # 次元1と次元2を入れ替える
        context = torch.bmm(attn_weight_tr, y_encoder)  # コンテキストベクトルの計算
        y = torch.cat([y, context], dim=2)  # 出力とコンテキストベクトルの合流
        y = self.fc(y)
        y = F.softmax(y, dim=2)
        return y, h

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_gpu=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.encoder.cuda()
            self.decoder.cuda()
        
    def forward(self, x_encoder, x_decoder):  # 訓練に使用
        if self.is_gpu:
            x_encoder, x_decoder = x_encoder.cuda(), x_decoder.cuda()
        batch_size = x_decoder.shape[0]
        n_time = x_decoder.shape[1]
        y_encoder, h = self.encoder(x_encoder)
        y_decoder = torch.zeros(batch_size, n_time, self.decoder.n_out)
        if self.is_gpu:
            y_decoder = y_decoder.cuda()
        for t in range(0, n_time):
            x = x_decoder[:, t:t+1]  # Decoderの入力を使用
            y, h= self.decoder(x, h, y_encoder)
            y_decoder[:, t:t+1, :] = y
        return y_decoder

    def predict(self, x_encoder, n_time):  # 予測に使用
        if self.is_gpu:
            x_encoder = x_encoder.cuda()
        batch_size = x_encoder.shape[0]
        y_encoder, h = self.encoder(x_encoder)
        y_decoder = torch.zeros(batch_size, n_time, dtype=torch.long)
        if self.is_gpu:
            y_decoder = y_decoder.cuda() 
        y = torch.ones(batch_size, 1, dtype=torch.long
                       ) * self.encoder.input_field.vocab.stoi["<sos>"]
        for t in range(0, n_time):
            x = y  # 前の時刻の出力を入力に
            if self.is_gpu:
                x = x.cuda()
            y, h= self.decoder(x, h, y_encoder)
            y = y.argmax(2)
            y_decoder[:, t:t+1] = y  
        return y_decoder

def evaluate_model(model, iterator, input_field, reply_field, y_n_time, silent = False):
    model.eval()  # 評価モード
    batch = next(iter(iterator))
    x = batch.inp_text
    y = model.predict(x, y_n_time)  # 予測する
    for i in range(x.size()[0]):
        inp_text = ""
        for j in range(x.size()[1]):
            word = input_field.vocab.itos[x[i][j]]
#           if word=="<pad>":
#               break
            inp_text += word

        rep_text = ""
        for j in range(y.size()[1]):
            word = reply_field.vocab.itos[y[i][j]]
#            if word=="<eos>":
#                break
            rep_text += " " + word           # 一時的に空白でつなげる 2020/10/28

        if not silent:
            print("input:", inp_text)
            print("reply:", rep_text)
    #最後に評価された文字列だけ返す
    return inp_text, rep_text

def create_seq2seq(input_field, reply_field, is_gpu):
    n_h = 32
    n_vocab_inp = len(input_field.vocab.itos)
    n_vocab_rep = len(reply_field.vocab.itos)
    n_emb = 16
    n_out = n_vocab_rep
    num_layers = 2
    bidirectional = True
    dropout = 0.1
    encoder = Encoder(n_h, n_vocab_inp, n_emb, 
                  input_field, num_layers, bidirectional, dropout=dropout)
    decoder = Decoder(n_h, n_out, n_vocab_rep, n_emb, num_layers, dropout=dropout)
    seq2seq = Seq2Seq(encoder, decoder, is_gpu=is_gpu)
    return encoder, decoder, seq2seq

def accuracy_rate(reply_field, y_dec, rep, is_gpu):
    torch.set_printoptions(precision=0, linewidth=100, sci_mode=False) 
    y_dec_tmp = y_dec.argmax(2)    # 予測の確率マトリクスから、予測マトリクスを取得
    eos_id = reply_field.vocab.stoi["<eos>"]
    pad_id = reply_field.vocab.stoi["<pad>"]
    eos_tensor = torch.ones(rep.shape, dtype=torch.long) * eos_id  # repの形で<eos>マトリクス作成
    rep_tmp = torch.where(rep != pad_id, rep, eos_tensor)  # 予測マトリクスと比較するため、
                                                           # 正解マトリクス内の<pad>を<eos>で置換
    if is_gpu:
        y_dec_tmp = y_dec_tmp.cuda()
        rep_tmp = rep_tmp.cuda()
    tmp = (y_dec_tmp == rep_tmp)  # 予測と正解を比較。テンソル成分をtrue falseに置換
    tmp, _ = tmp.min(dim=1)   # 行で全部一致していれば、true。一個でも違えばfalse。
    total_count = tmp.size()[0]  #  比較した全件数を取得
    correct_count = tmp.sum().item()       # 全部一致している行の件数を取得
    print("正解率:" , correct_count / total_count)
