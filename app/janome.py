# -*- coding: utf-8 -*-
"""
pip install janome
pip install jaconv

"""

from janome.tokenizer import Tokenizer
import jaconv

t = Tokenizer()

for token in t.tokenize(u'東証決済日スタート'):
    print(token.surface)
    print(token.reading)
    hira = jaconv.kata2hira(token.reading)
    print(hira)

