# utils.py
import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
from tensorflow.keras import backend as ktf  # ✅ 新式导入

def GeneSeg(payload):
    # 数字泛化为"0"
    payload = payload.lower()
    payload = unquote(unquote(payload))
    payload, num = re.subn(r'\d+', "0", payload)
    # 替换url为”http://u
    payload, num = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    # 分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)


def init_session():
    pass  # TF2 默认启用 eager execution，不需要手动设置 session
