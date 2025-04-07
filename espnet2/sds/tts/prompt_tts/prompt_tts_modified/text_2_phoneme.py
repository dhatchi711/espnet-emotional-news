import re
from typing import List

from g2p_en import G2p


# 1. config ファイルから読み取れるようにする
def read_lexicon(config):
    lexicon = {}
    with open(config.lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def get_eng_phoneme(text, g2p, lexicon, pad_sos_eos=True):
    """
    english g2p
    """
    filters = {",", " ", "'"}
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:
            
            for ph in lexicon[w.lower()]:
                if ph not in filters:
                    phones += ["[" + ph + "]"]

            if "sp" not in phones[-1]:
                phones += ["engsp1"]
        else:
            phone=g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():
                
                for ph in phone:
                    if ph not in filters:
                        phones += ["[" + ph + "]"]
                    if ph == " " and "sp" not in phones[-1]:
                        phones += ["engsp1"]
            elif phone == " ":
                continue
            elif phones:
                phones.pop() # pop engsp1
                phones.append("engsp4")
    if phones and "engsp" in phones[-1]:
        phones.pop()

    # mark = "." if text[-1] != "?" else "?"
    if pad_sos_eos:
        phones = ["<sos/eos>"] + phones + ["<sos/eos>"]
    return " ".join(phones)

class Text2Phoneme:
    def __init__(self, config):
        self.lexicon = read_lexicon(config)
        self.g2p = G2p()
    
    def convert(self, text: str) -> List[str]:
        phonemes = get_eng_phoneme(text, self.g2p, self.lexicon)

        phonemes = phonemes.split(" ")

        return phonemes
    
    def __call__(self, text: str) -> List[str]:
        return self.convert(text)