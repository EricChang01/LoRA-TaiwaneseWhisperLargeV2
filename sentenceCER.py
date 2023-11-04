import csv
import string
from zhon.hanzi import punctuation as zhon_punctuation
from module.metric import cer_cal, wer_cal
import regex as re

def removeMark(s):
    for i in string.punctuation:
        s = s.replace(i, '')
    for i in zhon_punctuation:
        s = s.replace(i, '')
    s = s.replace(' ', '')
    return s

groundTruth = "伊ê查某囝倚tī伊身軀邊。"
sent1 = "佢嘅女兒 我喺佢身邊"
sent2 = "伊ê查某囝，我tī伊身軀邊。"

groundTruth_no_mark = removeMark(groundTruth)
sent1_no_mark = removeMark(sent1)
sent2_no_mark = removeMark(sent2)

sent1_cer = cer_cal(groundTruth_no_mark, sent1_no_mark)
sent2_cer = cer_cal(groundTruth_no_mark, sent2_no_mark)

print(f"sent1 cer: {sent1_cer}\nsent2 cer: {sent2_cer}")