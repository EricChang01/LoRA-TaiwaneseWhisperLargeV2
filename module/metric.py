# -*- coding: utf-8 -*-
import editdistance as ed

# MARKS = [
#         "…", "⋯",
#         "、","。","『","』",
#         "－","─","-",
#         "「","」",
#         "？","，","｜","：","；"
# ]

MARKS = (
    "…⋯"
    "、。『』"
    "－─-"
    "「」"
    "？，｜：；"
)

# train
def postprocess(txt):
    if len(txt) > 0:
        if txt[-1] not in MARKS:
            txt += "。"
        if txt.endswith("，"):
            txt = txt[:-1] + "。"
        while len(txt) > 0 and txt[0] in MARKS:
            txt = txt[1:]
    return txt

def cer_cal(groundtruth, hypothesis):
    err = 0
    tot = 0
    # print(hypothesis)
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p.lower(), t.lower()))
        tot += len(t)
    return err / tot


def wer_cal(groundtruth, hypothesis):   
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.lower().split(' ')
        t = t.lower().split(' ')
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


