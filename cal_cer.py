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

# pred_file = "pred_20230619-200956_epoch_1.csv"
pred_file = "pred_20231013-164450_EricChang-TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch3-total5epoch_TD_test.csv"
predictions = []
cnt = 0 
sum_cer = 0
cnt_nonEng = 0
sum_cer_nonEng = 0

results = []

with open (pred_file, "r") as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        predictions.append({"label": row[0], "prediction": row[1], "orig_cer": row[2]})
f.close()

label_list = []
pred_list = []

for i in range(len(predictions)):
    label_str = removeMark(predictions[i]["label"])
    pred_str = removeMark(predictions[i]["prediction"])
    label_list.append(label_str)
    pred_list.append(pred_str)
#     cer = cer_cal(label_str, pred_str)
#     cnt += 1
#     sum_cer += cer
#     predictions[i]["cer"] = cer
#     t_label = re.search('[a-zA-Z]', label_str)
#     t_pred = re.search('[a-zA-Z]', pred_str)
#     if t_label is None and t_pred is None:
#         cnt_nonEng += 1
#         sum_cer_nonEng += cer
    
#     results.append({"label": label_str, "prediction": pred_str, "cer": cer})

all_cer = cer_cal(label_list, pred_list)
print(f"All CER: {all_cer}")

# with open (f"corrected_{pred_file}", "w") as f:
#     f.write(f"label, predictions, cer\n")
#     for i in range(len(results)):
#         f.write(f"{results[i]['label']}, {results[i]['prediction']}, {results[i]['cer']}\n")
# f.close()

# print(f"cer: {sum_cer / cnt}, non-Eng cer: {sum_cer_nonEng / cnt_nonEng}")