<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

### Base Model

https://huggingface.co/openai/whisper-large-v2

### Data

The format of data has to be csv:

```
path,text
/work/hungyi2022/aics/data/TAT-train-master/condenser/wav/TA_TAF0019/0010-6.34-03.wav,我佮伊攏耍甲足歡喜的。
/work/hungyi2022/aics/data/TAT-train-master/condenser/wav/TA_TAF0019/0010-4.48-03.wav,教阮愛長志。
...
```

#### 1. [TAT Dataset](https://paperswithcode.com/dataset/tat)

Audio: Taiwanese Audio

Label: Taiwanese Text (台文)

E.g.
```
結果嘛是發生林家滅門血案。
無論查埔、查某的店頭家。
``` 

- train: 83.37022932291666 hr
- valid: 10.364263645833333 hr
- test: 10.622789305555557 hr


#### 2. TD: Taiwanese Drama Dataset (unreleased)

Audio: Taiwanese Audio

Label: Chinese Text

Set1: TD-341hr (`data/TD-341hr`)
- total: 341.6263453990207 hr

We only use the audio whose time length is longer than 2.6 second, and divide it into 3 split manually. 

Set2: TD-104hr (`data/TD-104hr`)
- train: 83.21494682291667 hr
- eval: 10.378483784722222 hr
- test: 10.39038796875 hr

### Experiment Result

We finetune `whisper-large-v2` by [LoRA](https://arxiv.org/abs/2106.09685) with the following hyperparameters:
- epoch: 5
- batch * gradient accumulation: 8
- lr: 2.34e-4
- weight decay: 0.02
- warmup step: 100


#### Finetune on TAT Dataset

<table>
    <tr>
        <td>Huggingface Model</td>
        <td>Finetuning Epoch </td>
        <td>CER on TAT eval</td>
        <td>CER on TAT test</td>
        <td>CER on TD-104hr test</td>
    </tr>
    <tr>
        <td>openai/whisper-large-v2</td>
        <td>0</td>
        <td>NAN</td>
        <td>0.74888</td>
        <td>0.77047</td>
    </tr>
    <tr>
        <td>cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch1-total5epoch</td>
        <td>1</td>
        <td>0.25106742875850874</td>
        <td>0.2580939890066167</td>
        <td>0.9583599308003293</td>
    </tr>
    <tr>
        <td>cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch2-total5epoch</td>
        <td>2</td>
        <td>0.23085107834176818</td>
        <td>0.231494716986693</td>
        <td>0.9343435745145384</td>
    </tr>
    <tr>
        <td>cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch3-total5epoch</td>
        <td>3</td>
        <td>0.26496787482777906</td>
        <td>0.2812449343491652</td>
        <td>1.005236231763389</td>
    </tr>
    <tr>
        <td>cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch4-total5epoch</td>
        <td>4</td>
        <td>0.23628030065341646</td>
        <td>0.24514065930827156</td>
        <td>0.9578973661569204</td>
    </tr>
    <tr>
        <td>cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>5</td>
        <td>0.2228163749710123</td>
        <td>0.22816428181965545</td>
        <td>0.9506813577197414</td>
    </tr>
</table>

![](https://hackmd.io/_uploads/r1_sxFdOh.png)

#### Finetune on TD-104hr Dataset

<table>
    <tr>
        <td>Huggingface Model</td>
        <td>Finetuning Epoch</td>
        <td>CER on TD-104hr eval</td>
        <td>CER on TD-104hr test</td>
        <td>CER on TAT</td>
    </tr>
    <tr>
        <td>openai/whisper-large-v2</td>
        <td>0</td>
        <td>NAN</td>
        <td>0.77047</td>
        <td>0.74888</td>
    </tr>
    <tr>
        <td>cathyi/tw-zh2.6-openai-whisper-large-v2-Lora-epoch1-total5epoch</td>
        <td>1</td>
        <td>0.345572974575643</td>
        <td>0.34513798303312887</td>
        <td>0.6583209301640166</td>
    </tr>
    <tr>
        <td>cathyi/tw-zh2.6-openai-whisper-large-v2-Lora-epoch2-total5epoch</td>
        <td>2</td>
        <td>0.3354458527907494</td>
        <td>0.3316403467384567</td>
        <td>0.6577756819287051</td>
    </tr>
    <tr>
        <td>cathyi/tw-zh2.6-openai-whisper-large-v2-Lora-epoch3-total5epoch</td>
        <td>3</td>
        <td>0.3304054554888444</td>
        <td>0.3249701645805001</td>
        <td>0.6865412104510824</td>
    </tr>
    <tr>
        <td>cathyi/tw-zh2.6-openai-whisper-large-v2-Lora-epoch4-total5epoch</td>
        <td>4</td>
        <td>0.3266066266399822</td>
        <td>0.32179697112671496</td>
        <td>0.696237787176351</td>
    </tr>
    <tr>
        <td>cathyi/tw-zh2.6-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>5</td>
        <td>0.32601363872211103</td>
        <td>0.31904008585199783</td>
        <td>0.7410218246658512</td>
    </tr>
</table>

![](https://hackmd.io/_uploads/S1Q-5WY_h.png)

#### Finetune on TAT + TD (both the entire datasets)

Following the above experiements, we try to fine-tune with the entire TAT and Taiwanese drama dataset. We have results for fine-tuning with 8 epochs. Note that the fine-tuning was conducted in two phases, initially we only ran 5 epochs, and the 6-8 epochs were fine-tuned contiually from the model of the fifth epoch.

<table>
    <tr>
        <td>Huggingface Model</td>
        <td>Finetuning Epoch</td>
        <td>CER on TD eval</td>
        <td>CER on TD test</td>
        <td>CER on TAT eval</td>
        <td>CER on TAT test</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch1-total5epoch</td>
        <td>1</td>
        <td>0.3216</td>
        <td>0.3197</td>
        <td>0.6257</td>
        <td>0.6345</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch2-total5epoch</td>
        <td>2</td>
        <td>0.2873</td>
        <td>0.2840</td>
        <td>0.6405</td>
        <td>0.6376</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch3-total5epoch</td>
        <td>3</td>
        <td>0.2653</td>
        <td>0.2579</td>
        <td>0.6549</td>
        <td>0.6685</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch4-total5epoch</td>
        <td>4</td>
        <td>0.2362</td>
        <td>0.2318</td>
        <td>0.6354</td>
        <td>0.6449</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>5</td>
        <td>0.2084</td>
        <td>0.2024</td>
        <td>0.6466</td>
        <td>0.6535</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch1-total5epoch</td>
        <td>6</td>
        <td>0.2195</td>
        <td>0.2132</td>
        <td>0.6754</td>
        <td>0.6951</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch2-total5epoch</td>
        <td>7</td>
        <td>0.2121</td>
        <td>0.2029</td>
        <td>0.6656</td>
        <td>0.6919</td>
    </tr>
    <tr>
        <td>EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch3-total5epoch</td>
        <td>8</td>
        <td>0.2019</td>
        <td>0.1953</td>
        <td>0.6775</td>
        <td>0.7026</td>
    </tr>
</table>

Note that as we combined the training data of TAT and Taiwanese drama, we noticed that the model was more proned to output Chinese text than Taiwen. Thus, the character error rate on TAT may be inaccurate as some of the output is correct but the model output in Chinese. Following are some examples:

* 叫伊共同學會失禮(ground truth) 叫他向同學道歉(model prediction) -> CER=0.57144
* 今仔日是三月初一拜六(ground truth) 今天是三月初一禮拜六(model prediction) -> CER=0.7
* 雖然引這免付錢(ground truth) 雖然他們這不用付錢(model prediction) -> CER=0.7143

We also calculate the character error rate of each model if we remove don't consider the marks(comma, period, etc.) in the groudtruth and model production. In other word, this shows the chracter error rate for only the texts.

<table>
    <tr>
        <td>Huggingface Model</td>
        <td>Finetuning Epoch</td>
        <td>CER on TD eval</td>
        <td>CER on TD test</td>
        <td>CER on TAT eval</td>
        <td>CER on TAT test</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>1</td>
        <td>0.3216</td>
        <td>0.3197</td>
        <td>0.6257</td>
        <td>0.6345</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch2-total5epoch</td>
        <td>2</td>
        <td>0.2873</td>
        <td>0.2840</td>
        <td>0.6405</td>
        <td>0.6376</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch3-total5epoch</td>
        <td>3</td>
        <td>0.2653</td>
        <td>0.2579</td>
        <td>0.6549</td>
        <td>0.6685</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch4-total5epoch</td>
        <td>4</td>
        <td>0.2362</td>
        <td>0.2318</td>
        <td>0.6354</td>
        <td>0.6449</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>5</td>
        <td>0.2084</td>
        <td>0.2024</td>
        <td>0.6466</td>
        <td>0.6535</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch1-total5epoch</td>
        <td>6</td>
        <td>0.2195</td>
        <td>0.2132</td>
        <td>0.6754</td>
        <td>0.6951</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch2-total5epoch</td>
        <td>7</td>
        <td>0.2121</td>
        <td>0.2029</td>
        <td>0.6656</td>
        <td>0.6919</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch3-total5epoch</td>
        <td>8</td>
        <td>0.2019</td>
        <td>0.1953</td>
        <td>0.6775</td>
        <td>0.7026</td>
    </tr>
</table>

We also evaluate the character error rate if we don't consider the marks (period, comma, etc.) in the groudtruth and model predictions.

<table>
    <tr>
        <td>Huggingface Model</td>
        <td>Finetuning Epoch</td>
        <td>CER on TD eval</td>
        <td>CER on TD test</td>
        <td>CER on TAT eval</td>
        <td>CER on TAT test</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch1-total5epoch</td>
        <td>1</td>
        <td>0.3211</td>
        <td>0.3192</td>
        <td>0.5913</td>
        <td>0.6044</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch2-total5epoch</td>
        <td>2</td>
        <td>0.2868</td>
        <td>0.2825</td>
        <td>0.6080</td>
        <td>0.6076</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch3-total5epoch</td>
        <td>3</td>
        <td>0.2648</td>
        <td>0.2575</td>
        <td>0.6274</td>
        <td>0.6478</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch4-total5epoch</td>
        <td>4</td>
        <td>0.2355</td>
        <td>0.2313</td>
        <td>0.6037</td>
        <td>0.6183</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch</td>
        <td>5</td>
        <td>0.2077</td>
        <td>0.2019</td>
        <td>0.6165</td>
        <td>0.6268</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch1-total5epoch</td>
        <td>6</td>
        <td>0.2189</td>
        <td>0.2127</td>
        <td>0.6500</td>
        <td>0.6766</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch2-total5epoch</td>
        <td>7</td>
        <td>0.2115</td>
        <td>0.2024</td>
        <td>0.6384</td>
        <td>0.6731</td>
    </tr>
    <tr>
        <td>https://huggingface.co/EricChang/TAT-TD-openai-whisper-large-v2-Lora-ContinualTraining-epoch3-total5epoch</td>
        <td>8</td>
        <td>0.2013</td>
        <td>0.1947</td>
        <td>0.6525</td>
        <td>0.6851</td>
    </tr>
</table>

## User Guide

### Run train script with required args
* Modify the following params in `train_peft_TAT_TD.py` and run `run.sh` to train.
  - huggingface_user_token: Specify the read/write huggingface access token
  - repo_name: Specify where the processed data is located
  - input_arg["custom_set_train"]: Specify training data
  - input_arg["custom_set_test"]: Specify evaluation or testing data
  - input_arg["load_cache"]: Set to true to load from processed data in cache directory, otherwise load the data csv file and processed again(may take lots of time)

### Run evaluate and test script

Modify the model path on huggingface in `eval_peft_TAT_TD.py` and run `eval.sh` to eva; or test.

## Developer Guide

This script allows you to finetune Whisper by Lora and "evaluate & save model on hub" for every epoch. (by using customed trainer)

### Reference
- https://github.com/ga642381/Taiwanese-Whisper
- https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
- https://github.com/YungYi0202/peft (a more general training and evaluation scripts can be found here)


### Note

Directly modify model in training script in https://github.com/ga642381/Taiwanese-Whisper to `PeftModel` will cause this problem:

Error occurs after training and then evaluating after 1st epoch because `peft` freeze some modules of model.

Error log:
```
File "train_large.py", line 310, in main
trainer.train(input_arg.get("checkpoint", None))
File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 1696, in train
return inner_training_loop(
File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 1973, in _inner_training_loop
tr_loss_step = self.training_step(model, inputs)
File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 2797, in training_step
self.scaler.scale(loss).backward()
File "/home/hungyi2022/.local/lib/python3.8/site-packages/torch/_tensor.py", line 488, in backward
torch.autograd.backward(
File "/home/hungyi2022/.local/lib/python3.8/site-packages/torch/autograd/init.py", line 197, in backward
Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```


Using example script in `peft` still causes 2 problems:

1. Model can only be evaluated and saved after training stage is finished.

Since `compute_metric` can not be set in trainer (See [reason](https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb)), it causes error if save_strategy is set "steps" or "epoch".

Error log:
```
  File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 1696, in train
    return inner_training_loop(
  File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 2052, in _inner_training_loop
    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
  File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 2360, in _maybe_log_save_evaluate
    self._save_checkpoint(model, trial, metrics=metrics)
  File "/home/hungyi2022/.local/lib/python3.8/site-packages/transformers/trainer.py", line 2473, in _save_checkpoint
    metric_value = metrics[metric_to_check]
KeyError: 'eval_cer'
```

2. `fp16` and `int8` can not be set together.

https://github.com/mymusise/ChatGLM-Tuning/issues/23

---