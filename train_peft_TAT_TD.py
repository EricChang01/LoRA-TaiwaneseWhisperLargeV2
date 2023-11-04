# Run without int8, with fp16
from huggingface_hub import notebook_login

notebook_login()

import inspect
import random
import sys

import nlp2
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer
from transformers import Trainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from datasets import load_from_disk

from module.args import parse_args
from module.data_processing import (
    encode_dataset,
    DataCollatorCTCWithPadding,
    prepare_dataset_hf,
    prepare_dataset_custom,
    prepare_dataset_whisper,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from module.metric import cer_cal, wer_cal, postprocess
from module.utility import FreezingCallback

from datetime import datetime

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig
from peft import prepare_model_for_int8_training

# Peft
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import torch
import subprocess

huggingface_user_token ='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' NOTE: specify the read/write huggingface access token

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

def load_peft_model_from_hub(peft_model_id):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, peft_model_id,  is_trainable=True) # the is_trainable parameter=true to make sure the model is tranable is we load the checkpoint instead of the base model. 
    
    print("Load model from hub successfully.")
    return model

# for LrRescheduleTrainer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

class LrRescheduleTrainer(Seq2SeqTrainer):
    def __init__(self, specified_epoch, total_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom attributes here
        if specified_epoch is None:
            self.total_epoch = 1
            self.specified_epoch = 0
        else:
            self.total_epoch = total_epoch
            self.specified_epoch = specified_epoch
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        
        self.lr_scheduler = self.get_linear_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
               The index of the last epoch when resuming training. 

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        lr_lambda = partial(
            self._get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _get_linear_schedule_with_warmup_lr_lambda(self, current_step: int, *, num_warmup_steps: int, num_training_steps: int):
        # The only difference
        current_step += num_training_steps * self.specified_epoch
        num_training_steps *= self.total_epoch

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def experiment(input_arg, model, processor, data_collator, repo_name, data_train, data_test, time):
    ################
    #     Train    #
    ################


    if input_arg.get("sweep_split_shard", False):
        shuffled_dataset = data_train.shuffle(seed=42)
        data_train = shuffled_dataset.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_train = data_train.shard(num_shards=input_arg.get("sweep_split_shard"), index=0)
        data_test = data_train


    training_args = Seq2SeqTrainingArguments(
        output_dir=input_arg.get("output_dir", repo_name),
        length_column_name="lengths",
        group_by_length=input_arg["group_by_length"],
        per_device_train_batch_size=int(input_arg["batch"]),
        per_device_eval_batch_size=int(input_arg["batch"]),
        gradient_accumulation_steps=int(input_arg["grad_accum"]),
        eval_accumulation_steps=int(input_arg["grad_accum"]),
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        # eval_steps=50,
        save_strategy="no",
        # save_steps=input_arg.get("eval_steps", 10),
        # eval_steps=input_arg.get("eval_steps", 10),
        ddp_find_unused_parameters=True,
        resume_from_checkpoint=input_arg.get("checkpoint", False),
        overwrite_output_dir=input_arg.get("overwrite_output_dir", False),
        # load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="cer",
        num_train_epochs=input_arg.get("epoch", 5),
        fp16=True,
        logging_steps=input_arg.get("logging_steps", 10),
        # learning_rate=input_arg.get("learning_rate", 2.34e-4),
        learning_rate=input_arg.get("learning_rate", 4.7e-5),
        warmup_steps=input_arg.get("warmup_steps", 100),
        # warmup_steps=input_arg.get("warmup_steps", 50),
        save_total_limit=input_arg.get("save_total_limit", 3),
        push_to_hub=False,
        report_to="all",
        weight_decay=input_arg.get("weight_decay", 0.02),
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    training_args.generation_max_length = 225

    trainer = LrRescheduleTrainer(
        specified_epoch=input_arg['specified_epoch'],
        total_epoch=input_arg['total_epoch'],
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Push to Hub    
    peft_model_id = "EricChang/" + f"TAT-TD-{input_arg['model_config']}-Lora-ContinualTraining".replace("/", "-")
    if input_arg['specified_epoch'] is not None:
        peft_model_id += f"-epoch{str(input_arg['specified_epoch']+1)}-total{input_arg['total_epoch']}epoch"
    print(f"peft_model_id: {peft_model_id}")

    if not input_arg.get("only_eval", False):
        trainer.train(input_arg.get("checkpoint", None))
        model.push_to_hub(peft_model_id, use_auth_token=huggingface_user_token) 
    elif input_arg.get("checkpoint", None) is None:
        model = load_peft_model_from_hub(peft_model_id)
    
    ###################
    #     Evaluate    #
    ###################
    eval_dataloader = DataLoader(data_test, batch_size=int(input_arg["batch"]), collate_fn=data_collator)
    
    model.eval()
    model = model.to("cuda")
    label_list = []
    pred_list = []
    pred_results = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    # input_features=batch["input_features"],
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    # decoder_input_ids=batch["labels"][:, :4],
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            pred_str = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
            pred_results += pred_result

            pred_list += pred_str
            label_list += label_str

            if step == 0:
                print(pred_result)
        

        del generated_tokens, labels, batch
        gc.collect()

    nlp2.write_csv(pred_results, f'pred_{time}_epoch_{input_arg["specified_epoch"]}.csv')
    cer = cer_cal(label_list, pred_list)
    wer = wer_cal(label_list, pred_list)
    print("********* Evaluation Result *********")
    print(f"cer: {cer}, wer: {wer}")
    print("*************************************")

    model.train()
    return model


def main(arg=None):
    input_arg, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    ############
    #  Config  #
    ############
    size = "large-v2"
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    input_arg["custom_set_train"] = "/work/hungyi2022/taiwanese-meta/taiwanese-meta-train.csv" # NOTE: specify your training data here
    input_arg["custom_set_test"] = "/work/hungyi2022/taiwanese-meta/taiwanese-meta-eval.csv" # NOTE: specify the evaluation or testing data
    input_arg["tokenize_config"] = f"openai/whisper-{size}"
    input_arg["model_config"] = f"openai/whisper-{size}"
    input_arg["output_dir"] = f"outputs/{time}"
    input_arg["group_by_length"] = True
    input_arg["cache_dir"] = '/work/hungyi2022/.cache'
    input_arg["load_cache"] = True # NOTE: set this to generate .data file (could cause lots of time)
    input_arg["epoch"] = 1
    dropout = input_arg.get("dropout", 0.0)

    repo_name = f"/work/hungyi2022/peft/{input_arg['model_config']}-TAT-TD" # NOTE: specify where the processed data is located
    
    ############
    #  Model   #
    ############

    processor = WhisperProcessor.from_pretrained(
        input_arg["model_config"], 
        task="transcribe", 
        language="chinese",
        dropout=dropout
        )
    processor.save_pretrained(repo_name)

    audio_feature_key = "input_ids" if size == "large-v2" else inspect.getfullargspec(model.forward).args[1]
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, audio_feature_key=audio_feature_key)

    
    if input_arg.get('checkpoint', None) is not None:
        model = load_peft_model_from_hub(input_arg.get('checkpoint', None))
    else:
        # Load from huggingface checkpoint
        model = load_peft_model_from_hub("EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch") 
        
        # load from base model
        # model = WhisperForConditionalGeneration.from_pretrained(input_arg["model_config"])
        # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        # model = get_peft_model(model, config)
       
    model = model.to("cuda")
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.print_trainable_parameters()

    ############
    #  Dataset #
    ############

    if not input_arg.get("load_cache", False):
        # data set
        dataset = load_dataset(
            "csv", 
            data_files=input_arg["custom_set_train"], 
            cache_dir=input_arg["cache_dir"], 
            # cache_dir=None, 
            )
        dataset = dataset.filter(lambda e: nlp2.is_file_exist(e["path"]))


        data_train = dataset["train"]
        data_train = data_train.map(
            prepare_dataset_whisper,
            num_proc=input_arg["num_proc"],
            fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        )

        if not input_arg.get("only_eval", False):
            data_train = data_train.map(encode_dataset, fn_kwargs={"processor": processor})
            data_train.save_to_disk(f"{repo_name}-train.data")

        # subprocess.run("rm -rf /home/hungyi2022/.cache", shell=True, check=True)

        # data_train = load_from_disk(f"{repo_name}-train.data")

        if "custom_set_test" in input_arg:
            dataset_test = load_dataset(
                "csv", 
                data_files=input_arg["custom_set_test"], 
                cache_dir=input_arg["cache_dir"],
                # cache_dir=None,
            )
            dataset_test = dataset_test.filter(lambda e: nlp2.is_file_exist(e["path"]))
            data_test = dataset_test["train"]
        else:
            dataset = dataset["train"].train_test_split(test_size=0.1)
            data_test = dataset["test"]

        # data_test = data_test.map(
        #     prepare_dataset_whisper,
        #     num_proc=input_arg["num_proc"],
        #     fn_kwargs={"feature_extractor": processor.feature_extractor, "audio_feature_key": audio_feature_key},
        # )
        
        data_test = data_test.map(encode_dataset, fn_kwargs={"processor": processor})
        if not input_arg.get("only_eval", False):
            data_test.save_to_disk(f"{repo_name}-test.data")

    else:
        print("Start loading cache dataset")
        data_train = load_from_disk(f"{repo_name}-train.data")
        data_test = load_from_disk(f"{repo_name}-eval.data")

    def compute_metrics(pred):
        # print(pred.shape)
        pred_ids = pred.predictions
        pred_ids = [i[i != -100] for i in pred_ids]
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
        cer = cer_cal(label_str, pred_str)
        wer = wer_cal(label_str, pred_str)
        pred_result = [[l, p, cer_cal([l], [p])] for l, p in zip(label_str, pred_str)]
        nlp2.write_csv(pred_result, f"pred_{time}.csv")
        # print 10 predict result randomly for debug
        random.shuffle(pred_result)
        return {"cer": cer, "wer": wer}

    for i in range(input_arg['total_epoch']):
        input_arg['specified_epoch'] = i
        print("===============================")
        print(f"input_arg['specified_epoch']: {input_arg['specified_epoch']}")
        print("===============================")
        model = experiment(input_arg, model, processor, data_collator, repo_name, data_train, data_test, time)

    
if __name__ == "__main__":
    main()
