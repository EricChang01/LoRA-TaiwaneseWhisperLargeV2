from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "openai/whisper-large-v2"
tokenizer_name_or_path = "openai/whisper-large-v2"



model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

print(model)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282