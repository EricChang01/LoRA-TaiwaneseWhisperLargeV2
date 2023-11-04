
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig

# Define the path to your audio file
audio_file = "audio1.wav"

peft_model_id = "cathyi/openai-whisper-large-v2-Lora"
language = "Chinese"
task = "transcribe"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
# device: Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, >=0 will run the model on the associated CUDA device id.
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, device=0)

with torch.cuda.amp.autocast():
    pred = pipe(audio_file, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]

print(f"Prediction: {pred}")
