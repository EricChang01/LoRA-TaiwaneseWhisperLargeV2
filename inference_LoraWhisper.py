
import torch
# import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, PeftConfig
import soundfile as sf
import torchaudio

# Define the path to your audio file
audio_file = "audio1.wav"

# Load the audio file using soundfile
waveform, sample_rate = sf.read(audio_file)

# Convert the audio data to mono if it has multiple channels
if len(waveform.shape) > 1:
    waveform = waveform.mean(dim=1)

# Resample the audio if required
# desired_sample_rate = sample_rate  # Replace with your desired sample rate
# if sample_rate != desired_sample_rate:
#     resampler = torchaudio.transforms.Resample(sample_rate, desired_sample_rate)
#     waveform = resampler(waveform)
#     sample_rate = desired_sample_rate

# Normalize the waveform if needed
normalized_waveform = waveform # torchaudio.transforms.Normalize()(waveform)

peft_model_id = "cathyi/openai-whisper-large-v2-Lora"
peft_config = PeftConfig.from_pretrained(peft_model_id)
# model = WhisperForConditionalGeneration.from_pretrained(
#     peft_config.base_model_name_or_path
# )
# model = PeftModel.from_pretrained(model, peft_model_id) 

# audio_dataset = Dataset.from_dict({"audio": ["audio1.wav"]}).cast_column("audio", Audio())
# print(audio_dataset[0]["audio"])

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

input_features = processor(normalized_waveform, sampling_rate=sample_rate, return_tensors="pt")

print(f"input features {input_features}")
# Extract the input tensor and length for the ASR pipeline
# input_tensor = input_features.input_values.squeeze(0)
# input_length = input_features.input_lengths.squeeze(0)

# Print the shape of the input tensor and length
# print("Input tensor shape:", input_tensor.shape)
# print("Input length:", input_length)

forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


# def transcribe(audio):
#     with torch.cuda.amp.autocast():
#         text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
#     return text

with torch.cuda.amp.autocast():
    text = pipe(audio_file, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]

print(f"Text: {text}")

# eval_dataloader = DataLoader(data_test, batch_size=int(input_arg["batch"]), collate_fn=data_collator)
# print("Load model from hub successfully.")
