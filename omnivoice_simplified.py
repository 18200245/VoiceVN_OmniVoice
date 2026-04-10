#!/usr/bin/env python3

import logging
import torch
import torchaudio
from pathlib import Path
from omnivoice import OmniVoice


def load_model(model_id: str = "k2-fsa/OmniVoice", device_map: str = "auto", dtype=torch.float16):
    """
    Load OmniVoice model
    
    Args:
        model_id: Model ID from HuggingFace (default: "k2-fsa/OmniVoice")
        device_map: Device mapping ("auto", "cpu", or specific GPU)
        dtype: Data type (torch.float16 or torch.float32)
    
    Returns:
        model: OmniVoice model instance
    """
    logging.info(f"Loading OmniVoice model: {model_id} (device_map={device_map})...")
    
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype
    )
    
    model.eval()
    logging.info("Model loaded successfully!")
    return model


@torch.inference_mode()
def generate_sentence(
    save_path: str,
    text: str,
    ref_audio: str,
    model: OmniVoice = None,
    model_id: str = "k2-fsa/OmniVoice",
    ref_text: str = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    speed: float = 1.0,
    duration: float = None,
    denoise: bool = True,
    t_shift: float = 0.1,
    position_temperature: float = 5.0,
    class_temperature: float = 0.0,
    layer_penalty_factor: float = 5.0,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    audio_chunk_duration: float = 15.0,
    audio_chunk_threshold: float = 30.0,
    sampling_rate: int = 24000,
):
    """
    Generate speech from text using OmniVoice
    
    Args:
        save_path: Output file path for audio
        text: Text to synthesize
        ref_audio: Path to reference audio file
        model: OmniVoice model instance
        model_id: Model ID from HuggingFace (used if model is None)
        ref_text: Transcription of reference audio. If None, Whisper ASR will auto-transcribe
        num_step: Number of iterative unmasking steps (default 32, use 16 for faster inference)
        guidance_scale: Classifier-free guidance scale (default 2.0)
        speed: Speed factor (>1.0 = faster, <1.0 = slower). Ignored if duration is set.
        duration: Fixed output duration in seconds. Overrides speed when set.
        denoise: Whether to add <|denoise|> token for cleaner speech
        t_shift: Time-step shift for noise schedule (smaller = emphasise earlier steps)
        position_temperature: Temperature for mask-position selection (0 = greedy/deterministic)
        class_temperature: Temperature for token sampling (0 = greedy/deterministic)
        layer_penalty_factor: Penalty for deeper codebook layers
        preprocess_prompt: Whether to preprocess reference audio
        postprocess_output: Whether to remove long silences from output
        audio_chunk_duration: Target chunk duration for long-form generation
        audio_chunk_threshold: Estimated duration threshold to activate chunking
        sampling_rate: Output sampling rate (default 24000)
    """
    if model is None:
        raise ValueError("Model is required. Call load_model() first.")
    
    logging.info(f"Generating speech for: {text[:50]}...")
    
    # Generate audio using OmniVoice
    audio_list = model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed if duration is None else None,
        duration=duration,
        denoise=denoise,
        t_shift=t_shift,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        layer_penalty_factor=layer_penalty_factor,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )
    
    # audio_list is a list of torch.Tensor with shape (1, T) at 24 kHz
    audio = audio_list[0]
    
    # Save audio file
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, sampling_rate)
    
    logging.info(f"Audio saved to: {save_path}")
    return save_path
