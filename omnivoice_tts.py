import logging
import torch
from omnivoice_simplified import load_model, generate_sentence


class OmniVoiceTTS:
    """Wrapper class for OmniVoice text-to-speech"""
    
    def __init__(self, device_map: str = "auto", dtype=torch.float16):
        """
        Initialize OmniVoice TTS
        
        Args:
            device_map: Device mapping ("auto", "cpu", or specific GPU)
            dtype: Data type (torch.float16 or torch.float32)
        """
        self.device_map = device_map
        self.dtype = dtype
        self.model = None
        
    def load(self):
        """Load OmniVoice model"""
        if self.model is None:
            logging.info("Loading OmniVoice model...")
            self.model = load_model(device_map=self.device_map, dtype=self.dtype)
            logging.info("OmniVoice model loaded successfully!")
        return self
    
    def synthesize(
        self,
        text: str,
        ref_audio: str,
        output_path: str,
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
    ):
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio file (voice clone)
            output_path: Output file path
            model_id: Model ID from HuggingFace (default: "k2-fsa/OmniVoice")
            ref_text: Transcription of reference audio. If None, Whisper ASR will auto-transcribe.
            num_step: Number of iterative unmasking steps (32 for quality, 16 for speed)
            guidance_scale: Classifier-free guidance scale
            speed: Speed factor (>1.0 = faster, <1.0 = slower)
            duration: Fixed output duration in seconds. Overrides speed when set.
            denoise: Whether to produce cleaner speech
            t_shift: Time-step shift for noise schedule
            position_temperature: Temperature for mask-position selection
            class_temperature: Temperature for token sampling
            layer_penalty_factor: Penalty for deeper codebook layers
            preprocess_prompt: Whether to preprocess reference audio
            postprocess_output: Whether to remove long silences
            audio_chunk_duration: Target chunk duration for long-form generation
            audio_chunk_threshold: Estimated duration threshold to activate chunking
        
        Returns:
            str: Output file path
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first!")
        
        return generate_sentence(
            save_path=output_path,
            text=text,
            ref_audio=ref_audio,
            model=self.model,
            model_id=model_id,
            ref_text=ref_text,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
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
