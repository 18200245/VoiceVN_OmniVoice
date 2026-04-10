import json
import logging
from pathlib import Path
from pydub import AudioSegment
from omnivoice_simplified import load_model, generate_sentence

logging.basicConfig(level=logging.INFO)


def process_json_to_chapters(
    json_path: str,
    output_dir: str,
    prompt_wav: str,
    prompt_text: str = None,
    start_chapter: str = None,
    end_chapter: str = None,
    device_map: str = "auto",
    num_step: int = 32,
    guidance_scale: float = 2.0,
):
    """
    Create chapter audio files from JSON and update timestamps with OmniVoice
    
    Args:
        json_path: Path to input JSON file
        output_dir: Directory for output files
        prompt_wav: Path to reference audio file (voice clone)
        prompt_text: Transcription of reference audio. If None, Whisper ASR will auto-transcribe.
        start_chapter: Starting chapter ID (e.g., "0001"), None = from start
        end_chapter: Ending chapter ID (e.g., "0005"), None = to end
        device_map: Device mapping ("auto", "cpu", or specific GPU)
        num_step: Number of generation steps (32 for quality, 16 for speed)
        guidance_scale: Classifier-free guidance scale
    """
    # Load model once
    logging.info("Loading OmniVoice model...")
    model = load_model(device_map=device_map)
    logging.info("Model loaded successfully!")
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by chapter (xxxx từ xxxx_yyyy)
    chapters = {}
    for item in data['data']:
        chapter_id = item['id'].split('_')[0]
        if chapter_id not in chapters:
            chapters[chapter_id] = []
        chapters[chapter_id].append(item)
    
    # Sort chapters
    sorted_chapters = sorted(chapters.items())
    
    # Filter chapters by range
    if start_chapter or end_chapter:
        filtered_chapters = []
        for chapter_id, segments in sorted_chapters:
            # Check if in range
            if start_chapter and chapter_id < start_chapter:
                continue
            if end_chapter and chapter_id > end_chapter:
                continue
            filtered_chapters.append((chapter_id, segments))
        sorted_chapters = filtered_chapters
    
    if not sorted_chapters:
        logging.error("No chapters found in the specified range!")
        return
    
    chapter_ids = [ch[0] for ch in sorted_chapters]
    logging.info(f"Processing {len(sorted_chapters)} chapters: {chapter_ids[0]} to {chapter_ids[-1]}\n")
    
    # Process each chapter
    for chapter_id, segments in sorted_chapters:
        logging.info(f"{'='*60}")
        logging.info(f"Processing Chapter {chapter_id} ({len(segments)} segments)...")
        logging.info(f"{'='*60}")
        
        chapter_audio = AudioSegment.empty()
        current_time = 0  # milliseconds
        
        for idx, segment in enumerate(segments, 1):
            logging.info(f"  [{idx}/{len(segments)}] Generating {segment['id']}...")
            
            # Generate audio
            temp_file = output_dir / f"temp_{segment['id']}.wav"
            generate_sentence(
                save_path=str(temp_file),
                text=segment['text'],
                ref_audio=prompt_wav,
                ref_text=prompt_text,
                model=model,
                num_step=num_step,
                guidance_scale=guidance_scale,
            )
            
            # Load generated audio
            audio_segment = AudioSegment.from_wav(str(temp_file))
            
            # Update timestamps (convert milliseconds to seconds)
            segment['start'] = round(current_time / 1000, 2)
            segment['end'] = round((current_time + len(audio_segment)) / 1000, 2)
            
            # Append to chapter audio
            chapter_audio += audio_segment
            current_time += len(audio_segment)
            
            # Clean up temp file
            temp_file.unlink()
            
            logging.info(f"      Duration: {segment['end'] - segment['start']:.2f}s "
                        f"(Start: {segment['start']:.2f}s, End: {segment['end']:.2f}s)")
        
        # Save chapter audio
        chapter_file = output_dir / f"chapter_{chapter_id}.wav"
        chapter_audio.export(str(chapter_file), format="wav")
        
        total_duration = len(chapter_audio) / 1000
        logging.info(f"\n✓ Saved {chapter_file}")
        logging.info(f"  Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)\n")
    
    # Save updated JSON
    output_json = output_dir / "data_updated.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"{'='*60}")
    logging.info(f"✓ All done! Updated JSON saved to: {output_json}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    # Example 1: Process all chapters
    # process_json_to_chapters(
    #     json_path="/content/data.json",
    #     output_dir="/content/chapters_output",
    #     prompt_wav="/content/1128.MP3",
    #     prompt_text="NĂM HAI NGHÌN KHÔNG TRĂM HAI LĂM...",
    # )
    
    # Example 2: Process only chapters 0001 to 0005
    process_json_to_chapters(
        json_path="/content/data.json",
        output_dir="/content/chapters_output",
        prompt_wav="/content/1128.MP3",
        prompt_text="NĂM HAI NGHÌN KHÔNG TRĂM HAI LĂM LẠI CÓ THÊM MỘT BỘ PHIM SÁT GIỐNG XUẤT SẮC ĐƯỢC RA MẮT DÀN DIỄN VIÊN ĐỀU ĐẾN TỪ CÁC NHÓM NHẠC THÂN TƯỢNG HÀNG ĐẦU TRONG KHU VỰC",
        start_chapter="0001",
        end_chapter="0002",
        device_map="auto",
        num_step=32,
        guidance_scale=2.0,
    )
    
    # Example 3: From chapter 0010 onwards
    # process_json_to_chapters(
    #     json_path="/content/data.json",
    #     output_dir="/content/chapters_output",
    #     prompt_wav="/content/1128.MP3",
    #     prompt_text="NĂM HAI NGHÌN KHÔNG TRĂM HAI LĂM...",
    #     start_chapter="0010",
    # )
    
    # Example 4: Up to chapter 0020
    # process_json_to_chapters(
    #     json_path="/content/data.json",
    #     output_dir="/content/chapters_output",
    #     prompt_wav="/content/1128.MP3",
    #     prompt_text="NĂM HAI NGHÌN KHÔNG TRĂM HAI LĂM...",
    #     end_chapter="0020",
    # )