import runpod
import struct
from orpheus_tts import OrpheusModel
from typing import Iterator

# Initialize the TTS engine
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")


def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def generate_audio_stream(prompt: str) -> Iterator[bytes]:
    yield create_wav_header()
    syn_tokens = engine.generate_speech(
        prompt=prompt,
        voice="tara",
        repetition_penalty=1.1,
        stop_token_ids=[128258],
        max_tokens=2000,
        temperature=0.4,
        top_p=0.9,
    )
    for chunk in syn_tokens:
        yield chunk


def generator_handler(job):
    """
    Generator handler for streaming TTS audio data.

    Args:
        job (dict): Contains the input data and request metadata

    Yields:
        dict: Chunks of audio data and metadata
    """
    print("Worker Start")
    input = job["input"]

    prompt = input.get(
        "prompt", "Hey there, looks like you forgot to provide a prompt!"
    )

    print(f"Received prompt: {prompt}")

    # First yield the header
    header = create_wav_header()
    yield {
        "header": header.hex(),
        "prompt": prompt,
        "chunk": 0,
        "total_chunks": -1,  # We don't know the total yet
    }

    # Then yield the audio chunks
    chunk_count = 1
    for audio_chunk in generate_audio_stream(prompt):
        if audio_chunk != header:  # Skip the header as we already sent it
            yield {
                "audio_chunk": audio_chunk.hex(),
                "chunk": chunk_count,
                "total_chunks": -1,
            }
            chunk_count += 1

    # Final yield to indicate completion
    yield {"status": "completed", "total_chunks": chunk_count - 1}


# Start the Serverless function when the script is run
if __name__ == "__main__":
    runpod.serverless.start(
        {"handler": generator_handler, "return_aggregate_stream": True}
    )
