from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import struct
from orpheus_tts import OrpheusModel
from typing import Iterator

app = FastAPI()
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


@app.get("/tts")
async def tts(
    prompt: str = "Hey there, looks like you forgot to provide a prompt!",
) -> StreamingResponse:
    def generate_audio_stream() -> Iterator[bytes]:
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

    return StreamingResponse(generate_audio_stream(), media_type="audio/wav")
