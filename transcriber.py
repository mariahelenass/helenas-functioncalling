import io
import torch
import numpy as np
from pydub import AudioSegment
import nemo.collections.asr as nemo_asr


model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)
model.eval()
model.to("cpu")


def decode_audio(audio_bytes: bytes) -> np.ndarray:

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1)

    audio = audio.set_frame_rate(16000)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    samples /= 32768.0

    return samples


def transcribe(audio_bytes: bytes) -> str:
    audio = decode_audio(audio_bytes)

    if len(audio) < 1600:
        return ""

    with torch.no_grad():
        hypotheses = model.transcribe(
            audio=[audio],
            batch_size=1
        )

    return hypotheses[0].text
