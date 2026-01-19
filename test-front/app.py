import streamlit as st
import asyncio
import websockets
from audiorecorder import audiorecorder

st.set_page_config(page_title="Transcrição de Áudio", layout="centered")

st.title("🎙️ Teste Front")
st.write("interface de teste.")

WS_URL = "ws://inference:8080/ws/audio_socket"


async def send_audio_and_get_transcription(audio_bytes: bytes) -> str:
    async with websockets.connect(
        WS_URL,
        max_size=10 * 1024 * 1024
    ) as websocket:
        await websocket.send(audio_bytes)
        transcription = await websocket.recv()
        return transcription

audio = audiorecorder(
    "▶️ Gravar",
    "⏹️ Parar",
)

if len(audio) > 0:
    st.audio(audio.export().read())

    if st.button("📤 Enviar para transcrição"):
        with st.spinner("Transcrevendo áudio..."):
            audio_bytes = audio.export().read()

            try:
                texto = asyncio.run(
                    send_audio_and_get_transcription(audio_bytes)
                )

                st.success("Transcrição concluída!")
                st.text_area(
                    "📝 Texto transcrito",
                    texto,
                    height=200
                )

            except Exception as e:
                st.error(f"Erro ao transcrever: {e}")

