import os
import logging
import os
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket
from transcriber import transcribe
import json
from inference import run_inference
app = FastAPI()


app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

@app.websocket("/ws/audio_socket")
async def get_action(socket: WebSocket):
    await socket.accept()

    audio_bytes = await socket.receive_bytes()
    print(f"Recebido {len(audio_bytes)} bytes")

    texto = transcribe(
        audio_bytes
    )
    print(texto)

    output = run_inference(input=texto)
    print(output)
    await socket.send_text(json.dumps(output))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    port = int(os.getenv('PORT', '8080'))
    host = '0.0.0.0'

    uvicorn.run(
        'action_api:app',
        host=host,
        port=port,
        reload=False,
        log_level="debug",
        access_log=False,
        use_colors=False,
        loop="uvloop",
        http="httptools",
    )



