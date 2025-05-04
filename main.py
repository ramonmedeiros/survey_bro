import sounddevice as sd
import numpy as np
import asyncio
import os

from typing import Any
from agents.voice import VoicePipeline, SingleAgentVoiceWorkflow, AudioInput
from agents import trace, set_default_openai_client, add_trace_processor
from agents.tracing.processors import TracingExporter, BatchTraceProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from openai import AsyncAzureOpenAI
from stt_agent import stt_model
from tts_agent import tts_model
from main_agent import main_agent


class ConsoleSpanExporter(TracingExporter):
    """Prints the traces and spans to the console."""

    def export(self, items: list[Trace | Span[Any]]) -> None:
        for item in items:
            if isinstance(item, Trace):
                print(
                    f"[Exporter] Export trace_id={item.trace_id}, name={item.name}, ")
            elif isinstance(item, Span):
                pass

# Set the default OpenAI client for the Agents SDK
set_default_openai_client(AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
))

# Set up console tracing
console_exporter = ConsoleSpanExporter()
console_processor = BatchTraceProcessor(exporter=console_exporter)
add_trace_processor(console_processor)


# Settings
sample_rate = 44100
block_size = 1024
channels = 1
dtype = 'float32'
silence_threshold = 0.01  # Volume below this is considered silence
silence_duration = 2.0  # Seconds of silence before stopping

audio_data = []
silent_blocks = 0
max_silent_blocks = int((silence_duration * sample_rate) / block_size)

def callback(indata, frames, time_info, status):
    global silent_blocks, audio_data

    volume_norm = np.linalg.norm(indata)
    audio_data.append(indata.copy())

    if volume_norm < silence_threshold:
        silent_blocks += 1
    else:
        silent_blocks = 0  # Reset on noise

    if silent_blocks >= max_silent_blocks:
        raise sd.CallbackStop()

# Common system prompt for voice output best practices:
async def main():
    devices = sd.query_devices(kind='input')
    samplerate = devices['default_samplerate']

    pipeline = VoicePipeline(
                stt_model=stt_model,
                tts_model=tts_model,
                workflow=SingleAgentVoiceWorkflow(main_agent),
            )

    while True:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback):
            try:
                sd.sleep(int(5 * 1000))  # Max 5 seconds
            except sd.CallbackStop:
                pass

        # Concatenate chunks into single buffer
        recording = np.concatenate(audio_data, axis=0)

        # Input the buffer and await the result
        audio_input = AudioInput(buffer=recording)

        with trace("ACME App Voice Assistant"):
            result = await pipeline.run(audio_input)

            # Play the audio stream as it comes in
            response_chunks = []
            async for event in result.stream():
                match event.type:
                    case "voice_stream_event_audio":
                        response_chunks.append(event.data)
                    case "voice_stream_event_lifecycle":
                        print(f"lifecycle: {event}")
                    case "voice_stream_event_error":
                        print(f"error: {event}")
        response_audio = np.concatenate(response_chunks, axis=0)
        sd.play(response_audio, samplerate=samplerate/2)
        sd.wait()

if __name__ == "__main__":
    asyncio.run(main())
