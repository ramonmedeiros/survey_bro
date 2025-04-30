from agents.voice import TTSModelSettings, VoicePipeline, VoicePipelineConfig, SingleAgentVoiceWorkflow, AudioInput
import sounddevice as sd
import numpy as np
import asyncio

from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

set_default_openai_key("YOUR_API_KEY")

@function_tool
def get_account_info(user_id: str) -> dict:
    """Return dummy account info for a given user."""
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive"
    }


# Common system prompt for voice output best practices:
voice_system_prompt = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""

# --- Agent: Search Agent ---
search_voice_agent = Agent(
    name="SearchVoiceAgent",
    instructions=voice_system_prompt + (
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)

# --- Agent: Knowledge Agent ---
knowledge_voice_agent = Agent(
    name="KnowledgeVoiceAgent",
    instructions=voice_system_prompt + (
        "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
    ),
    tools=[FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
        ),],
)

# --- Agent: Account Agent ---
account_voice_agent = Agent(
    name="AccountVoiceAgent",
    instructions=voice_system_prompt + (
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],
)

# --- Agent: Triage Agent ---
triage_voice_agent = Agent(
    name="VoiceAssistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Acme Shop. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- KnowledgeAgent for product FAQs
- SearchAgent for anything requiring real-time web search
"""),
    handoffs=[account_voice_agent, knowledge_voice_agent, search_voice_agent],
)

# Define custom TTS model settings with the desired instructions
custom_tts_settings = TTSModelSettings(
    instructions="Personality: upbeat, friendly, persuasive guide"
    "Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable."
    "Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow."
    "Tempo: Speak relatively fast, include brief pauses and after before questions"
    "Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey."
)

async def voice_assistant_optimized():
    samplerate = sd.query_devices(kind='input')['default_samplerate']
    voice_pipeline_config = VoicePipelineConfig(tts_settings=custom_tts_settings)

    while True:
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_voice_agent), config=voice_pipeline_config)

        # Check for input to either provide voice or exit
        cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
        if cmd.lower() == "esc":
            print("Exiting...")
            break       
        print("Listening...")
        recorded_chunks = []

         # Start streaming from microphone until Enter is pressed
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())):
            input()

        # Concatenate chunks into single buffer
        recording = np.concatenate(recorded_chunks, axis=0)

        # Input the buffer and await the result
        audio_input = AudioInput(buffer=recording)

        with trace("ACME App Optimized Voice Assistant"):
            result = await pipeline.run(audio_input)

         # Transfer the streamed result into chunks of audio
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)
        response_audio = np.concatenate(response_chunks, axis=0)

        # Play response
        print("Assistant is responding...")
        sd.play(response_audio, samplerate=samplerate)
        sd.wait()
        print("---")

def main():
    # Run the voice assistant
    asyncio.run(voice_assistant_optimized())

if __name__ == "__main__":
    main()

