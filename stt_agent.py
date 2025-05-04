from agents import Agent

voice_system_prompt = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""

stt_model = Agent(
    name="STT Model",
    model="gpt-4o-transcribe",
    instructions="""
    You will speak only in english.
    """ + voice_system_prompt
)