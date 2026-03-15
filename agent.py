"""
Internet Browsing Agent powered by Claude

This agent can search the web, fetch pages, and answer questions
using real-time internet data.

Usage:
    python agent.py "What's the latest news about AI?"
    python agent.py  # interactive mode
"""

import sys
import anyio
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage, TextBlock

load_dotenv()

SYSTEM_PROMPT = """You are a helpful internet browsing assistant. You can:
- Search the web for current information
- Fetch and read web pages
- Summarize articles and web content
- Answer questions using up-to-date information from the internet

Always cite your sources by mentioning the URLs you visited.
Be concise but thorough in your responses."""


async def browse(prompt: str) -> str:
    """Run the browsing agent with a given prompt and return the result."""
    result_text = ""

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            system_prompt=SYSTEM_PROMPT,
            model="claude-opus-4-6",
            max_turns=10,
        ),
    ):
        if isinstance(message, ResultMessage):
            result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

    return result_text


async def interactive_mode():
    """Run the agent in interactive mode."""
    print("Internet Browsing Agent (powered by Claude)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\nAgent: ", end="", flush=True)
        result = await browse(user_input)
        if result and not result.endswith("\n"):
            print()
        print()


async def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"Browsing for: {prompt}\n")
        result = await browse(prompt)
        if result:
            print(f"\nResult: {result}")
    else:
        await interactive_mode()


if __name__ == "__main__":
    anyio.run(main)
