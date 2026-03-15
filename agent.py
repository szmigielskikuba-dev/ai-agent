"""
Internet Browsing Agent powered by Claude

This agent can search the web, fetch pages, and interact with
Facebook Marketplace — finding listings and sending messages to sellers.

Usage:
    python agent.py                          # interactive mode (general browsing)
    python agent.py --marketplace            # interactive mode (marketplace focus)
    python agent.py "search query"           # one-shot general query
    python agent.py --marketplace "query"    # one-shot marketplace query

Facebook Marketplace notes:
  - The agent will open a real browser window via Playwright.
  - You must log into Facebook manually on first run.
  - Your browser session is saved in ./browser-profile/ for future runs.
  - Automated messaging on Facebook may be subject to their Terms of Service.
"""

import os
import sys
import anyio
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage, TextBlock

load_dotenv()

BROWSER_PROFILE_DIR = os.path.join(os.path.dirname(__file__), "browser-profile")

GENERAL_SYSTEM_PROMPT = """You are a helpful internet browsing assistant. You can:
- Search the web for current information
- Fetch and read web pages
- Summarize articles and web content
- Answer questions using up-to-date information from the internet

Always cite your sources by mentioning the URLs you visited.
Be concise but thorough in your responses."""

MARKETPLACE_SYSTEM_PROMPT = """You are a Facebook Marketplace assistant. You help users find listings and contact sellers.

## How to navigate Facebook Marketplace

1. Go to https://www.facebook.com/marketplace
2. If not logged in, wait for the user to log in manually, then proceed.
3. To search listings: use the search bar at the top of the Marketplace page.
4. To filter by location/price/category: use the filter panel on the left side.
5. To view a listing: click on it to open the detail page.
6. To message a seller: click the "Message" button on the listing page, type a message in the chat box, and click Send.

## Your workflow for finding and responding to ads

When the user asks you to find listings:
1. Navigate to Facebook Marketplace.
2. Search for the requested item or category.
3. Browse the results and summarize the most relevant listings (title, price, location, URL).
4. Ask the user which listing(s) they want to respond to.

When the user asks you to send a message to a seller:
1. Open the specific listing URL.
2. Click "Message" to open the chat.
3. Type the user's message (or a message you draft based on their instructions).
4. Confirm with the user before sending, then click Send.

## Important rules
- Always confirm with the user before sending any message.
- Do not send messages without explicit user approval.
- If you encounter a CAPTCHA or login prompt, pause and ask the user to complete it.
- Report the listing title, price, and seller name when summarizing results."""


def get_playwright_mcp_config() -> dict:
    """Return the Playwright MCP server configuration."""
    os.makedirs(BROWSER_PROFILE_DIR, exist_ok=True)
    return {
        "command": "npx",
        "args": [
            "@playwright/mcp@latest",
            "--user-data-dir", BROWSER_PROFILE_DIR,
            "--browser", "chromium",
        ],
    }


async def run_agent(prompt: str, marketplace_mode: bool = False) -> str:
    """Run the browsing agent and return the result."""
    result_text = ""

    if marketplace_mode:
        system_prompt = MARKETPLACE_SYSTEM_PROMPT
        mcp_servers = {"playwright": get_playwright_mcp_config()}
        allowed_tools = []  # use all MCP tools from playwright
    else:
        system_prompt = GENERAL_SYSTEM_PROMPT
        mcp_servers = {}
        allowed_tools = ["WebSearch", "WebFetch"]

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=allowed_tools if allowed_tools else None,
            mcp_servers=mcp_servers,
            system_prompt=system_prompt,
            model="claude-opus-4-6",
            max_turns=20,
        ),
    ):
        if isinstance(message, ResultMessage):
            result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

    return result_text


async def interactive_mode(marketplace_mode: bool = False):
    """Run the agent in interactive mode."""
    if marketplace_mode:
        print("Facebook Marketplace Agent (powered by Claude + Playwright)")
        print("A browser window will open. Log into Facebook if prompted.")
    else:
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
        result = await run_agent(user_input, marketplace_mode=marketplace_mode)
        if result and not result.endswith("\n"):
            print()
        print()


async def main():
    args = sys.argv[1:]
    marketplace_mode = "--marketplace" in args
    if marketplace_mode:
        args = [a for a in args if a != "--marketplace"]

    if args:
        prompt = " ".join(args)
        mode_label = "Marketplace" if marketplace_mode else "Browsing"
        print(f"{mode_label} search: {prompt}\n")
        result = await run_agent(prompt, marketplace_mode=marketplace_mode)
        if result:
            print(f"\nResult: {result}")
    else:
        await interactive_mode(marketplace_mode=marketplace_mode)


if __name__ == "__main__":
    anyio.run(main)
