"""
Internet Browsing Agent powered by Claude

This agent can search the web, fetch pages, interact with Facebook Marketplace,
and sign up to gyms by submitting your contact details on their websites.

Usage:
    python agent.py                          # interactive mode (general browsing)
    python agent.py --marketplace            # interactive mode (marketplace focus)
    python agent.py --gym-signup             # interactive mode (gym sign-up focus)
    python agent.py "search query"           # one-shot general query
    python agent.py --marketplace "query"    # one-shot marketplace query
    python agent.py --gym-signup "gyms in Austin TX"  # find & sign up to gyms

Gym sign-up notes:
  - Set GYM_CONTACT_NAME and GYM_CONTACT_PHONE in your .env file.
  - A browser window will open via Playwright to fill out contact forms.
  - The agent will show you each form before submitting and ask for confirmation.

Facebook Marketplace notes:
  - The agent will open a real browser window via Playwright.
  - You must log into Facebook manually on first run.
  - Your browser session is saved in ./browser-profile/ for future runs.
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

GYM_SIGNUP_SYSTEM_PROMPT = """You are a gym sign-up assistant. You help users register their contact details on gym websites so gyms can call them back.

## Your workflow

1. **Find gyms**: Search Google for gyms matching the user's query (e.g. "CrossFit gyms in Austin TX").
   Collect a list of gym names and their website URLs.

2. **For each gym website**:
   a. Navigate to the gym's website.
   b. Look for a contact/sign-up/free trial/callback request form. Common locations:
      - A "Free Trial", "Join Now", or "Get Started" button
      - A "Contact Us" page
      - A pop-up or banner asking for contact details
      - A footer form
   c. Fill in the user's name and phone number in the appropriate fields.
      Leave any fields you're unsure about empty rather than guessing.
   d. **Show the user exactly what you are about to submit** and ask for confirmation.
   e. Only submit after the user says yes.

3. **Report results**: After each gym, report whether the sign-up succeeded or failed (e.g. no form found, CAPTCHA, form error).

## Rules
- Never submit a form without explicit user confirmation.
- Never fill in fields other than name and phone (e.g. do not enter payment info).
- If a CAPTCHA appears, pause and ask the user to solve it.
- If you cannot find a contact form after checking the homepage and "Contact" / "Join" pages, skip that gym and note it in your report.
- Do not create accounts or set passwords on behalf of the user."""

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


async def run_agent(prompt: str, marketplace_mode: bool = False, gym_signup_mode: bool = False) -> str:
    """Run the browsing agent and return the result."""
    result_text = ""

    if gym_signup_mode:
        name = os.getenv("GYM_CONTACT_NAME", "")
        phone = os.getenv("GYM_CONTACT_PHONE", "")
        if not name or not phone:
            print("Error: Set GYM_CONTACT_NAME and GYM_CONTACT_PHONE in your .env file.")
            return ""
        contact_info = f"Name: {name}\nPhone: {phone}"
        system_prompt = GYM_SIGNUP_SYSTEM_PROMPT + f"\n\n## Contact details to submit\n{contact_info}"
        mcp_servers = {"playwright": get_playwright_mcp_config()}
        allowed_tools = None
    elif marketplace_mode:
        system_prompt = MARKETPLACE_SYSTEM_PROMPT
        mcp_servers = {"playwright": get_playwright_mcp_config()}
        allowed_tools = None
    else:
        system_prompt = GENERAL_SYSTEM_PROMPT
        mcp_servers = {}
        allowed_tools = ["WebSearch", "WebFetch"]

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            system_prompt=system_prompt,
            model="claude-opus-4-6",
            max_turns=30,
        ),
    ):
        if isinstance(message, ResultMessage):
            result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

    return result_text


async def interactive_mode(marketplace_mode: bool = False, gym_signup_mode: bool = False):
    """Run the agent in interactive mode."""
    if gym_signup_mode:
        print("Gym Sign-Up Agent (powered by Claude + Playwright)")
        print("Make sure GYM_CONTACT_NAME and GYM_CONTACT_PHONE are set in .env")
        print("A browser window will open to fill in your contact details.")
    elif marketplace_mode:
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
        result = await run_agent(user_input, marketplace_mode=marketplace_mode, gym_signup_mode=gym_signup_mode)
        if result and not result.endswith("\n"):
            print()
        print()


async def main():
    args = sys.argv[1:]
    marketplace_mode = "--marketplace" in args
    gym_signup_mode = "--gym-signup" in args
    args = [a for a in args if a not in ("--marketplace", "--gym-signup")]

    if args:
        prompt = " ".join(args)
        if gym_signup_mode:
            mode_label = "Gym sign-up"
        elif marketplace_mode:
            mode_label = "Marketplace"
        else:
            mode_label = "Browsing"
        print(f"{mode_label}: {prompt}\n")
        result = await run_agent(prompt, marketplace_mode=marketplace_mode, gym_signup_mode=gym_signup_mode)
        if result:
            print(f"\nResult: {result}")
    else:
        await interactive_mode(marketplace_mode=marketplace_mode, gym_signup_mode=gym_signup_mode)


if __name__ == "__main__":
    anyio.run(main)
