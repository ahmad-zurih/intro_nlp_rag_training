import os
import sys
import json
import re
import asyncio
import uuid
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# --- Configuration ---
MODEL = "gpt-oss-120b"  # GPUStack model
GPUSTACK_BASE_URL = "https://gpustack.unibe.ch/v1"
OPENAI_MODEL_FALLBACK = "gpt-4o"  # optional fallback if you flip PROVIDER below
PROVIDER = "gpustack"

SYSTEM_PROMPT = (
    "You have access to tools.\n"
    "If a tool is needed, FIRST respond with a single JSON object on one line:\n"
    '{"name": "<tool_name>", "arguments": { ... }}\n'
    "Do not add extra text before or after the JSON. "
    "If no tool is required, answer normally.\n"
)

def load_api_key(filepath: str, prefix: str) -> str | None:
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                return line
    except FileNotFoundError:
        return None
    return None

def get_client_and_model():
    if PROVIDER == "gpustack":
        api_key = load_api_key("../api-key", "gpustack")
        if not api_key:
            print("ERROR: Could not find GPUStack API key.")
            print("Create 'gpustack-api' with a line: gpustack=YOUR_GPUSTACK_API_KEY")
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key, base_url=GPUSTACK_BASE_URL)
        return client, MODEL
    else:
        api_key = load_api_key("../api-key", "openai")
        if not api_key:
            print("ERROR: Could not find OpenAI API key.")
            print("Create 'api-key' with a line: openai=sk-...")
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key)
        return client, OPENAI_MODEL_FALLBACK

async def get_mcp_tools(session: ClientSession) -> list:
    print("--- Client: Fetching tools from MCP server... ---")
    tool_list_response = await session.list_tools()
    openai_tools = []
    for tool in tool_list_response.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,  # JSON Schema
            },
        })
    print(f"--- Client: Loaded {len(openai_tools)} tools. ---")
    return openai_tools

# ---- Fallback parser for textual tool calls ---------------------------------
# It looks for a JSON object with "name" and "arguments".
_TOOL_JSON_RE = re.compile(
    r'(?P<json>\{\s*"name"\s*:\s*"(.*?)"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})',
    re.DOTALL,
)

def try_parse_tool_call_from_text(text: str):
    if not text:
        return None
    # Remove common wrappers like </tool_call>, code fences, etc.
    cleaned = text.strip()
    cleaned = cleaned.replace("</tool_call>", "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    m = _TOOL_JSON_RE.search(cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group("json"))
        name = obj.get("name")
        args = obj.get("arguments") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args}
    except Exception:
        return None
    return None
# -----------------------------------------------------------------------------

async def main():
    client, model_name = get_client_and_model()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server_inclass.py"], # Make sure this filename matches your server file
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_mcp_tools(session)

            print("\nMCP Chat Client Ready.")
            print("Provider:", PROVIDER)
            print("Model:", model_name)
            print("Type 'exit' to quit.")

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            while True:
                user_input = input("\n> ")
                if user_input.lower() == "exit":
                    break

                messages.append({"role": "user", "content": user_input})

                # --- START OF AGENT LOOP ---
                # We loop until the model decides to stop calling tools.
                while True:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.7,
                        top_p=0.95,
                    )

                    response_message = response.choices[0].message
                    messages.append(response_message)

                    # 1. Check for native tool calls
                    tool_calls = getattr(response_message, "tool_calls", None)
                    
                    # 2. Check for fallback parsed text (only if native calls missing)
                    parsed = None
                    if not tool_calls:
                        parsed = try_parse_tool_call_from_text(response_message.content)

                    # If no tools required, break the inner loop and wait for user input
                    if not tool_calls and not parsed:
                        print(f"\nAssistant:\n{response_message.content}")
                        break
                    
                    # --- Execute Tools ---
                    
                    # Case A: Native Tool Calls (Handles parallel calls)
                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments or "{}")
                            tool_call_id = tool_call.id

                            print(f"--- Client: Model wants '{tool_name}' with args: {tool_args} ---")
                            
                            try:
                                result = await session.call_tool(tool_name, arguments=tool_args)
                                tool_output = result.content[0].text if result.content else str(result)
                            except Exception as e:
                                tool_output = f"Error executing tool: {str(e)}"

                            print(f"--- Client: Tool output: '{tool_output[:200]}...' ---")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_output,
                            })

                    # Case B: Fallback Manual Parsing (Usually handles one call)
                    elif parsed:
                        tool_name = parsed["name"]
                        tool_args = parsed["arguments"]
                        tool_call_id = f"call-{uuid.uuid4()}" # Generate fake ID for fallback

                        print(f"--- Client: Model (Fallback) wants '{tool_name}' with args: {tool_args} ---")
                        
                        try:
                            result = await session.call_tool(tool_name, arguments=tool_args)
                            tool_output = result.content[0].text if result.content else str(result)
                        except Exception as e:
                            tool_output = f"Error executing tool: {str(e)}"
                            
                        print(f"--- Client: Tool output: '{tool_output[:200]}...' ---")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_output,
                        })
                    
                    # The loop now repeats. 
                    # The updated 'messages' list (with tool outputs) is sent back to the LLM 
                    # at the top of the 'while True' loop.

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
