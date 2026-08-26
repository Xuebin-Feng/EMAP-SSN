# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
commands/agent.py — CLI portal and registration stub for the SSN Viewer LLM agent.

All backend logic lives in web_ui/agent_backend.py.
"""

from web_ui.agent_backend import (
    activate_agent_from_card,
    activate_agent,
    deactivate_agent,
    run_web_agent_query,
    register,
    call_api,
)

def print_help():
    print("""
    LLM Agent CLI Portal
    ====================
    Usage: agent
           agent <Model Custom Name>
           agent off
           agent deactivate
           agent <message>
           agent help

    Description:
      Provides a CLI portal to interact with the LLM Agent. Allows registering the Web UI, 
      activating/deactivating models, and sending direct natural language queries.

    Arguments:
      agent                     - Opens the Agent Web UI in the default browser.
      agent <Model Custom Name> - Activates the agent and loads the specified model custom name 
                                  enclosed in <> brackets (e.g. agent <Gemini API>).
      agent off / deactivate    - Deactivates the LLM agent and unloads the active model.
      agent <message>           - Forwards a query message directly to the agent (e.g. agent "make cluster 1 red").
                                  The agent will automatically load your first model card if not already active.

    Examples:
      agent
      agent <Ollama (Local)>
      agent <Gemini API>
      agent off
      agent "select all nodes with length > 500"
      agent hide noise clusters
    """)

def run(viewer, args):
    """Called by SSN_Viewer at startup (--register-only) and when user types 'agent'."""
    register(viewer)

    if args and args[0] == "--register-only":
        return

    # Help check
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # 1. Calling 'agent' alone: Open browser UI (current behavior)
    if not args:
        viewer.open_agent_ui()
        return

    import Command_Engine
    full_arg = " ".join(args).strip()

    # 2. Deactivate check
    if full_arg.lower() in ["off", "deactivate"]:
        deactivate_agent(viewer)
        return

    # Helper to check if a string matches any loaded model card custom name
    from web_ui.agent_backend import load_model_cards
    cards = load_model_cards()

    def find_matching_card(custom_name):
        name_clean = custom_name.strip().lower()
        for card in cards:
            if (card.get("name") or "").lower() == name_clean:
                return card
        return None

    # 3. Model spec check: strictly must start with < and end with >
    is_model_spec = full_arg.startswith("<") and full_arg.endswith(">")

    if is_model_spec:
        model_custom_name = full_arg[1:-1].strip()
        card = find_matching_card(model_custom_name)
        if card:
            activate_agent_from_card(viewer, card)
        else:
            available_names = ", ".join([f"<{c.get('name')}>" for c in cards if c.get("name")])
            Command_Engine.print_help(viewer, f"Error: Model Custom Name '{model_custom_name}' not found in configured model cards.\nAvailable model names: {available_names}")
        return

    # 4. Message forwarding: strip quotes if present
    message = full_arg
    if (message.startswith('"') and message.endswith('"')) or (message.startswith("'") and message.endswith("'")):
        message = message[1:-1].strip()

    # Auto-activate first card if agent isn't turned on
    if not getattr(viewer, "llm_loaded", False):
        if cards:
            activate_agent_from_card(viewer, cards[0], quiet=True)
        else:
            Command_Engine.print_help(viewer, "Error: LLM agent is not loaded. Please configure a model in the Agent UI first.")
            return

    # Forward the message to the agent backend
    run_web_agent_query(viewer, message)
