# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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
commands/agent.py — CLI portal and registration stub for the EMAP-SSN Viewer LLM agent.

All backend logic lives in web_ui/agent_backend.py.
"""

import Command_Engine
from utilities.Localization import JoinedMessage, Message

from web_ui.agent_backend import (
    activate_agent_from_card,
    activate_agent,
    deactivate_agent,
    run_web_agent_query,
    broadcast_backend_state,
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

def _load_saved_cards(viewer):
    """Return the saved model cards, or None after reporting an unusable model_card.json."""
    from web_ui.agent_backend import ModelCardsError, load_model_cards
    try:
        return load_model_cards()
    except ModelCardsError as error:
        message = Message("Error: {error}", error=error)
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return None

def _open_agent_page(viewer):
    """Open the Agent page; return whether it opened.

    A command run for the MCP or the agent page cannot wait for someone to
    close the dialog that says the page is already open.
    """
    from Viewer_Command_Portal import CURRENT
    if CURRENT.get() is None:
        return viewer.open_agent_ui()
    return viewer.open_agent_ui(show_existing_dialog=False)


def _agent_page_connected(viewer):
    has_event_client = getattr(getattr(viewer, "web_server", None), "has_event_client", None)
    return bool(callable(has_event_client) and has_event_client("agent"))


def run(viewer, args):
    """Called by EMAPSSN_Viewer at startup (--register-only) and when user types 'agent'."""
    register(viewer)

    if args and args[0] == "--register-only":
        Command_Engine.command_succeeded(viewer, 'Registered the Agent interface.')
        return

    # Help check: a message may start with the word "help"
    if len(args) == 1 and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # 1. Calling 'agent' alone: Open browser UI (current behavior)
    if not args:
        if _open_agent_page(viewer):
            Command_Engine.command_succeeded(viewer, "Opened the Agent interface; no model request was started.")
        elif _agent_page_connected(viewer):
            Command_Engine.command_succeeded(viewer, "The Agent interface is already open; no model request was started.")
        else:
            Command_Engine.command_failed(viewer, "The Agent interface was not opened; the Viewer's console line says why.")
        return

    full_arg = " ".join(args).strip()

    # 2. Deactivate check
    if full_arg.lower() in ["off", "deactivate"]:
        if deactivate_agent(viewer):
            broadcast_backend_state(viewer)
            Command_Engine.command_succeeded(viewer, 'Agent deactivated.')
        else:
            Command_Engine.command_succeeded(viewer, 'Agent is already inactive.')
        return

    # Helper to check if a string matches any loaded model card custom name
    def find_matching_card(custom_name):
        name_clean = custom_name.strip().lower()
        for card in cards:
            if (card.get("name") or "").lower() == name_clean:
                return card
        return None

    # 3. Model spec check: strictly must start with < and end with >
    is_model_spec = full_arg.startswith("<") and full_arg.endswith(">")

    if is_model_spec:
        cards = _load_saved_cards(viewer)
        if cards is None:
            return
        model_custom_name = full_arg[1:-1].strip()
        card = find_matching_card(model_custom_name)
        if card:
            if not activate_agent_from_card(viewer, card):
                Command_Engine.command_failed(viewer, "Agent backend activation failed")
            else:
                broadcast_backend_state(viewer)
                Command_Engine.command_succeeded(viewer, f"Activated Agent model {model_custom_name!r}.")
        else:
            available_names = ", ".join([f"<{c.get('name')}>" for c in cards if c.get("name")])
            # The console line shows the first line; the list of names is for the terminal.
            message = JoinedMessage([
                Message(
                    "Error: Model Custom Name '{name}' not found in configured model cards.",
                    name=model_custom_name,
                ),
                f"Available model names: {available_names}",
            ], separator="\n")
            Command_Engine.print_help(viewer, message)
            Command_Engine.command_failed(viewer, message)
        return

    # 4. Message forwarding: strip quotes if present
    message = full_arg
    if (message.startswith('"') and message.endswith('"')) or (message.startswith("'") and message.endswith("'")):
        message = message[1:-1].strip()

    # Auto-activate first card if agent isn't turned on
    if not getattr(viewer, "llm_loaded", False):
        cards = _load_saved_cards(viewer)
        if cards is None:
            return
        if cards:
            if not activate_agent_from_card(viewer, cards[0], quiet=True):
                Command_Engine.command_failed(viewer, "Agent backend activation failed")
                return
            broadcast_backend_state(viewer)
        else:
            message = Message("Error: LLM agent is not loaded. Please configure a model in the Agent UI first.")
            Command_Engine.print_help(viewer, message)
            Command_Engine.command_failed(viewer, message)
            return

    # Forward the message to the agent backend
    rejections = []
    if not run_web_agent_query(viewer, message, on_rejected=rejections.append):
        reason = rejections[0] if rejections else Message("The Agent backend did not accept the message.")
        failure = Message("Error: {error}", error=reason)
        Command_Engine.print_help(viewer, failure)
        Command_Engine.command_failed(viewer, failure)
        return
    Command_Engine.command_succeeded(viewer, 'Submitted the message to the Agent backend.')
