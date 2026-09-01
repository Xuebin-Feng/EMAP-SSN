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
agent_backend.py — Core LLM agent logic for EMAP-SSN Viewer.

All heavy lifting: model card loading, server activation/deactivation,
API calls, worker threads, viewer context extraction, query execution,
and button registration. Imported by commands/agent.py (thin stub) and
referenced via that stub by Web_Server.py.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import gc
import re

# src/ directory so we can resolve sibling packages regardless of cwd.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import Command_Engine
from PySide6 import QtCore
from web_ui.Plugin_Manager import ensure_registry
from utilities.Cache_Selection import resolve_selected_cache

# ─── Model card helpers ───────────────────────────────────────────────────────

def load_model_cards():
    """Loads model cards from resources/agent/model_card.json."""
    path = os.path.join(_SRC_DIR, "resources", "agent", "model_card.json")
    defaults = [
        {"id": "ollama",   "name": "Ollama (Local)",    "url": "http://localhost:11434/v1", "model": "qwen2.5:1.5b", "api_key": "", "temperature": 0.0},
        {"id": "lmstudio", "name": "LM Studio (Local)", "url": "http://localhost:1234/v1",  "model": "",             "api_key": "", "temperature": 0.0},
        {"id": "llamacpp", "name": "Llama.cpp (Local)", "url": "http://localhost:8080/v1",  "model": "",             "api_key": "", "temperature": 0.0},
    ]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cards = data.get("cards", [])
                return cards if cards else defaults
        except Exception as e:
            print(f"Warning: Failed to parse model_card.json: {e}")
    return defaults

# ─── Server detection ─────────────────────────────────────────────────────────

def detect_running_servers():
    """Probes standard local ports for running OpenAI-compatible API servers."""
    probes = [
        {"name": "Ollama",    "url": "http://localhost:11434/v1/chat/completions", "test_url": "http://localhost:11434/v1/models", "default_model": "qwen2.5:1.5b"},
        {"name": "LM Studio", "url": "http://localhost:1234/v1/chat/completions",  "test_url": "http://localhost:1234/v1/models",  "default_model": None},
        {"name": "Llama.cpp", "url": "http://localhost:8080/v1/chat/completions",  "test_url": "http://localhost:8080/v1/models",  "default_model": None},
    ]
    for probe in probes:
        try:
            req = urllib.request.Request(probe["test_url"], method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    model_name = probe["default_model"]
                    try:
                        models_data = json.loads(response.read().decode("utf-8"))
                        if "data" in models_data and models_data["data"]:
                            model_name = models_data["data"][0]["id"]
                    except Exception:
                        pass
                    return {"name": probe["name"], "url": probe["url"], "model": model_name or "default"}
        except Exception:
            continue
    # Ollama plain-HTTP fallback
    try:
        req = urllib.request.Request("http://localhost:11434", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as response:
            if response.status == 200:
                return {"name": "Ollama", "url": "http://localhost:11434/v1/chat/completions", "model": "qwen2.5:1.5b"}
    except Exception:
        pass
    return None

# ─── Activation / deactivation ────────────────────────────────────────────────

def activate_agent_from_card(viewer, card, quiet=False):
    """Activates the LLM agent using a model card dict."""
    deactivate_agent(viewer, quiet=True)

    url         = (card.get("url") or "").strip()
    model       = (card.get("model") or "").strip() or "default"
    api_key     = card.get("api_key") or None
    temperature = float(card.get("temperature") or 0.0)
    name        = card.get("name", "Model")
    options     = card.get("options") or None

    if not url:
        Command_Engine.print_help(viewer, f"Error: Model card '{name}' has no URL configured. Open ⚙ Models to edit it.")
        return False

    # Attempt to resolve the real model name if set to "default"
    if model == "default" and url:
        try:
            models_url = url.rstrip("/")
            if not models_url.endswith("/models"):
                models_url += "/models"
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(models_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "data" in data and data["data"]:
                        detected = data["data"][0].get("id")
                        if detected:
                            model = detected
        except Exception:
            pass

    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]

    viewer.llm_backend     = "server"
    viewer.llm_url         = url
    viewer.llm_model_name  = model
    viewer.llm_loaded      = True
    viewer.llm_temperature = temperature
    viewer.llm_api_key     = api_key
    viewer.llm_options     = options

    if not quiet:
        Command_Engine.print_help(viewer, f"LLM Agent Activated: {name} → {model}")
    return True

def activate_agent(viewer, force_backend=None, quiet=False):
    """Activates the LLM agent (terminal helper — all modes unified via model cards)."""
    cards = load_model_cards()
    deactivate_agent(viewer, quiet=True)

    if force_backend == "api":
        api_card = next((c for c in cards if c.get("api_key")), None)
        if api_card:
            return activate_agent_from_card(viewer, api_card, quiet=quiet)
        Command_Engine.print_help(viewer, "Error: No model card with an API key found.\nAdd one in the ⚙ Models panel of the Agent UI.")
        return False

    elif force_backend == "local":
        if not quiet:
            print("Agent: Probing local API servers (Ollama, LM Studio, Llama.cpp)...")
        backend = detect_running_servers()
        if backend:
            viewer.llm_backend     = "server"
            viewer.llm_url         = backend["url"]
            viewer.llm_model_name  = backend["model"]
            viewer.llm_loaded      = True
            viewer.llm_temperature = 0.0
            viewer.llm_api_key     = None
            if not quiet:
                Command_Engine.print_help(viewer, f"LLM Agent Activated: {backend['name']} → {backend['model']}")
            return True
        Command_Engine.print_help(viewer, "Error: No local running server (Ollama, LM Studio, Llama.cpp) was detected.")
        return False

    else:  # auto — use first card
        if cards:
            return activate_agent_from_card(viewer, cards[0], quiet=quiet)
        Command_Engine.print_help(viewer, "Error: No model cards configured. Open the Agent UI and add one in ⚙ Models.")
        return False

def deactivate_agent(viewer, quiet=False):
    """Deactivates the LLM agent and frees memory."""
    if hasattr(viewer, "llm_history"):
        viewer.llm_history = []

    if not getattr(viewer, "llm_loaded", False):
        if not quiet:
            Command_Engine.print_help(viewer, "Agent is already inactive.")
        return

    backend    = getattr(viewer, "llm_backend", None)
    model_name = getattr(viewer, "llm_model_name", "Unknown")

    for attr in ("llm_backend", "llm_url", "llm_model_name", "llm_temperature", "llm_api_key", "llm_options"):
        if hasattr(viewer, attr):
            delattr(viewer, attr)

    viewer.llm_loaded = False
    gc.collect()

    if not quiet:
        Command_Engine.print_help(viewer, f"LLM Agent Deactivated. Unloaded: {model_name} ({backend})")

# ─── API call ────────────────────────────────────────────────────────────────

def call_api(url, model, system_prompt, user_query, history=None, temperature=0.0, api_key=None, options=None):
    """Sends an OpenAI-compatible chat completion request."""
    if history is None:
        history = []

    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_query}],
        "temperature": temperature
    }
    if options and isinstance(options, dict):
        payload.update(options)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if "choices" in res_data and res_data["choices"]:
                msg = res_data["choices"][0]["message"]
                content = msg.get("content") or ""
                content = content.strip()
                reasoning = msg.get("reasoning_content") or ""
                reasoning = reasoning.strip()

                for close_tag in ["</think>", "</thought>"]:
                    if close_tag in content:
                        parts = content.split(close_tag, 1)
                        think_part = parts[0]
                        open_tag = close_tag.replace("/", "")
                        if open_tag in think_part:
                            think_part = think_part.split(open_tag, 1)[1]
                        content = parts[1].strip()
                        if think_part.strip() and not reasoning:
                            reasoning = think_part.strip()
                        break

                usage   = res_data.get("usage", {})
                tokens  = {
                    "prompt":     usage.get("prompt_tokens", 0),
                    "completion": usage.get("completion_tokens", 0),
                    "total":      usage.get("total_tokens", 0)
                }
                return {"content": content, "reasoning": reasoning, "tokens": tokens}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            print(f"\n[Agent Error Detail]\nHTTP Status Code: {e.code}\nServer Response: {body}\n")
        except Exception:
            pass
        raise e
    return None

# ─── Worker threads ───────────────────────────────────────────────────────────

class AgentWorker(QtCore.QThread):
    finished = QtCore.Signal(str, str, str, str)  # (response, reasoning, tokens_json, error)

    def __init__(self, backend, url, model_name, system_prompt, query, history, temperature, api_key, options=None):
        super().__init__()
        self.backend       = backend
        self.url           = url
        self.model_name    = model_name
        self.system_prompt = system_prompt
        self.query         = query
        self.history       = history
        self.temperature   = temperature
        self.api_key       = api_key
        self.options       = options

    def run(self):
        try:
            res_dict = None
            if self.backend == "server":
                res_dict = call_api(self.url, self.model_name, self.system_prompt, self.query, self.history, self.temperature, self.api_key, self.options)
            if not res_dict or not res_dict.get("content"):
                self.finished.emit("", "", "", "No response received from LLM.")
            else:
                self.finished.emit(res_dict["content"], res_dict.get("reasoning", ""), json.dumps(res_dict.get("tokens", {})), "")
        except Exception as e:
            self.finished.emit("", "", "", str(e))


class RefinementWorker(QtCore.QThread):
    finished = QtCore.Signal(str, str, str, str)  # (refined_explanation, reasoning, tokens_json, error)

    def __init__(self, backend, url, model_name, user_query, commands, terminal_output, temperature, api_key, options=None):
        super().__init__()
        self.backend         = backend
        self.url             = url
        self.model_name      = model_name
        self.user_query      = user_query
        self.commands        = commands
        self.terminal_output = terminal_output
        self.temperature     = temperature
        self.api_key         = api_key
        self.options         = options

    def run(self):
        try:
            refinement_prompt = (
                "You are a helpful biological assistant.\n"
                "You have just executed the following visualizer commands on behalf of the user:\n"
                f"```\n{self.commands}\n```\n\n"
                "The commands printed the following output in the visualizer terminal:\n"
                f"```\n{self.terminal_output}\n```\n\n"
                f"The user's original request was: \"{self.user_query}\"\n\n"
                "Write a clear, friendly, and concise response explaining the results and answering their question directly. "
                "Do NOT output any commands or repeat 'Explanation:'. Just write the direct answer."
            )
            res_dict = None
            if self.backend == "server":
                res_dict = call_api(self.url, self.model_name, "You are a helpful biological assistant.", refinement_prompt, history=None, temperature=self.temperature, api_key=self.api_key, options=self.options)
            if not res_dict or not res_dict.get("content"):
                self.finished.emit("", "", "", "No response from refinement.")
            else:
                self.finished.emit(res_dict["content"], res_dict.get("reasoning", ""), json.dumps(res_dict.get("tokens", {})), "")
        except Exception as e:
            self.finished.emit("", "", "", str(e))

# ─── Viewer context ───────────────────────────────────────────────────────────

def get_viewer_session_context(viewer):
    """Builds a snapshot of the current viewer state for the LLM system prompt."""
    import numpy as np
    lines = ["\n--- ACTIVE EMAP-SSN VIEWER STATE ---"]

    n_nodes = getattr(viewer, "n_nodes", 0)
    lines.append(f"Number of Nodes: {n_nodes}")

    ref_seq = getattr(viewer, "resolved_ref_full", None)
    if ref_seq:
        lines.append(f"Active Reference Sequence: {ref_seq}")

    selected_indices = getattr(viewer, "selected_indices", [])
    if selected_indices:
        lines.append(f"Current Selection: {len(selected_indices)} nodes selected (accessible via $sele$)")
    else:
        lines.append("Current Selection: No nodes currently selected")

    metadata = getattr(viewer, "metadata", {})
    if metadata:
        lines.append("Available Metadata Properties:")
        for key, prop_data in metadata.items():
            prop_type = prop_data.get("type", "text")
            values    = prop_data.get("values")
            val_desc  = ""
            if values is not None and len(values) > 0:
                if prop_type == "number":
                    arr = np.asarray(values)
                    valid = arr[~np.isnan(arr)]
                    if len(valid) > 0:
                        val_desc = f" (min: {float(valid.min()):.1f}, max: {float(valid.max()):.1f})"
                else:
                    unique_vals = list(set(str(v) for v in values if v is not None and str(v).strip()))
                    if unique_vals:
                        val_desc = f" (e.g., {', '.join(unique_vals[:5])})"
            lines.append(f"  - {key}: type={prop_type}{val_desc}")

    cluster_labels = getattr(viewer, "cluster_labels", None)
    if cluster_labels is not None:
        unique_clusters = np.unique(cluster_labels)
        valid_clusters  = [c for c in unique_clusters if str(c).lower() not in ("noise", "none", "-1", "nan")]
        if valid_clusters:
            lines.append(f"Available Clusters: {', '.join(map(str, valid_clusters[:15]))}" + ("..." if len(valid_clusters) > 15 else ""))

    group_labels = getattr(viewer, "group_labels", None)
    if group_labels:
        unique_groups = set()
        for g_set in group_labels:
            if isinstance(g_set, set):
                unique_groups.update(g_set)
        if unique_groups:
            lines.append(f"Defined Custom Groups: {', '.join(sorted(unique_groups))}")

    lines.append("---------------------------------\n")
    return "\n".join(lines)

def get_agent_history_path(viewer):
    try:
        import emapssn_config as cfg

        cache_path, _ = resolve_selected_cache(cfg)
        return os.path.join(os.path.dirname(cache_path), "agent_history.json")
    except Exception as e:
        print(f"Warning: Could not resolve agent history path ({e})")
        import emapssn_config as cfg

        saved_layout_dir = getattr(cfg, 'SAVED_LAYOUT_DIR', os.path.join("Cache_Files", "Saved_Layouts"))
        return os.path.join(saved_layout_dir, "agent_chat_history_fallback.json")

def load_agent_history(viewer):
    path = get_agent_history_path(viewer)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load agent history from {path}: {e}")
    return []

def save_agent_history(viewer):
    path = get_agent_history_path(viewer)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(getattr(viewer, "llm_history", []), f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save agent history to {path}: {e}")

# ─── Query execution ──────────────────────────────────────────────────────────

def run_web_agent_query(viewer, query):
    if not getattr(viewer, "llm_loaded", False):
        viewer.broadcast_event({"type": "agent_error", "error": "LLM is not loaded. Select a model and activate it in the Agent UI."})
        return

    prompt_path = os.path.join(_SRC_DIR, "resources", "agent", "system_prompt.md")
    if not os.path.exists(prompt_path):
        viewer.broadcast_event({"type": "agent_error", "error": f"System prompt file missing at {prompt_path}"})
        return

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        viewer.broadcast_event({"type": "agent_error", "error": f"Could not read prompt file: {e}"})
        return

    system_prompt += get_viewer_session_context(viewer)

    model_name = getattr(viewer, "llm_model_name", "LLM")
    viewer.broadcast_event({"type": "agent_thinking", "model_name": model_name})

    backend     = viewer.llm_backend
    url         = getattr(viewer, "llm_url", None)
    temperature = getattr(viewer, "llm_temperature", 0.0)
    api_key     = getattr(viewer, "llm_api_key", None)
    options     = getattr(viewer, "llm_options", None)

    if not hasattr(viewer, "llm_history") or not viewer.llm_history:
        viewer.llm_history = load_agent_history(viewer)

    history = [{"role": msg["role"], "content": msg["content"]} for msg in viewer.llm_history]

    viewer._web_agent_worker = AgentWorker(backend, url, model_name, system_prompt, query, history, temperature, api_key, options)
    viewer._web_agent_worker.finished.connect(lambda res, reasoning, tokens, err: on_web_worker_finished(viewer, query, res, reasoning, tokens, err))
    viewer._web_agent_worker.start()

# ─── Response handling ────────────────────────────────────────────────────────

def on_web_worker_finished(viewer, query, translated_output, reasoning, tokens_json, error_msg):
    if error_msg:
        viewer.broadcast_event({"type": "agent_error", "error": error_msg})
        return

    cmd_lines         = []
    explanation_lines = []

    for line in translated_output.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
        lower_line = line_strip.lower()
        if lower_line.startswith("command:"):
            cmd_lines.append(line_strip[8:].strip())
        else:
            explanation = line_strip
            if lower_line.startswith("explanation:"):
                explanation = line_strip[12:].strip()
            elif lower_line.startswith("comment:"):
                explanation = line_strip[8:].strip()
            elif line_strip.startswith("#"):
                explanation = line_strip[1:].strip()
            
            if not lower_line.startswith("input:") and not lower_line.startswith("output:"):
                explanation_lines.append(explanation)

    explanation_str = "\n".join(explanation_lines) if explanation_lines else "Command executed successfully."
    commands_str    = "\n".join(cmd_lines)

    import io, contextlib
    captured_outputs = []
    if cmd_lines:
        for cmd in cmd_lines:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    viewer.process_command(cmd)
                except Exception as e:
                    print(f"Error executing command '{cmd}': {e}")
            output = f.getvalue().strip()
            if output:
                captured_outputs.append(output)

    terminal_output = "\n".join(captured_outputs) if captured_outputs else ""

    # Set requests count to 1 for the initial step
    if tokens_json:
        try:
            t = json.loads(tokens_json)
            t["requests"] = 1
            tokens_json = json.dumps(t)
        except Exception:
            pass
    else:
        tokens_json = json.dumps({"prompt": 0, "completion": 0, "total": 0, "requests": 1})

    if not terminal_output:
        save_and_broadcast_agent_response(viewer, query, explanation_str, commands_str, "", tokens_json, reasoning)
    else:
        start_refinement_worker(viewer, query, explanation_str, commands_str, terminal_output, tokens_json, reasoning)


def save_and_broadcast_agent_response(viewer, query, explanation, commands, terminal_output, tokens_json, reasoning=""):
    if not hasattr(viewer, "llm_history") or not viewer.llm_history:
        viewer.llm_history = load_agent_history(viewer)

    tokens = json.loads(tokens_json) if tokens_json else None

    content_payload = explanation
    if commands:
        content_payload += f"\n\nCommands executed:\n```\n{commands}\n```"
    if terminal_output:
        content_payload += f"\n\nTerminal output:\n```\n{terminal_output}\n```"

    viewer.llm_history.append({"role": "user", "content": query})
    viewer.llm_history.append({
        "role": "assistant", "content": content_payload,
        "explanation": explanation, "commands": commands,
        "terminal_output": terminal_output, "tokens": tokens,
        "reasoning": reasoning
    })
    viewer.llm_history = viewer.llm_history[-10:]
    save_agent_history(viewer)

    viewer.broadcast_event({
        "type": "agent_response", "query": query, "explanation": explanation,
        "commands": commands, "terminal_output": terminal_output,
        "tokens": tokens, "reasoning": reasoning, "llm_history": viewer.llm_history
    })


def start_refinement_worker(viewer, query, original_explanation, commands, terminal_output, initial_tokens_json, original_reasoning):
    model_name = getattr(viewer, "llm_model_name", "LLM")
    viewer.broadcast_event({"type": "agent_thinking", "model_name": f"{model_name} (Analyzing results)"})

    backend     = viewer.llm_backend
    url         = getattr(viewer, "llm_url", None)
    temperature = getattr(viewer, "llm_temperature", 0.0)
    api_key     = getattr(viewer, "llm_api_key", None)
    options     = getattr(viewer, "llm_options", None)

    viewer._refinement_worker = RefinementWorker(backend, url, model_name, query, commands, terminal_output, temperature, api_key, options)

    def on_refinement_finished(refined_explanation, refinement_reasoning, refinement_tokens_json, err):
        final_explanation  = refined_explanation if not err and refined_explanation else original_explanation
        combined_tokens_json = initial_tokens_json
        try:
            t1 = json.loads(initial_tokens_json) if initial_tokens_json else {}
            t2 = json.loads(refinement_tokens_json) if refinement_tokens_json else {}
            combined_tokens_json = json.dumps({
                "prompt":     t1.get("prompt", 0)     + t2.get("prompt", 0),
                "completion": t1.get("completion", 0) + t2.get("completion", 0),
                "total":      t1.get("total", 0)      + t2.get("total", 0),
                "requests":   2
            })
        except Exception:
            pass

        combined_reasoning = original_reasoning
        if refinement_reasoning:
            combined_reasoning = (original_reasoning + "\n\n[Analysis Thought]\n" + refinement_reasoning).strip() if original_reasoning else refinement_reasoning

        save_and_broadcast_agent_response(viewer, query, final_explanation, commands, terminal_output, combined_tokens_json, combined_reasoning)

    viewer._refinement_worker.finished.connect(on_refinement_finished)
    viewer._refinement_worker.start()

# ─── Viewer registration ──────────────────────────────────────────────────────

def handle_agent_query(viewer, data):
    query = data.get("query")
    run_web_agent_query(viewer, query)

def handle_set_backend(viewer, data):
    card = data.get("card")
    if card:
        activate_agent_from_card(viewer, card, quiet=True)
    else:
        deactivate_agent(viewer, quiet=True)
        
    viewer.broadcast_event({
        "type": "backend_state",
        "llm_loaded": getattr(viewer, 'llm_loaded', False),
        "llm_backend": getattr(viewer, 'llm_backend', None),
        "llm_model_name": getattr(viewer, 'llm_model_name', "Unknown")
    })

def handle_save_model_cards(viewer, data):
    cards = data.get("cards", [])
    try:
        path = os.path.join(_SRC_DIR, "resources", "agent", "model_card.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cards": cards}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save model_card.json: {e}")

def handle_clear_history(viewer, data):
    viewer.llm_history = []
    save_agent_history(viewer)

def _extend_initial_web_state(viewer, state):
    viewer.llm_history = load_agent_history(viewer)
    state["llm_history"] = viewer.llm_history
    return state


def register_backend(registry, viewer):
    """Register Agent web capabilities without changing sidebar state."""
    registry.register_action(
        "agent", "agent_query", lambda data: handle_agent_query(viewer, data)
    )
    registry.register_action(
        "agent", "set_backend", lambda data: handle_set_backend(viewer, data)
    )
    registry.register_action(
        "agent", "save_model_cards", lambda data: handle_save_model_cards(viewer, data)
    )
    registry.register_action(
        "agent", "clear_history", lambda data: handle_clear_history(viewer, data)
    )
    registry.register_static_route(
        "agent",
        "/agent_resource/",
        os.path.join(_SRC_DIR, "resources", "agent"),
    )
    registry.register_state_provider("agent", "history", _extend_initial_web_state)


def activate(viewer):
    """Show the Agent sidebar entry without eagerly reading chat history."""
    if hasattr(viewer, "add_sidebar_button"):
        viewer.add_sidebar_button(
            name="agentBtn",
            label="🤖 Agent",
            callback=viewer.open_agent_ui,
            tooltip="Open AI Agent Chat Console in browser"
        )


def register(viewer):
    """Compatibility wrapper: ensure backend registration, then activate its UI."""
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("agent")
    return activate(viewer)
