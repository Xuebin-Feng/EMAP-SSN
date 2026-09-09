# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Allowlisted workflow dispatch with shared discovery and validation models."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any, Literal, get_type_hints

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, ValidationError, create_model, model_validator

import mcp_server.pipeline.Pipeline_Operations as pipeline_ops
import mcp_server.viewer.Viewer_Operations as viewer_ops

PipelineAction = Literal[
    "help", "describe", "list_tools", "get_tool_schema", "get_compute_capabilities",
    "inspect_file", "export_tool_settings", "export_layout_settings", "validate_settings",
    "start_job", "start_layout_job", "list_jobs", "get_job", "read_log", "cancel_job",
]
ViewerDataAction = Literal["help", "describe", "list_sessions", "get_summary", "query_nodes", "read_log", "describe_fields", "create_subset", "summarize_subset", "read_value", "get_command_request", "list_command_requests", "read_command_output", "capture_view", "get_command_catalog"]
ViewerControlAction = Literal[
    "help", "describe", "get_settings_schema", "export_settings", "validate_settings",
    "start_session", "connect_session", "disconnect_session", "close_session", "execute_commands",
]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DescribeArguments(Arguments):
    action: str


class CommandLookupArguments(Arguments):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "oneOf": [
            {"required": ["request_id"], "properties": {"request_id": {"type": "string", "minLength": 1, "pattern": r"\S"}, "submission_id": {"type": "null"}}},
            {"required": ["submission_id"], "properties": {"submission_id": {"type": "string", "minLength": 1, "pattern": r"\S"}, "request_id": {"type": "null"}}},
        ]})

    @model_validator(mode="after")
    def check_identifier(self):
        if (self.request_id is None) == (self.submission_id is None):
            raise ValueError('Supply exactly one of request_id or submission_id')
        value = self.request_id if self.request_id is not None else self.submission_id
        if not value.strip():
            raise ValueError('request_id or submission_id must be a nonempty string')
        return self


@dataclass(frozen=True)
class Action:
    module: Any
    handler_name: str
    model: type[BaseModel]
    needs_context: bool
    effects: str
    example: dict[str, Any]


_SPECS = {
    "emapssn_pipeline": {
        "list_tools": (pipeline_ops, "list_pipeline_tools", "Read pipeline catalog and shared queue capacity.", {}),
        "get_tool_schema": (pipeline_ops, "get_pipeline_tool_schema", "Read one pipeline's settings contract.", {"tool_id": "sanitize_sequences"}),
        "get_compute_capabilities": (pipeline_ops, "get_compute_capabilities", "Read runtime device metadata without computation benchmarks.", {}),
        "inspect_file": (pipeline_ops, "inspect_pipeline_file", "Read a selected file without modifying it.", {"path": "input.fasta"}),
        "export_tool_settings": (pipeline_ops, "export_pipeline_settings", "Create a settings file from saved pipeline preferences.", {"tool_id": "sanitize_sequences"}),
        "export_layout_settings": (pipeline_ops, "export_layout_settings", "Create a layout settings file inheriting saved Config preferences; does not reserve a cache name.", {}),
        "validate_settings": (pipeline_ops, "validate_pipeline_settings", "Validate pipeline settings without submitting a job or writing files; not a layout validator.", {"tool_id": "sanitize_sequences", "parameters": {}}),
        "start_job": (pipeline_ops, "start_pipeline_job", "Enqueue a pipeline; may create or overwrite files according to settings.", {"tool_id": "sanitize_sequences", "settings_path": "pipeline.json"}),
        "start_layout_job": (pipeline_ops, "start_layout_job", "Validate and enqueue layout-cache generation in the shared pipeline queue; writes cache artifacts, does not launch a Viewer.", {"settings_path": "layout.json"}),
        "list_jobs": (pipeline_ops, "list_pipeline_jobs", "Read recent server-owned pipeline and layout jobs.", {}),
        "get_job": (pipeline_ops, "get_pipeline_job", "Read pipeline or layout job status and output locations.", {"job_id": "job-id"}),
        "read_log": (pipeline_ops, "read_pipeline_log", "Read a bounded byte page of a pipeline or layout log.", {"job_id": "job-id", "stream": "stdout"}),
        "cancel_job": (pipeline_ops, "cancel_pipeline_job", "Cancel queued work or terminate a running job; does not undo artifact writes.", {"job_id": "job-id"}),
    },
    "emapssn_viewer_data": {
        "get_command_request": (viewer_ops, "get_command_request", 'Read command outcomes using exactly one request_id or submission_id in the selected Viewer.', {'submission_id': 'client-generated-id'}),
        "list_command_requests": (viewer_ops, "list_command_requests", 'Recover Viewer command requests.', {}),
        "read_command_output": (viewer_ops, "read_command_output", 'Read command-scoped diagnostics using exactly one request_id or submission_id.', {'request_id': 'request-id'}),
        "capture_view": (viewer_ops, "capture_view", 'Read current canvas as a PNG image.', {}),
        "get_command_catalog": (viewer_ops, "get_command_catalog", 'Read command summaries, argument choices/aliases and syntax; supply command for detailed help without executing it.', {"command": "reset"}),
        "list_sessions": (viewer_ops, "list_viewer_sessions", "Read available sessions without selecting one.", {}),
        "get_summary": (viewer_ops, "get_viewer_summary", "Capture an immutable Viewer snapshot and overview.", {}),
        "query_nodes": (viewer_ops, "query_viewer_nodes", "Read snapshot nodes; omitted columns returns no metadata.", {"snapshot_id": "snapshot-id", "limit": 25, "columns": []}),
        "describe_fields": (viewer_ops, "describe_viewer_fields", 'Page snapshot metadata types, missingness and provenance availability.', {'snapshot_id': 'snapshot-id'}),
        "create_subset": (viewer_ops, "create_viewer_subset", 'Intersect explicit scope with metadata/header/label/selection predicates; no commands or file/residue predicates.', {'snapshot_id': 'snapshot-id', 'scope': 'all'}),
        "summarize_subset": (viewer_ops, "summarize_viewer_subset", 'Exact metadata and membership statistics; omitted subset uses all nodes. Page complete category counts.', {'snapshot_id': 'snapshot-id'}),
        "read_value": (viewer_ops, "read_viewer_value", 'Read exact JSON-text character slices; concatenate text then JSON-decode. Continue from next_offset.', {'snapshot_id': 'snapshot-id', 'index': 0, 'field': 'node_id'}),
        "read_log": (viewer_ops, "read_viewer_log", "Read a bounded byte page of captured Viewer output.", {}),
    },
    "emapssn_viewer_control": {
        "execute_commands": (viewer_ops, "execute_viewer_commands", "Execute existing Viewer commands; can write files and open interactive interfaces.", {"submission_id": "client-generated-id", "commands": ["select help"]}),
        "get_settings_schema": (viewer_ops, "get_viewer_settings_schema", "Read the Viewer settings contract.", {}),
        "export_settings": (viewer_ops, "export_viewer_settings", "Create a full Viewer settings file inheriting saved Config preferences.", {}),
        "validate_settings": (viewer_ops, "validate_viewer_settings", "Read and validate Viewer inputs/cache identity without launching.", {"settings_path": "viewer.json"}),
        "start_session": (viewer_ops, "start_viewer_session", "Launch an independent Viewer and connect this transport to it.", {"settings_path": "viewer.json"}),
        "connect_session": (viewer_ops, "connect_viewer_session", "Change this transport's selected Viewer; does not launch one.", {}),
        "disconnect_session": (viewer_ops, "disconnect_viewer_session", "Clear this transport's selection, leaving the Viewer running.", {}),
        "close_session": (viewer_ops, "close_viewer_session", "Terminate the selected or explicitly identified Viewer.", {}),
    },
}


def _build_action(workflow, name, spec):
    module, handler_name, effects, example = spec
    handler = getattr(module, handler_name)
    hints = get_type_hints(handler, include_extras=True)
    parameters = inspect.signature(handler).parameters
    fields = {
        key: (hints[key], ... if parameter.default is inspect.Parameter.empty else parameter.default)
        for key, parameter in parameters.items() if key != "ctx"
    }
    base = CommandLookupArguments if name in {'get_command_request', 'read_command_output'} else Arguments
    model = create_model(f"{workflow}_{name}_Arguments", __base__=base, **fields)
    return Action(module, handler_name, model, "ctx" in parameters, effects, example)


REGISTRY = {
    workflow: {name: _build_action(workflow, name, spec) for name, spec in specs.items()}
    for workflow, specs in _SPECS.items()
}

_PUBLIC_NAMES = {
    action.handler_name: f"{workflow}(action='{name}')"
    for workflow, actions in REGISTRY.items() for name, action in actions.items()
}
_EXPORT_NAMES = {
    "export_config_settings(kind='layout')": "emapssn_pipeline(action='export_layout_settings')",
    "export_config_settings(kind='viewer')": "emapssn_viewer_control(action='export_settings')",
    "export_config_settings": "the workflow's settings-export action",
}
_TEXT_NAMES = {**_PUBLIC_NAMES, **_EXPORT_NAMES}
_NAME_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(map(re.escape, sorted(_TEXT_NAMES, key=len, reverse=True))) + r")(?!\w)"
)


def _public_text(text):
    return _NAME_PATTERN.sub(lambda match: _TEXT_NAMES[match.group()], text)


def _public_schema(value):
    if isinstance(value, dict):
        return {key: _public_text(item) if key == "description" and isinstance(item, str)
                else _public_schema(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_schema(item) for item in value]
    return value


def _validate(model, arguments):
    try:
        return model.model_validate(arguments).model_dump()
    except ValidationError as error:
        raise ToolError(str(error)) from error


async def dispatch(workflow: str, action: str, arguments: dict[str, Any], ctx) -> dict[str, Any]:
    actions = REGISTRY.get(workflow)
    if actions is None:
        raise ToolError(f"Unknown workflow: {workflow}")
    if action == "help":
        _validate(Arguments, arguments)
        return {"workflow": workflow, "actions": [
            {"action": "help", "description": "List this workflow's actions."},
            {"action": "describe", "description": "Get one action's argument schema, effects and example."},
            *[{"action": name, "description": entry.effects} for name, entry in actions.items()],
        ]}
    if action == "describe":
        target = _validate(DescribeArguments, arguments)["action"]
        if target in ("help", "describe"):
            model = Arguments if target == "help" else DescribeArguments
            example = {} if target == "help" else {"action": "help"}
            return {"action": target, "arguments_schema": model.model_json_schema(),
                    "effects": "Read discovery metadata only.",
                    "example": {"action": target, "arguments": example}}
        entry = actions.get(target)
        if entry is None:
            raise ToolError(f"Unknown action for {workflow}: {target}; use help.")
        return {"action": target,
                "description": _public_text(inspect.getdoc(getattr(entry.module, entry.handler_name)) or entry.effects),
                "arguments_schema": _public_schema(entry.model.model_json_schema()),
                "effects": entry.effects,
                "example": {"action": target, "arguments": entry.example}}
    entry = actions.get(action)
    if entry is None:
        raise ToolError(f"Unknown action for {workflow}: {action}; use help.")
    kwargs = _validate(entry.model, arguments)
    if entry.needs_context:
        kwargs["ctx"] = ctx
    result = getattr(entry.module, entry.handler_name)(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result.model_dump(mode="json") if isinstance(result, BaseModel) else result


__all__ = [
    "Action",
    "PipelineAction",
    "REGISTRY",
    "ViewerControlAction",
    "ViewerDataAction",
    "dispatch",
]
