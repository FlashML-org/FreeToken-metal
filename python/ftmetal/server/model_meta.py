"""Protocol-level thinking toggles folded into the chat template kwargs.

Port of FreeToken server/model_meta.py (thinking_toggle_kwargs, effort_toggle_kwargs), with
the toggle broadcasts of its tokenizer/effort.py. Divergences: the reasoning-effort
vocabulary is not ported (no template probing, no /v1/models effort fields, no
/v1/cache/status thinking gears), so the routes refuse reasoning_effort and
effort_toggle_kwargs only ever folds the on/off direction; no MoE slot-cache sizes (this
engine has no expert cache).
"""

from __future__ import annotations

#: Protocol-level thinking toggles, broadcast in every spelling the ecosystem's
#: templates read (``enable_thinking`` bool: qwen/glm/gemma/dsv4;
#: ``thinking_mode`` string: minimax-m3). Jinja ignores undeclared variables,
#: so a template simply picks the knob it knows -- no per-family routing needed.
THINKING_OFF_KWARGS = {"enable_thinking": False, "thinking_mode": "disabled"}
THINKING_ON_KWARGS = {"enable_thinking": True, "thinking_mode": "enabled"}


def thinking_toggle_kwargs(enabled: bool) -> dict:
    """``chat_template_kwargs`` for a protocol-level thinking on/off toggle
    (Anthropic ``thinking.type``, DeepSeek ``thinking``, Responses
    ``reasoning.effort``): every spelling the ecosystem's templates read,
    broadcast at once. A template picks the knob it knows and ignores the rest
    (Jinja never sees undeclared variables), so no per-family routing exists."""
    return dict(THINKING_ON_KWARGS if enabled else THINKING_OFF_KWARGS)


_THINKING_KWARG_KEYS = ("enable_thinking", "thinking", "thinking_mode", "reasoning_effort")
_DISABLE_EFFORTS = ("none", "off")


def effort_toggle_kwargs(
    effort: str | None,
    chat_template_kwargs: dict | None,
    thinking_type: str | None = None,
) -> dict:
    """Fold a protocol-level reasoning-effort request into the template kwargs.
    An explicit thinking-related key wins wholesale; unrelated extras ride along.
    Effort "none"/"off" (case-insensitive) disables thinking; any other or absent
    effort enables it, forwarded for templates that grade it (quantized against
    the checkpoint's probed vocabulary at render time). ``thinking_type`` is the
    DeepSeek-wire ``thinking: {"type": ...}`` toggle; when present it decides
    the on/off direction outright, "disabled" winning over any effort."""
    ctk = dict(chat_template_kwargs or {})
    if any(key in ctk for key in _THINKING_KWARG_KEYS):
        return ctk
    if isinstance(effort, str):
        effort = effort.strip().lower()
    disabled = effort in _DISABLE_EFFORTS
    if thinking_type == "disabled":
        disabled = True
    elif thinking_type == "enabled":
        disabled = False
    mapped = thinking_toggle_kwargs(not disabled)
    if effort and not disabled and effort not in _DISABLE_EFFORTS:
        mapped.setdefault("reasoning_effort", effort)
    mapped.update(ctk)
    return mapped
