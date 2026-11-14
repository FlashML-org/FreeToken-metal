# CLI reference

| Command | |
|---|---|
| `ftm serve --model DIR` | Serve a model |
| `ftm shell` | Chat with a running server |
| `ftm launch <agent>` | Start a coding agent against a running server |
| `ftm pack --model DIR` | Pack a MoE checkpoint's experts for offload |

Every command takes `--help` for its full options.

## ftm serve

| Flag | Default | |
|---|---|---|
| `--model` | required | Checkpoint directory |
| `--host`, `--port` | 127.0.0.1, 1919 | Bind address |
| `--served-model-name` | directory name | Model id in the API |
| `--api-key` | off | Require this key on every request |
| `--max-running-requests` | 1 | Requests served at once |
| `--num-token` | what memory holds | KV cache capacity in tokens |
| `--memory-ratio` | automatic | Fraction of GPU memory to use |
| `--moe-strategy` | auto | `fused` keeps experts in memory, `offload` reads them from the SSD |

### API

- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`
- Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
- Status: `/health`, `/v1/stats`

Turn a model's thinking off with `reasoning_effort: "none"` or
`chat_template_kwargs: {"enable_thinking": false}`.

## ftm shell

`ftm shell [--server URL]` chats with a running server. `/think` toggles thinking, `/reset`
clears the conversation, `/exit` quits.

## ftm launch

`ftm launch <agent>` with `claude`, `codex`, `opencode`, `openclaw`, `hermes`, `pi` or `dsh`
configures the agent for the running server, offers to install it if missing, and starts it. Changed
files are backed up. `--dry-run` shows what it would do; arguments after `--` go to the agent.

## Environment

| Variable | |
|---|---|
| `FTMETAL_HOST` | Server URL for `ftm shell` and `ftm launch` |
| `FTMETAL_API_KEY` | Default of `--api-key` |
