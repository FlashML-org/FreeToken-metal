# Quick start

Assumes FreeToken-metal is installed ([install.md](install.md)).

## Start a server

```bash
ftm serve --model /path/to/Qwen3.6-35B-A3B-MLX
```

It listens on `127.0.0.1:1919` and serves the model under its directory name. Memory, cache sizes
and parsers are set from the model and the Mac; [cli.md](cli.md) lists the options.

## Send a request

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B-MLX",
    "messages": [{"role": "user", "content": "What is a Mixture-of-Experts model?"}],
    "stream": true
  }'
```

Any OpenAI or Anthropic client works with its base URL pointed at the server.

## Chat in the terminal

```bash
ftm shell
```

## Use a coding agent

```bash
ftm launch claude   # or codex, opencode, openclaw, hermes, pi, dsh
```

It configures the agent for your server, offers to install it if missing, and starts it.
