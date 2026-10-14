# Tero CLI

CLI to authenticate with Tero, manage API keys, and send messages to agents.

## Prerequisites

- `curl` — used by the install command and `install.sh` to download the CLI binary

## Installation

Installs the latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/abstracta/tero/main/src/cli/install.sh | bash
```

### CI

```yaml
- run: curl -fsSL https://raw.githubusercontent.com/abstracta/tero/main/src/cli/install.sh | bash
- run: tero --version
```

## Commands

### Login

Interactive browser login:

```bash
tero login --url "https://tero.abstracta.us"
```

Stores the URL and token in `~/.tero/config.json`.

### Create API key

```bash
tero create-api-key --name "ci-key"
```

Save the printed `apiKey` value and use it as `TERO_API_KEY`.

### Send message to agent

```bash
tero ask --agent-id 23 --message "Hello, help me with this issue"
```

Save the reply to a file using `--output <path>`:

```bash
tero ask --agent-id 23 --message "Summarize the release" --output ~/reports/report.md
tero ask --agent-id 23 --message "Summarize the release" --output ~/reports/report.html
```

If the path ends in `.html`, the output is rendered as a self-contained HTML with code highlighting, tables, echarts, and bare `@startuml` diagrams (fenced `plantuml` blocks stay as code). Any other extension saves raw markdown. Intermediate directories are created automatically.


## Environment variables

- `TERO_URL`: Tero base URL (used when no URL is saved from login).
- `TERO_API_KEY`: API key value returned by `create-api-key`.
