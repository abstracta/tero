# Tero CLI

CLI to authenticate with Tero, manage API keys, and send messages to agents.

## Prerequisites

- `curl` — used by the install command and `install.sh` to download the CLI binary

Create a [GitLab personal access token](https://docs.gitlab.com/user/profile/personal_access_tokens/) with access to this repository and these scopes:

- `read_repository` — fetch `install.sh` from the repository
- `read_api` — download CLI binaries from the GitLab package registry

Export it as `GITLAB_TOKEN` before running the install command.

## Installation

```bash
GITLAB_TOKEN=glpat-xxx curl -fsSL -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.abstracta.us/api/v4/projects/350/repository/files/src%2Fcli%2Finstall.sh/raw?ref=master" | bash
```

Set `TERO_VERSION` to install a specific release (e.g. `0.1.0`) or the latest master build (`dev`):

```bash
GITLAB_TOKEN=glpat-xxx curl -fsSL -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.abstracta.us/api/v4/projects/350/repository/files/src%2Fcli%2Finstall.sh/raw?ref=master" | TERO_VERSION=dev bash
```

### CI/CD Pipelines

Add `GITLAB_TOKEN` as a secret/variable in your pipeline and run:

```yaml
script:
  - curl -fsSL -H "PRIVATE-TOKEN: $GITLAB_TOKEN" "https://gitlab.abstracta.us/api/v4/projects/350/repository/files/src%2Fcli%2Finstall.sh/raw?ref=master" | bash
  - tero --version
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
