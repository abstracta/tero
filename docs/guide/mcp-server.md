# Tero MCP Server

Tero exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server, letting external AI tools — such as Claude Code, Claude Desktop, or Cursor — interact directly with your Tero agents.

## Server URL

The MCP server is available at your Tero instance URL with `/mcp` appended, for example: `https://your-tero-instance.example.com/mcp`.

## Authentication

Tero's MCP server uses **OAuth via OpenID Connect**. When you connect a client for the first time, the client will open a browser window for you to log in with your Tero credentials. After authorizing, the client stores the session and subsequent requests are authenticated automatically.

## Client Configuration

### Claude Code

You can add the Tero MCP server either via the CLI or by editing the configuration file directly.

**Via CLI:**

```bash
claude mcp add --transport http tero https://your-tero-instance.example.com/mcp
```

**Via JSON (`~/.claude.json` for user scope, or `.mcp.json` in project root for project scope):**

```json
{
  "mcpServers": {
    "tero": {
      "type": "http",
      "url": "https://your-tero-instance.example.com/mcp"
    }
  }
}
```

After saving the JSON file, open a Claude Code session and run the `/mcp` command to connect and authenticate:

1. Type `/mcp` in the Claude Code prompt and press Enter.
2. Claude Code will list your configured MCP servers and show their connection status.
3. If the Tero server shows as **Not connected**, select it and choose **Authenticate** to start the OAuth login flow.
4. A browser window will open — log in with your Tero credentials and authorize the connection.
5. Run `/mcp` again to confirm the server status shows as **Connected**.

### Claude Desktop

1. Open Claude Desktop and go to **Settings** → **Developer** → **Edit Config**.
2. Add the Tero server under `mcpServers`:

```json
{
  "mcpServers": {
    "tero": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://your-tero-instance.example.com/mcp"
      ]
    }
  }
}
```

3. Save the file and restart Claude Desktop. On first use, it will open a browser window to complete the OAuth login.

::: tip Version requirement
Remote HTTP MCP servers with OAuth require a recent version of Claude Desktop. Make sure you are running the latest version.
:::

### Cursor

1. Open **Cursor Settings** → **MCP** and click **New MCP Server**.
2. Add the following under `mcpServers`:

```json
{
  "mcpServers": {
    "tero": {
      "type": "http",
      "url": "https://your-tero-instance.example.com/mcp"
    }
  }
}
```

3. Open **Cursor Settings** → **MCP** and click **Connect** next to the Tero server.
4. A browser window will open to complete the OAuth login.

Refer to the [Cursor MCP documentation](https://docs.cursor.com/context/model-context-protocol) for full details.

### VS Code

Create or edit `.vscode/mcp.json` in your workspace root and add the Tero server:

```json
{
  "servers": {
    "tero": {
      "type": "http",
      "url": "https://your-tero-instance.example.com/mcp"
    }
  }
}
```

Reload the VS Code window after saving. On first use, Copilot will prompt you to complete the OAuth login in your browser.

::: tip Version requirement
MCP support in VS Code requires version 1.99 or later with GitHub Copilot enabled.
:::

## Usage

Your AI client needs to be explicitly told to use Tero agents. For example:

> "Use Tero to ask the support agent: what is the refund policy?"

> "Ask the Tero QA agent to write test cases for this feature."

This tells the client to invoke the MCP tools and route the request through Tero.

## Available Tools

Once connected, your AI client can use the following tools:

### `list-agents`

List the Tero agents accessible to your account, optionally filtered by a search term.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | No | Filter agents by name or description |

**Returns** a list of agents, each with:
- `id` — numeric agent identifier
- `name` — agent display name
- `description` — agent description

---

### `start-conversation`

Create a new conversation thread with a specific agent. If an empty thread already exists for that agent, it is returned instead of creating a duplicate.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | integer | Yes | The ID of the agent to talk to (from `list-agents`) |

**Returns**:
- `thread_id` — identifier for the new thread, used in `send-message`
- `thread_name` — display name of the thread (may be `null` until the first message is sent)

---

### `send-message`

Send a message to an agent in an existing thread and receive the complete response. This is a synchronous call — it waits for the full agent response before returning.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `thread_id` | integer | Yes | The conversation thread ID (from `start-conversation`) |
| `message` | string | Yes | The message text to send |

**Returns**:
- `response` — the agent's complete text response
- `thread_id` — the thread the response belongs to

::: tip Typical flow
Use `list-agents` to find an agent → `start-conversation` to open a thread → `send-message` one or more times to exchange messages.
:::

---

### `add-time-saving`

Record the estimated time saved while using an external AI agent. This is the MCP equivalent of the **Register other agents** button in the [AI Console](./console.md).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_name` | string | Yes | Name of the external AI agent (e.g. `"Cursor"`) |
| `minutes_saved` | integer | Yes | Estimated number of minutes saved |

This tool is typically called automatically by the AI client after each productive interaction, using a rule or instruction you configure.
