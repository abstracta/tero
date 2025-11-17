# Quick Start

## Try the hosted demo

The easiest way to try Tero is through the online demo. Request access at `https://abstracta.us/tero`.

## Run locally

> Prerequisite: Docker

1. Clone the [repository](https://github.com/abstracta/tero). Make sure you have git-lfs installed to get all the files properly.
2. Generate an OpenAI API key or an Azure OpenAI endpoint and key.
3. Copy `src/sample.env` to `.env` and set `OPENAI_API_KEY` or `AZURE_API_KEYS` and `AZURE_ENDPOINTS`.
4. Start the app and dependencies:

   ```bash
   docker compose up
   ```

5. Open `http://localhost:8000` and log in with username `test` and password `test`.

::: info
Some tools (e.g., `Web`, advanced file processing for `Docs`) require additional configuration. Check comments in `src/sample.env` for details.
:::

## Welcome screen

After you log in to Tero, the welcome screen helps you get started fast. Start by chatting with the default agent or browse shared agents in Discover.

![Welcom Screen](./img/welcome-screen.png)

### Sidebar

Use the sidebar to move quickly between agents and conversations:

- **Recents**: Browse recently used agents and chats.
- **Search**: Find agents and chats instantly.
- **Organize**: Collapse/expand the sidebar, rename chats, and remove items you no longer need.
- **Usage**: See your current monthly budget usage. See [Usage](./chat.md#budget) for details.
