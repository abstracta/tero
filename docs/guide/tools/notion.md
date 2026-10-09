### Notion

Search and retrieve pages and content from a Notion workspace. Access is read-only.

The agent can help with:

- **Search** — find pages by title across the workspace
- **Pages** — retrieve a page and its properties
- **Content** — read the block content of a page

::: info Shared credentials
You configure the integration token once. All users of the agent will use it with that Notion workspace data (read-only). Only the agent owner can configure or update this tool.
:::

#### Authentication

This tool uses a Notion internal integration token.

**Steps to set up:**

1. Go to [My integrations](https://www.notion.so/my-integrations) in Notion and create a new **internal integration**.
2. Copy the **Internal Integration Secret** (token).
3. In Notion, open each page the agent should access → **⋯** → **Connections** → connect your integration.
4. Paste the token into the tool configuration.

::: tip Page access
The integration can only see pages you explicitly connect. If search returns nothing, check that the relevant pages are shared with the integration.
:::
