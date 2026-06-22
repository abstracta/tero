# Contributing

Thanks for your interest in contributing! Whether you’re here to build new capabilities or refine existing ones, you’re in the right place.

### How you can contribute

- **Build new agent tools**: Add integrations or capabilities agents can use.
- **Improve existing tools**: Fix bugs, enhance features, or optimize performance and UX.
- **Backend or frontend enhancements**: Extend APIs, domain models, UI, or developer workflows.
- **Docs and examples**: Clarify guides, add samples, and improve onboarding.
- **Tests and quality**: Increase coverage, add type hints, and refine automated checks.

### Fast path to your first tool

Impactful contributions today come from adding new agent tools (integrations and capabilities agents can use). If you’re here to build one, follow these steps:

1. Set up the environment following [Development environment](#development-environment).
3. Follow [Implementing new tools](#implementing-new-tools) to implement the new tool.
4. Open a draft PR early to get quick feedback.

### Communication and support

- Questions or ideas? Open an issue in [GitHub Issues](https://github.com/abstracta/tero/issues) or chat with us on [Discord](https://discord.gg/tP2hge7QHC).
- Prefer small, focused PRs with clear motivation and testing notes.

Before submitting changes, please run checks and tests as described in [Automated tests](#automated-tests), and keep code style consistent with existing conventions (see [Conventions and best practices](#conventions-and-best-practices)).


## Development environment

### Requirements

- [Docker](https://www.docker.com/)
- [devbox](https://www.jetpack.io/devbox)
- [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
- If you use a Visual Studio Code based IDE (like Cursor, Antigravity, etc), then consider installing the [EditorConfig extension](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig) to apply all the rules defined in `.editorconfig` and keep the code format consistent.

### Setup

To install development environment dependencies run:

```bash
devbox install && devbox run install
```

> The first devbox command installs devbox packages (Python, Node, etc.), and the second installs Poetry and PNPM dependencies, in addition to creating a `.env` file from `sample.env`. This process may take a while the first time you run it.

Review [.env](./.env) and at least set an OpenAI API key or Azure OpenAI endpoint and API key. Check the rest of the available settings in the `.env` file in case you want to enable additional features.

Start the Postgres database:

```bash
devbox run postgres
```

Start Keycloak instance used for authentication:

```bash
devbox run keycloak
```

#### Browser Tool

If you want to try the Tero Browser tool, you’ll also need to start the Playwright MCP server:

```bash
devbox run playwright
```

### Run Tero

You can run the backend with hot reloads with this command:

```bash
devbox run backend
```

And the frontend with hot realoads with this one:

```bash
devbox run frontend
```

You can now access Tero at the URL provided by the frontend run command and log in with `test` user and `test` password.

Now you have a full environment with hot reloads that you can use to apply changes to Tero backend or frontend.

### Browser Copilot

If you want to work with the Browser Copilot, you can run this command to start a browser with the extension installed and with hot reloads:

```bash
devbox run browser
```

### Automated tests

To run automated backend tests run:

```bash
devbox run tests
```

Additionally, you should also run type checks for backend code when you modify it (this avoids many unexpected bugs when running the application):

```bash
devbox run check
```

### Database migrations

When you need to alter the SQL schema or data you can generate new DB migrations with:

```bash
devbox run new-migration "<migration_name>"
```

This will generate a new migration based on changes in any of the domain classes extending `SQLModel` with `table=True`.

To run DB migrations against the local database:

```bash
devbox run migrations
```

> `devbox run postgres` automatically updates the database with migrations when it is run.


### User guide changes

To preview updates to the published user guide locally with hot reload, start a docs server:

```bash
devbox run docs
```

And open the URL printed in the terminal to view the docs.

## Architecture

Here are the main components of the Tero solution:

![architecture diagram](https://www.plantuml.com/plantuml/png/dLPPaw9847xtLx3pwjeSekBCpaHmgMWhNdXqx4i1nQ5SGY7YnFptvMX5kdjw8jOmD8-gBxEoygY_T6ZO7iIUvQ6ymYo4WkWhZeSWUJ9jPCLg7C5Gypq4FVIa-IuUEC13iUkq8gGxjegRjgE2CFk2LgKAGXl127T1X6281vuPDVvwjSDBl3px7tOqEzD-RY8wsbwGEU9Y0ZH0xA0rl23MiKtFpL0nBVk1uRU7PKxc44e3Y0M-a-PCem4qv8TOxWm68koBaX5V6Y7K4F1C1s9gw2St7LVI_th_sAxvjOWVhbcXxyAdp9sVaGOS-9ruwltwyOYW-O7j-EWH0UDBsGT8hGIgMGJyy7FCE8pyqYc-85-HRb4kzKLONH0U8R8zCqC5oDPmtKyynmSw3eAWLSIOwkYjB6WMgfaasdMRfiK_d6Heer6iVdYGydfRh-V-x1ZPxVW5sZeZQXX1YYpVUxzN1mYe88AOhXOR1Qw-pO-G6SBev7nczWXJBOl0iK2Ol39l8lZxFBtjlXCjruwoi498jDi6fNVrhjhHsg2hGGQg30qffcFqOATBTNNLwEAER_fu7lUQYrJTnIpjqoJCxwHmSbwnKtwEhM1uVfE7rORt9syWtuJ8olBkvuC9Fu0Ui0juEIDPgOPNQbll_2m0Vpt0W6iSmWA5pQaQY6jheH_v1ceLgzu1PVvWs1iekNAh0CHAvxxKLkg-PcStucHP2psjD2_uMboKlVIWUUDh4fN0srNDpqp-9zzKg8U-LgOCLz9rR5CBoiBWv_LgyFrCtSeo9TR_PCI4hksLPnyMP6xYuzj3u6mxQLay1VbgD2-GhxOBJ9Yto6zvhNJfl-qD9oyJOZ8qVJRxp5QA9IfcHl4voveyky_-k2lqGY8d6D4H5vibpK9gCrMk9uLR3IbWs7DlPMnP0KxIbtv2paSzR-2JWpQrV06segIza_lSjXNtC1XDFPgnN4eDvhlbYWoCAqLo2vBW0E_GxXeVYYOp8lfzggcwr9RI9LCTDws0x7GFUk_2gyYzKll52t_YBZGXuqAgxAt9ZiClpSDYpbyMCoYeONldf20Xzekhf72MEFU4TNl0qgBBJQJ4arCQYgRYaquys-43EH4aSS9tsmjBIhQqa8PdTdMIj45lICgJvh3VkOHMRtj8UiUHOYcof2HBfHVpcqGCpB71ggoBztsSujfROOvE-_WPSM8FFsnVy7L2h4Z3YxTjbaI2n-dgbCHHWdk6_op92-1uysRpoasUYTdEICZn0UVsQkhXv5BPZKTZpUNY4nADVkUCXcw_gmyNuj1mLI0G_PGLJNgC1clh7A-v6MquuoafiH6zfQOhWvbszq7Jn3Vn6IO7VvSs9Ro_6yrUzgU3jBT31fy4epRTZEPi7Cqv8CczPEs8VJfOXQD8jZMxCu7Sn23XXRB4qNftdIR7H3HqWqE81kVzOkEUZEMC7NefP8uzLcvx8vGiEwie3dlApfminh4mrtg0U678vKm0U1_9KogPf1Onc0qsuuGu3TRYooZQMcIiqrUL9yBJchrSdZLE6jkaOdFC1Phmod2z-Dcsus6qWRCkDnt9EvTao413K_oYd6c1wIJomf93QxyHjfqaJrjkl5GwOdWScwRv_NjzM1LLju8WrAoodWe1Lae-Aj0_Txg37O82QvmHDOXSKzlRhNTpLmlAbbriVzV1RpC7dBFY3QFVp9qO9ncAOfWEqx_R8OlstcWKpvVMZzjhfUIVVbGp-oufNmQDsxCcbxokk8bQ6Ku-c6k2swnht5upzrMFWlkgwgrmNrSNLNZbIwQ2g_ZxckAPyB54LqlwAJj8ER8g_c6SLC8tawDIl3RuIb0suuelE-w3_Y75AccU2XLNJULATa-6_m00)

> Some of the components are optional depending on Tero usage and configuration.

## Project structure

Inside [src](src) folder you can find a folder for each of the main Tero components:

| Folder | Contents |
| --- | --- |
| [backend](src/backend) | The source code of the backend Python service |
| [frontend](src/frontend) | The source code of the frontend. The frontend is packaged and delivered as part of the backend. |
| [browser-extension](src/browser-extension/) | The source code for the browser copilot extension |
| [common](src/common) | Common source code used both by the frontend and browser copilot extension |

## Implementing new tools

Before implementing a new tool, check the available tools in the [src/backend/tero/tools](src/backend/tero/tools) folder. It is also good practice to ask about plans for the tool by creating a GitHub issue, in case someone else has the same idea so we can collaborate.

To create a new tool:

1. Create a new module inside the `tools` directory with the name of the tool (e.g., `my-tool`).
2. Create a `tool-schema.json` that specifies the parameters the user can set to configure the tool. This is used by the frontend to display the configuration modal, and by the backend to validate configurations.
3. Create a `tool.py` containing a class extending `AgentTool`.
4. Set a stable `id` for your tool. This is used in the database to reference the tool configuration.
5. If an agent could use multiple instances of the tool, each with its own configuration, add a `-*` suffix to the `id`. In the `configure` method, override it so each configuration has a unique id. See the [MCP Tool](src/backend/tero/tools/mcp/tool.py) for an example.
6. Implement the necessary logic in `AgentTool` methods, especially `build_langchain_tools`, which should build [LangChain](https://docs.langchain.com/oss/python/langchain/overview) tools used by the agent. Check `AgentTool` in [core.py](src/backend/tero/tools/core.py) for other methods you can override to control the tool lifecycle.
7. Add your tool to the [ToolRepository](src/backend/tero/tools/repos.py), so it can be listed when retrieving available tools and then configured.
8. Implement at least one test that verifies the proper behavior of your tool in [test_tools.py](src/backend/tests/test_tools.py).
9. Run the entire test suite with `devbox run tests` and `devbox run check` to verify proper type resolution in Python and avoid potential bugs.
10. Select a proper icon for your tool. Ideally one that is provided by [Tabler Icons](https://tabler.io/icons) or at least aligns with the design.
11. Add internationalization (i18n) strings for the configuration parameters, name of the tool, and description in [AgentToolConfigEditor.vue](src/frontend/src/components/agent/AgentToolConfigEditor.vue). You can use a Tero agent to translate texts you’re unsure about; the Tero team will also help refine texts in your PR.
12. Add a mapping from your tool id to the proper icon in [useToolConfig.ts](src/frontend/src/composables/useToolConfig.ts).
13. Add a new file in [docs/guide/tools](docs/guide/tools/) for the new tool explaining what it does and how to configure it.
14. Add proper include to the new file in [docs/guide/tools/index.md](docs/guide/tools/index.md).
15. Test your tool and share your shiny new tool with the rest of the community by sending a pull request to this GitHub repository!

Optionally, create additional Python scripts, new migrations, and any additional resources you need for your new tool.

## Conventions and best practices

In general we follow common programming good practices and some language specific practices. Here are some that we’d like to highlight:

- Run automated checks and tests before submitting a PR. Don’t waste time waiting for the pipeline or for someone to point out an issue you could have detected earlier.
- Always check existing code and follow the same conventions and coding style to keep it consistent (naming, order of methods and parameters, framework usage, module structure, etc.).
- Use proper TypeScript types and Python type hints as much as possible to clarify method contracts and avoid bugs.
- When adding new functionality, add appropriate automated tests.
- Prefer “Clean Code” ordering (callers before callees), keep variables and declarations near where they are used, and follow the KISS (Keep It Simple Stupid) principle.
- Use comments to explain the “why” (when necessary), not the “what”. Use function and variable names for the “what”. Avoid comments that simply restate what the code already expresses.
