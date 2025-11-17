<img src="./docs/.vuepress/public/full-logo.png" height="80"/>

Tero is a platform to create, share and use AI agents with focus on empowering software quality processes.

![screenshot](./docs/.vuepress/public/screenshot.png)


## Key features

- **Create and use agents quickly** with your preferred AI models.
- **Share agents within your teams** to accelerate AI adoption.
- **Leverage a growing [community catalog](https://github.com/abstracta/tero-agents)** of contributed agents.
- **Build powerful agents** using built-in tools (e.g., Jira, MCP, Browser, Web, Docs).
- **Safely experiment** by defining and running test suites for your agents.
- **Review usage and impact** through reports in the AI Console.
- **Control costs** and use AI providers efficiently with per-user budget.
- **Integrate with developer tools** like GitHub Copilot and Cursor.
- **Use agents anywhere on the web** via the Tero Copilot browser extension.


## Quickstart

The easiest way to try Tero is through the online demo. Request access [here](https://abstracta.us/tero).

To run Tero locally:

1. Generate an OpenAI API key or an Azure OpenAI endpoint and key.
2. Clone this repository. Make sure you have git-lfs installed to get all the files properly.
3. Copy `src/sample.env` to `.env` in the project root and set `OPENAI_API_KEY` or `AZURE_API_KEYS` and `AZURE_ENDPOINTS`.
4. Start the app and dependencies with `docker compose up -d`.
5. Open `http://localhost:8000` and log in with username `test` and password `test`.

> [Docker](https://www.docker.com/) is required. 

> Some tools, like `Web` or advanced file processing for `Docs` tool, require additional configuration. Check comments in [sample.env](src/sample.env) for more details.


## Support

- Join our [Discord server](https://discord.gg/tP2hge7QHC) to engage with fellow Tero enthusiasts, ask questions, and share experiences. 
- Visit [GitHub Issues](https://github.com/abstracta/tero/issues) for bug reports, feature requests and share ideas.

[Abstracta](https://abstracta.us), the main supporter of Tero, offers enterprise support with faster responses, customizations, and consulting.


## Contributing

We value all sort of contributions, including questions and feature requests.

- Create [issues](https://github.com/abstracta/tero/issues) to ask questions & share ideas.
- Share agents with the community in [the agents repository](https://github.com/abstracta/tero-agents).
- If you are interested on helping us improve the platform, check the [Contributing Guide](CONTRIBUTING.md)
