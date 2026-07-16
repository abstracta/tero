### Zephyr

Manage test cases, plans, cycles, and executions in Zephyr alongside your Jira projects. The agent can work across test library structure, linkage to Jira issues, planning, and execution tracking.

When configuring the tool, choose **Zephyr Cloud** or **Zephyr Essential Cloud**. The access token must match the selected product.

Zephyr does not require the [Jira](./jira.md) tool, but enabling both is recommended when you need full issue context—reading or updating stories and bugs, JQL search, or backlog work beyond test management and traceability links.

The agent can help with:

- **Test cases** — search test cases by project, status, and name; clone test cases; read and update test cases; folders, labels, and custom fields; steps and scripted tests
- **Traceability** — view and create links between test cases and Jira issues
- **Test plans & cycles** — create and browse plans and cycles; link cycles to plans when supported by the workflow
- **Executions** — create and update executions per cycle and environment; review execution outcomes
- **Project context** — list projects, folders, statuses, priorities, and environments relevant to configuring tests

#### Authentication

This tool uses a **personal Zephyr API access token**. Each Tero user connects with their own token; Tero stores it per user and reuses it until Zephyr rejects it (for example when the token expires), at which point you are prompted to enter a new one.

Anything the agent creates or updates in Zephyr is performed as the **Zephyr/Jira user who owns that token**, with that account’s permissions—not as a generic “agent” user.

**Generate an access token**

1. In Jira/Zephyr, create an API access token for your account ([Zephyr Essential Cloud](https://smartbear.portal.swaggerhub.com/zephyr-squad/default/authentication#generate-a-key) or [Zephyr Cloud](https://support.smartbear.com/zephyr/docs/en/rest-api/api-access-tokens-management.html), depending on your product).

**Connect in Tero**

1. **When configuring the agent** — select your **Product**, paste the token in **Access token**, and save. This validates the tool and saves the connection for your Tero user.
2. **When using the agent** — every other user must enter their own token the first time they chat with that agent (or again after their token stops working). You will see a prompt asking for the token; it is not enough that only the agent creator configured one.
