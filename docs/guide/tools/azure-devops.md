### Azure DevOps

Browse projects and teams, manage work items, and work with test plans in Azure DevOps Services.

The agent can help with:

- **Projects & teams** — list and read projects and teams, and inspect team membership
- **Work items** — search with queries; read, create, update, and delete work items; inspect work item types, fields, states, and relation types; manage field definitions when needed
- **Traceability** — link work items together (for example user stories and test cases)
- **Test plans** — browse and manage test plans, suites, and configurations
- **Test cases** — list cases in a suite, add or remove them from suites, and find which suites contain a given case

#### Authentication

Microsoft Entra delegated OAuth. Register an app in Entra ID and add **Azure DevOps** delegated permissions matching the scopes you select in Tero.

**Setup:**

1. [Register an application](https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app) in Microsoft Entra ID.
2. Add a **Web** redirect URI: the callback URL shown in the tool configuration.
3. Under **API permissions**, add **Azure DevOps** delegated permissions matching the scopes available in Tero (e.g. `vso.profile`, `vso.project`, `vso.work`, `vso.work_write`, `vso.test`, `vso.test_write`).
4. Copy **Application (client) ID**, create a **client secret**, and enter **tenant ID** in Tero.

Each user authorizes with their Microsoft work account when they first use the tool. Actions run with that user's Azure DevOps permissions.

The scopes you select in Tero control what the agent can do:

- **Profile (read)** — read the signed-in user's profile
- **Projects and teams (read)** — read projects and teams
- **Work items (read)** — read work items
- **Work items (write)** — create, update, and delete work items
- **Test plans (read)** — read test plans, suites, and cases
- **Test plans (write)** — manage test plans, suites, configurations, and suite membership

See [Entra OAuth for Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/integrate/get-started/authentication/entra-oauth?view=azure-devops).
