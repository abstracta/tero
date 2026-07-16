### SQL

Query databases with natural language. The agent can explore your data and answer questions without writing SQL manually.

The agent can help with:

- **Databases** — list available databases on the server
- **Tables** — discover tables and their schemas
- **Queries** — run SELECT, INSERT, UPDATE, DELETE, and other SQL statements

::: warning
The agent can modify or delete data. To reduce risk, grant database users read-only permissions unless write access is explicitly needed.
:::

::: info
Only Azure SQL Database is currently supported.
:::

#### Authentication

This tool uses OAuth 2.0. You need to register an app in Azure Active Directory and grant it access to Azure SQL Database.

**Steps to set up:**

1. Go to the [Azure portal](https://portal.azure.com) and open **Microsoft Entra ID** → **App registrations** → **New registration**.
2. Give the app a name and set the **Redirect URI** (type: Web) to:
   ```
   {your-tero-url}/tools/sql/oauth-callback
   ```
3. After creating the app, go to **Certificates & secrets** → **New client secret**. Copy the secret value.
4. Go to **API permissions** → **Add a permission** → **APIs my organization uses** → search for **Azure SQL Database**.
5. Select the **Delegated** permission `user_impersonation` and click **Add permissions**.
6. Copy the **Application (client) ID** and the **Directory (tenant) ID** from the app's **Overview** page.

Enter the server hostname, tenant ID, client ID, and client secret in the tool configuration. After saving, each user will be prompted to authorize the connection with their Azure account.

#### Permissions

Because the tool acts on behalf of each user, the authorizing account must have the right database permissions:

- **Connect to `master`** — the database list is retrieved by querying `sys.databases` on the `master` database, so the account must have a user in `master` (on Azure SQL you can only connect to databases where your account is provisioned).
- **List databases** — permission to enumerate the databases on the server (e.g. `VIEW ANY DATABASE`); without it `sys.databases` only returns the databases the account owns.
- **Access the database** — a user provisioned in each database the user needs to work with, with at least read access to its tables and schemas (and write access if data modification is needed).

Without these permissions the agent won't be able to discover or query the data, even if the OAuth authorization succeeds.
