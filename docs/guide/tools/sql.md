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

Enter the following fields in the tool configuration:

- **Server** (required) — SQL Server fully qualified domain name.
- **Tenant ID** (required) — Directory (tenant) ID of the app registration.
- **Client ID** (required) — Application (client) ID of the app registration.
- **Client secret** (required) — secret created for the app registration.
- **Alias** (optional) — see [Multiple servers](#multiple-servers) below.

After saving, each user will be prompted to authorize the connection with their Azure account.

#### Multiple servers

The tool can be added several times to the same agent, once per server. Each connection has an optional **Alias** that identifies it — if left empty, the server name is used. The alias must be unique among the connections of the agent, and the agent uses it to choose the connection to work with, so give it a meaningful name.

No matter how many connections are configured, the agent always sees the same tools: when there is more than one connection they just take the alias as an extra input.

All connections share the same redirect URI, so a single app registration can be reused for every server. Users still need to be provisioned in the databases of each server, and each connection is authorized separately.

#### Granting access to users

Because the tool acts on behalf of each user, every user must be provisioned in the databases they need to work with — without this the agent won't be able to discover or query the data, even if the OAuth authorization succeeds.

Connect to the target database as the Microsoft Entra admin of the server and create a user from the external provider, then add it to a role with the appropriate permissions:

```sql
CREATE USER "{user-email}" FROM EXTERNAL PROVIDER;
ALTER ROLE db_datareader ADD MEMBER "{user-email}";
```

`db_datareader` grants read-only access, which is the recommended setup. If the user needs to modify data, also add them to `db_datawriter`, or use a broader role such as `db_owner` for full access:

```sql
ALTER ROLE db_owner ADD MEMBER "{user-email}";
```

Repeat this in each database the user should be able to query. The user must also be created in `master`, since the database list is retrieved by querying `sys.databases` there (on Azure SQL you can only connect to databases where your account is provisioned):

```sql
-- connected to master
CREATE USER "{user-email}" FROM EXTERNAL PROVIDER;
```

Note that `sys.databases` only returns the databases the account owns unless it has permission to enumerate all databases on the server (e.g. `VIEW ANY DATABASE`).
