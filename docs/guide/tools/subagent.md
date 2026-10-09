### Subagent

Delegate parts of a request to another Tero agent. Use this to keep each agent focused on a specialty — for example an orchestrator that routes work to a researcher, a writer, and a reviewer — instead of packing every skill into one prompt.

The parent agent decides when to delegate. Each subagent runs with its own model, instructions, and tools, then returns a result the parent can use to continue. If a subagent is also Advanced and has its own subagents, it can delegate further.

::: warning Advanced mode only
Subagents can only be configured on **Advanced** agents. Switching the parent back to Basic removes the configured subagents.
:::

#### Setup

1. Set **Agent mode** to Advanced in the agent's advanced settings.
2. Add the **Subagent** tool.
3. Select the agent to delegate to. You can search by name.

The selected agent's **description** is what the parent uses to decide when to delegate, so keep it specific about what that agent is for.

::: info Who you can select
You cannot select the agent you are configuring as a subagent of itself. You can select other agents, including ones that already have their own subagents.

If A delegates to B and B delegates back to A, Tero skips the loop at runtime so the chain does not run forever.
:::

::: tip Multiple subagents
The tool can be added several times to the same agent, once per subagent. Each instance appears with the selected agent's name.
:::

#### Sharing

Before you share the parent with a team, each subagent must already be shared with that same team, or with the Global team. Private subagents cannot be used by a published orchestrator.

::: info Thought process
When the parent delegates, the thought process shows the subagent by name (for example, “Delegating to Researcher”) and the tools that subagent runs.
:::

::: tip Budget
Each subagent uses its own model, so you can keep a stronger model on the parent and delegate simpler tasks to cheaper, faster ones. Delegated work still counts toward the chat’s budget.
:::
