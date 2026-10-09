### Docs

Upload files to your agent so it can use their content when answering questions. The agent searches the uploaded files using semantic similarity and grounds its responses in what the documents actually say, citing the source files.

This is useful for agents that need to reason over technical documentation, test plans, specifications, runbooks, or any domain-specific knowledge that isn't in the model's training data.

The agent can:

- **Answer questions** from uploaded documents using semantic search across all files
- **Cite sources** — responses reference the specific file sections they were drawn from
- **Reason across multiple files** — the agent considers all uploaded files together when answering

Supported file types include PDF, plain text, Markdown, and common document formats.

::: tip Basic vs Advanced PDF processing

**Basic** processing is faster and works well for most PDFs. **Advanced** processing (when available) handles complex PDFs with tables, multi-column layouts, and embedded images more accurately, but it costs more and takes longer.

Use Advanced only when the agent consistently misses or misreads content from your PDFs.
:::

::: tip Verify answers

**Verify answers** is on by default. When enabled, each answer is checked against the uploaded files and source links are added, so the response stays grounded in the documents instead of invented facts. This extra pass takes longer and uses more budget.

Turn it off for faster, cheaper responses. Stronger models hallucinate less, so verification matters most with smaller models or when you need cited sources.
:::
