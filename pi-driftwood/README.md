# pi-driftwood

The pi-coding-agent extension for BigMoeOnEdge `bmoe-server`.

It uses the server's fixed local endpoint, `http://127.0.0.1:8080`, and does not read environment variables. Start `bmoe-server`, then run:

```bash
pi -e ./pi-driftwood/index.ts
```

The `bmoe` provider uses pi's OpenAI-compatible adapter for streaming chat, multi-turn messages, reasoning controls, function tools, tool calls, tool results, sampling, and usage/finish metadata. `/model` refreshes `/v1/models`; `/bmoe-version` checks `/`.

Run the local check with:

```bash
npm install
npm test
```
