import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import registerPiDriftwood from "../index.ts";

test("registers bmoe-server without control endpoints or env configuration", async (t) => {
	const requestedUrls: string[] = [];
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async (input) => {
		requestedUrls.push(input.toString());
		return Response.json({ data: [{ id: "model.gguf", meta: { n_ctx: 4096 } }] });
	};
	t.after(() => { globalThis.fetch = originalFetch; });

	const handlers = new Map<string, (...args: any[]) => unknown>();
	const registrations: Array<{ id: string; config: any }> = [];
	const pi = {
		registerCommand() {},
		registerProvider(id: string, config: any) { registrations.push({ id, config }); },
		on(name: string, handler: (...args: any[]) => unknown) { handlers.set(name, handler); },
	} as unknown as ExtensionAPI;

	await registerPiDriftwood(pi);
	assert.equal(registrations[0].id, "bmoe");
	assert.equal(registrations[0].config.name, "BigMoeOnEdge");
	assert.equal(registrations[0].config.models[0].contextWindow, 4096);
	assert.equal(registrations[0].config.models[0].compat.supportsReasoningEffort, true);
	assert.deepEqual(requestedUrls, ["http://127.0.0.1:8080/v1/models"]);

	const payload = { model: "model.gguf", messages: [{ role: "user", content: "hello" }], max_tokens: 1 };
	handlers.get("before_provider_request")?.({ payload }, { model: registrations[0].config.models[0] });
	assert.equal(payload.max_tokens, 1024);
});
