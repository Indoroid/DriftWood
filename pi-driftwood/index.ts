/**
 * BigMoeOnEdge bmoe-server provider for pi-coding-agent.
 *
 * The server is always local and resident: model discovery comes from
 * /v1/models and inference uses the standard OpenAI chat-completions adapter.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { Compile } from "typebox/compile";

const PROVIDER_ID = "bmoe";
const PROVIDER_NAME = "BigMoeOnEdge";
const SERVER_URL = "http://127.0.0.1:8080";
const BASE_URL = `${SERVER_URL}/v1`;
const DEFAULT_CONTEXT_WINDOW = 8192;
const DEFAULT_MAX_TOKENS = 16384;
const FALLBACK_MAX_TOKENS = 1024;
const CONTEXT_GUARD_TOKENS = 256;

const ModelsResponseSchema = Type.Object({
	data: Type.Optional(
		Type.Array(
			Type.Object({
				id: Type.String(),
				meta: Type.Optional(Type.Object({ n_ctx: Type.Optional(Type.Number()) })),
			}),
		),
	),
});
const validateModelsResponse = Compile(ModelsResponseSchema);

type PiModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type MutableModel = Pick<PiModel, "contextWindow" | "maxTokens" | "compat">;

const COMPAT = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	supportsFinishReason: true,
	maxTokensField: "max_tokens",
	supportsStrictMode: false,
	supportsOpenAIGrammarTools: false,
	sendSessionAffinityHeaders: false,
	supportsLongCacheRetention: false,
} satisfies NonNullable<PiModel["compat"]>;

function recoverOutputBudget(payload: unknown, model: MutableModel | undefined): void {
	if (!model || typeof payload !== "object" || payload === null) return;
	const request = payload as { max_tokens?: unknown; messages?: unknown };
	if (typeof request.max_tokens !== "number" || request.max_tokens > 1) return;

	// ponytail: approximate tokens once at the request boundary; the server's tokenizer remains authoritative.
	const promptTokens = Math.ceil(JSON.stringify(request.messages ?? []).length / 3);
	const available = Math.max(1, model.contextWindow - promptTokens - CONTEXT_GUARD_TOKENS);
	request.max_tokens = Math.min(FALLBACK_MAX_TOKENS, model.maxTokens, available);
}

function modelFromApi(model: { id: string; meta?: { n_ctx?: number } }): PiModel {
	const contextWindow = model.meta?.n_ctx ?? DEFAULT_CONTEXT_WINDOW;
	return {
		id: model.id,
		name: model.id,
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow,
		maxTokens: Math.min(DEFAULT_MAX_TOKENS, contextWindow),
		compat: COMPAT,
	} as PiModel;
}

export default async function registerPiDriftwood(pi: ExtensionAPI): Promise<void> {
	let models: PiModel[] = [];

	pi.registerCommand("bmoe-version", {
		description: "Get bmoe-server version",
		handler: async (_args, ctx) => {
			try {
				const response = await fetch(`${SERVER_URL}/`);
				if (!response.ok) {
					ctx.ui.notify(`[bmoe] / returned ${response.status}`, "error");
					return;
				}
				const data = (await response.json()) as { version?: unknown };
				ctx.ui.notify(
					typeof data.version === "string" ? `bmoe-server ${data.version}` : "bmoe-server did not report a version",
					typeof data.version === "string" ? "info" : "warning",
				);
			} catch (error) {
				ctx.ui.notify(`[bmoe] ${error instanceof Error ? error.message : String(error)}`, "error");
			}
		},
	});

	async function refreshProvider(): Promise<void> {
		try {
			const response = await fetch(`${BASE_URL}/models`);
			if (!response.ok) throw new Error(`/v1/models returned ${response.status}`);
			const payload: unknown = await response.json();
			if (!validateModelsResponse.Check(payload)) throw new Error("invalid /v1/models response");
			models = (payload.data ?? []).map(modelFromApi);
			if (!models.length) throw new Error("bmoe-server returned no models");
			pi.registerProvider(PROVIDER_ID, {
				name: PROVIDER_NAME,
				baseUrl: BASE_URL,
				apiKey: "local",
				authHeader: false,
				api: "openai-completions",
				models,
			});
		} catch (error) {
			console.warn(`[bmoe] ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	await refreshProvider();

	pi.on("input", async (event) => {
		if (event.text.trim().toLowerCase() === "/model") await refreshProvider();
	});

	pi.on("before_provider_request", (event, ctx) => {
		if ((event.payload as { model?: unknown })?.model !== ctx.model?.id) return;
		recoverOutputBudget(event.payload, ctx.model);
	});
}
