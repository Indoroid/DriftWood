#define main bmoe_server_test_entry
#include "../cli/server_main.cpp"
#undef main

#include <cmath>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    SamplingConfig defaults;
    defaults.temp = 0.25f;
    defaults.top_p = 0.9f;

    ApiCompletionRequest req;
    std::string error;
    const std::string chat_body = R"({
        "messages": [
            {"role":"system", "content":"Answer briefly."},
            {"role":"user", "content":"First"},
            {"role":"assistant", "content":"Prior answer"},
            {"role":"user", "content":[{"type":"text","text":"Final "},{"type":"text","text":"question"}]}
        ],
        "max_completion_tokens": 17,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": true,
        "model": "local-model",
        "stream_options": {"include_usage": true}
    })";
    check(parse_completion_request(chat_body, true, defaults, 128, req, error), "chat request parses");
    check(req.messages.size() == 4, "full transcript is retained");
    check(req.messages.size() >= 3 && req.messages[0].role == "system", "system message is retained");
    check(req.messages.size() >= 3 && req.messages[2].role == "assistant", "assistant history is retained");
    check(req.prompt == "Final question", "text content blocks concatenate");
    check(req.n_predict == 17, "max_completion_tokens is honored");
    check(req.stream, "stream flag is honored");
    check(req.model == "local-model", "requested model name is retained");
    check(req.stream_include_usage, "stream usage option is honored");
    check(std::fabs(req.sampling.temp - 0.7f) < 0.0001f, "temperature is honored");
    check(std::fabs(req.sampling.top_p - 0.8f) < 0.0001f, "top_p is honored");

    ApiCompletionRequest controls;
    error.clear();
    check(parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"HIGH","reasoning_budget_tokens":32,"chat_template_kwargs":{"mode":"fast","flag":true,"count":7}})",
              true, defaults, 128, controls, error),
          "reasoning controls and generic kwargs parse");
    check(!controls.think.has_value() && controls.reasoning_effort == "high",
          "reasoning effort implies request-local enablement");
    check(controls.reasoning_effort == "high", "standard reasoning effort names normalize to lowercase");
    check(controls.reasoning_budget_tokens && *controls.reasoning_budget_tokens == 32,
          "reasoning budget is request-local");
    check(controls.chat_template_kwargs.at("mode") == "\"fast\"" &&
              controls.chat_template_kwargs.at("flag") == "true" && controls.chat_template_kwargs.at("count") == "7",
          "generic kwargs preserve JSON literals");

    error.clear();
    check(parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"think":false,"reasoning_effort":"high"})",
              true, defaults, 128, controls, error),
          "explicit thinking off accepts and clears effort");
    check(controls.think.has_value() && !*controls.think && controls.reasoning_effort.empty(),
          "explicit thinking off wins over effort");

    error.clear();
    check(!parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"think":true,"chat_template_kwargs":{"enable_thinking":false}})",
              true, defaults, 128, controls, error),
          "conflicting typed and generic thinking controls are rejected");
    error.clear();
    check(!parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":[]})",
              true, defaults, 128, controls, error),
          "non-object generic kwargs are rejected");
    error.clear();
    check(!parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"reasoning_effort":""}})",
              true, defaults, 128, controls, error),
          "empty generic reasoning effort is rejected");
    error.clear();
    check(parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"thinking_budget_tokens":0})",
              true, defaults, 128, controls, error),
          "reasoning budget alias parses");
    check(controls.reasoning_budget_tokens && *controls.reasoning_budget_tokens == 0,
          "zero ends reasoning immediately");
    error.clear();
    check(!parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"reasoning_budget_tokens":-2})",
              true, defaults, 128, controls, error),
          "invalid negative reasoning budget is rejected");
    error.clear();
    check(!parse_completion_request(
              R"({"messages":[{"role":"user","content":"x"}],"reasoning_budget_tokens":3,"thinking_budget_tokens":4})",
              true, defaults, 128, controls, error),
          "conflicting reasoning budget aliases are rejected");

    ApiCompletionRequest default_length;
    error.clear();
    check(parse_completion_request(R"({"prompt":"x"})", false, defaults, 73, default_length, error),
          "completion request parses");
    check(default_length.n_predict == 73, "server --n-predict supplies request default");

    const std::string tool_body = R"({
        "messages": [
            {"role":"user", "content":"Check Paris weather"},
            {"role":"assistant", "content":null, "tool_calls":[{
                "id":"call-1", "type":"function",
                "function":{"name":"weather", "arguments":"{\"city\":\"Paris\"}"}
            }]},
            {"role":"tool", "tool_call_id":"call-1", "content":"sunny"},
            {"role":"user", "content":"Summarize"}
        ],
        "tools":[{"type":"function","function":{
            "name":"weather", "description":"Read weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
        }}],
        "tool_choice":"required",
        "parallel_tool_calls":true
    })";
    ApiCompletionRequest tool_req;
    error.clear();
    check(parse_completion_request(tool_body, true, defaults, 64, tool_req, error), "tool conversation parses");
    check(tool_req.messages.size() == 4, "tool conversation retains all messages");
    check(tool_req.messages.size() > 1 && tool_req.messages[1].tool_calls.size() == 1,
          "assistant tool call is retained");
    check(tool_req.messages.size() > 2 && tool_req.messages[2].tool_call_id == "call-1",
          "tool result call id is retained");
    check(tool_req.tools.size() == 1 && tool_req.tools[0].name == "weather", "tool definition is retained");
    check(tool_req.tool_choice == ChatToolChoice::Required, "required tool choice is retained");
    check(tool_req.parallel_tool_calls, "parallel tool setting is retained");

    const json completion_delta =
        json::parse(make_stream_delta(false, "cmpl-1", "text_completion", 1, "local-model", "hi", {}, true));
    const json & completion_choice = completion_delta["choices"][0];
    check(completion_choice.value("text", "") == "hi", "completion SSE uses choices[].text");
    check(!completion_choice.contains("delta"), "completion SSE does not use chat delta shape");
    check(completion_delta.value("model", "") == "local-model", "SSE echoes the requested model");
    check(completion_delta.contains("system_fingerprint") && completion_delta["system_fingerprint"].is_null(),
          "SSE includes a null system fingerprint");
    check(completion_delta.contains("usage") && completion_delta["usage"].is_null(),
          "ordinary SSE chunks carry null usage when requested");

    const json chat_delta =
        json::parse(make_stream_delta(true, "chatcmpl-1", "chat.completion.chunk", 1, "local-model", "hi"));
    check(chat_delta["choices"][0]["delta"].value("content", "") == "hi", "chat SSE uses choices[].delta.content");
    check(chat_delta["choices"][0].contains("logprobs") && chat_delta["choices"][0]["logprobs"].is_null(),
          "chat SSE includes null logprobs");

    StreamTextState streamed;
    check(stream_suffix("Hel", streamed.text) == "Hel", "first parsed text delta is emitted");
    check(stream_suffix("Hello", streamed.text) == "lo", "parsed text emits only the new suffix");
    check(stream_suffix("Hello", streamed.text).empty(), "unchanged parsed text emits nothing");
    check(stream_suffix("think", streamed.reasoning) == "think", "reasoning delta is tracked separately");
    const json reasoning_delta = json::parse(
        make_stream_delta(true, "chatcmpl-2", "chat.completion.chunk", 1, "local-model", "answer", "thought"));
    check(reasoning_delta["choices"][0]["delta"].value("reasoning_content", "") == "thought",
          "chat SSE carries reasoning separately");

    ToolCall missing_id;
    check(response_tool_call_id(missing_id, 2, "123_4") == "call_123_4_2",
          "missing tool-call id gets deterministic fallback");
    missing_id.id = "model-call-id";
    check(response_tool_call_id(missing_id, 2, "123_4") == "model-call-id", "model tool-call id is preserved");

    const std::string invalid_utf8(1, static_cast<char>(0xFF));
    const json replacement_delta =
        json::parse(make_stream_delta(false, "cmpl-2", "text_completion", 1, "local-model", invalid_utf8));
    check(replacement_delta["choices"][0]["text"].is_string(), "invalid UTF-8 token bytes cannot crash SSE JSON");

    ApiCompletionRequest invalid;
    error.clear();
    check(!parse_completion_request(R"({"messages":[],"stream":true})", true, defaults, 128, invalid, error),
          "empty chat transcript is rejected");
    error.clear();
    check(!parse_completion_request(R"({"prompt":"x","temperature":3})", false, defaults, 128, invalid, error),
          "out-of-range temperature is rejected");
    error.clear();
    check(!parse_completion_request(R"({"prompt":"x","model":""})", false, defaults, 128, invalid, error),
          "empty model is rejected");
    error.clear();
    check(!parse_completion_request(R"({"prompt":"x","stream_options":{"include_usage":"yes"}})", false, defaults, 128,
                                    invalid, error),
          "non-boolean stream usage option is rejected");

    RunResult usage_result;
    usage_result.summary.n_prompt = 3;
    usage_result.summary.n_generated = 2;
    const json usage = completion_usage(usage_result);
    check(usage.value("prompt_tokens", -1) == 3 && usage.value("completion_tokens", -1) == 2 &&
              usage.value("total_tokens", -1) == 5,
          "usage has OpenAI token totals");
    const json usage_chunk =
        make_stream_usage("chatcmpl-usage", "chat.completion.chunk", 1, "local-model", usage_result);
    check(usage_chunk["choices"].empty() && usage_chunk["usage"].value("total_tokens", -1) == 5,
          "final SSE usage chunk has empty choices and populated usage");
    check(std::string(completion_finish_reason(usage_result, 2)) == "length", "token-limit completions report length");
    check(std::string(completion_finish_reason(usage_result, 3)) == "stop", "early completions report stop");

    int sockets[2] = {-1, -1};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair opens");
    if (sockets[0] >= 0 && sockets[1] >= 0) {
        const std::string body = R"({"prompt":"case insensitive"})";
        const std::string wire =
            "POST /v1/completions HTTP/1.1\r\nCONTENT-LENGTH: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        http_write(sockets[0], wire);
        shutdown(sockets[0], SHUT_WR);
        std::string raw;
        check(read_request(sockets[1], raw, 1024 * 1024), "uppercase Content-Length is accepted");
        HttpRequest http;
        check(parse_http_request(raw, http) && http.body == body, "HTTP body is retained exactly");
        close(sockets[0]);
        close(sockets[1]);
    }

    if (failures != 0) return 1;
    std::cout << "server API tests passed\n";
    return 0;
}
