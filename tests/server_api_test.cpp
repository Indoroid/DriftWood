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
        "stream": true
    })";
    check(parse_completion_request(chat_body, true, defaults, 128, req, error), "chat request parses");
    check(req.messages.size() == 4, "full transcript is retained");
    check(req.messages.size() >= 3 && req.messages[0].role == "system", "system message is retained");
    check(req.messages.size() >= 3 && req.messages[2].role == "assistant", "assistant history is retained");
    check(req.prompt == "Final question", "text content blocks concatenate");
    check(req.n_predict == 17, "max_completion_tokens is honored");
    check(req.stream, "stream flag is honored");
    check(std::fabs(req.sampling.temp - 0.7f) < 0.0001f, "temperature is honored");
    check(std::fabs(req.sampling.top_p - 0.8f) < 0.0001f, "top_p is honored");

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

    const json completion_delta = json::parse(make_stream_delta(false, "cmpl-1", "text_completion", 1, "hi"));
    const json & completion_choice = completion_delta["choices"][0];
    check(completion_choice.value("text", "") == "hi", "completion SSE uses choices[].text");
    check(!completion_choice.contains("delta"), "completion SSE does not use chat delta shape");

    const json chat_delta = json::parse(make_stream_delta(true, "chatcmpl-1", "chat.completion.chunk", 1, "hi"));
    check(chat_delta["choices"][0]["delta"].value("content", "") == "hi", "chat SSE uses choices[].delta.content");

    ToolCall missing_id;
    check(response_tool_call_id(missing_id, 2, "123_4") == "call_123_4_2",
          "missing tool-call id gets deterministic fallback");
    missing_id.id = "model-call-id";
    check(response_tool_call_id(missing_id, 2, "123_4") == "model-call-id", "model tool-call id is preserved");

    const std::string invalid_utf8(1, static_cast<char>(0xFF));
    const json replacement_delta = json::parse(make_stream_delta(false, "cmpl-2", "text_completion", 1, invalid_utf8));
    check(replacement_delta["choices"][0]["text"].is_string(), "invalid UTF-8 token bytes cannot crash SSE JSON");

    ApiCompletionRequest invalid;
    error.clear();
    check(!parse_completion_request(R"({"messages":[],"stream":true})", true, defaults, 128, invalid, error),
          "empty chat transcript is rejected");
    error.clear();
    check(!parse_completion_request(R"({"prompt":"x","temperature":3})", false, defaults, 128, invalid, error),
          "out-of-range temperature is rejected");

    int sockets[2] = {-1, -1};
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair opens");
    if (sockets[0] >= 0 && sockets[1] >= 0) {
        const std::string body = R"({"prompt":"case insensitive"})";
        const std::string wire =
            "POST /v1/completions HTTP/1.1\r\nCONTENT-LENGTH: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        http_write(sockets[0], wire);
        shutdown(sockets[0], SHUT_WR);
        std::string raw;
        check(read_request(sockets[1], raw), "uppercase Content-Length is accepted");
        HttpRequest http;
        check(parse_http_request(raw, http) && http.body == body, "HTTP body is retained exactly");
        close(sockets[0]);
        close(sockets[1]);
    }

    if (failures != 0) return 1;
    std::cout << "server API tests passed\n";
    return 0;
}
