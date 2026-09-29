/* Binary compatibility of the C ABI with callers compiled against older headers. Each old caller is
 * reproduced by its exact struct layout, with the bytes an old compiler would leave uninitialized set
 * to garbage, and passed to the current library. */
#include "bmoe/meitte.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char * name) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

/* meitte_config as published before the dense-stream fields: it ends at release_mmap, and its sizeof
 * includes tail padding where the current layout places dense_stream. */
typedef struct legacy_config_before_dense {
    size_t struct_size;
    uint32_t abi_version;
    const char * model_path;
    const char * projector_path;
    int32_t context_size;
    int32_t threads;
    int32_t streaming;
    int32_t cache_mb;
    int32_t chat;
    int32_t kv_unified;
    int32_t speculation;
    int32_t context_grow;
    int32_t context_summarize;
    int32_t context_trim;
    int32_t context_min;
    int32_t context_max;
    float video_fps;
    int32_t video_max_frames;
    const char * ffmpeg_bin_dir;
    uint64_t media_max_bytes;
    int32_t release_mmap;
} legacy_config_before_dense;

int main(void) {
    const size_t release_end = offsetof(meitte_config, release_mmap) + sizeof(int32_t);
    check(offsetof(legacy_config_before_dense, release_mmap) == offsetof(meitte_config, release_mmap),
          "legacy layout mirrors the published prefix");

    /* An older, shorter struct: init writes defaults and never touches the bytes after it. */
    struct {
        legacy_config_before_dense config;
        unsigned char canary[64];
    } legacy;
    memset(&legacy, 0xAB, sizeof(legacy));
    meitte_config_init((meitte_config *) &legacy.config, sizeof(legacy.config));
    int canary_intact = 1;
    for (size_t i = 0; i < sizeof(legacy.canary); ++i)
        canary_intact &= legacy.canary[i] == 0xAB;
    check(canary_intact, "config init stays inside the caller's struct");
    check(legacy.config.struct_size == sizeof(legacy.config) && legacy.config.abi_version == MEITTE_ABI_VERSION &&
              legacy.config.context_size == 2048 && legacy.config.chat == 1,
          "config init writes the defaults that fit");

    /* The old caller's tail padding holds garbage. Read as dense_stream, it would request the
     * removed dense streaming, which meitte_open rejects before any load is attempted. */
    legacy.config.model_path = "/nonexistent/meitte-abi-test.gguf";
    legacy.config.streaming = 1;
    memset((unsigned char *) &legacy.config + release_end, 0xFF, sizeof(legacy.config) - release_end);
    char error[256];
    meitte_session * session = meitte_open((const meitte_config *) &legacy.config, error, sizeof(error));
    check(!session && strstr(error, "failed to load model") != NULL,
          "legacy config padding is not read as dense_stream");
    if (!strstr(error, "failed to load model")) fprintf(stderr, "legacy config error: %s\n", error);
    if (session) meitte_close(session);

    /* Request init against a shorter struct: the original request ends at media_count. */
    struct {
        meitte_request request;
        unsigned char canary[64];
    } request;
    memset(&request, 0xAB, sizeof(request));
    const size_t short_request = offsetof(meitte_request, clear_kv) + sizeof(int32_t);
    meitte_request_init(&request.request, short_request);
    const unsigned char * tail = (const unsigned char *) &request.request + short_request;
    int tail_intact = 1;
    for (size_t i = 0; i < sizeof(request.request) - short_request; ++i)
        tail_intact &= tail[i] == 0xAB;
    check(tail_intact && request.request.struct_size == short_request && request.request.max_tokens == 128 &&
              request.request.clear_kv == 1,
          "request init honours a shorter struct");

    /* A size too small to hold the header is left untouched. */
    meitte_config tiny;
    memset(&tiny, 0xCD, sizeof(tiny));
    meitte_config_init(&tiny, sizeof(size_t));
    check(((const unsigned char *) &tiny)[0] == 0xCD, "init ignores a size without the header");
    return failures ? 1 : 0;
}
