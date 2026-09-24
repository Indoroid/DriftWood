#ifndef MEITTE_H
#define MEITTE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MEITTE_SHARED)
#ifdef MEITTE_BUILDING_LIBRARY
#define MEITTE_API __declspec(dllexport)
#else
#define MEITTE_API __declspec(dllimport)
#endif
#elif defined(MEITTE_SHARED) && (defined(__GNUC__) || defined(__clang__))
#define MEITTE_API __attribute__((visibility("default")))
#else
#define MEITTE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MEITTE_ABI_VERSION 1
#define MEITTE_CONTEXT_OFF 0
#define MEITTE_CONTEXT_ON 1
#define MEITTE_CONTEXT_AUTO 2
#define MEITTE_MEDIA_AUTO 0
#define MEITTE_MEDIA_IMAGE 1
#define MEITTE_MEDIA_AUDIO 2
#define MEITTE_MEDIA_VIDEO 3
typedef struct meitte_session meitte_session;
typedef struct meitte_result meitte_result;

typedef struct meitte_config {
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
    /* 0: none; 1: n-gram; 2: MTP. */
    int32_t speculation;
    /* Context modes: 0: off; 1: on at capacity; 2: automatic preflight. */
    int32_t context_grow;
    int32_t context_summarize;
    int32_t context_trim;
    int32_t context_min;
    int32_t context_max;
    float video_fps;
    int32_t video_max_frames;
    const char * ffmpeg_bin_dir;
    uint64_t media_max_bytes;
    /* Nonzero releases the model-file mapping after safe expert-stream initialization. */
    int32_t release_mmap;
    /* Dense CPU streaming. Fields after struct_size are optional for older callers. */
    int32_t dense_stream;
    int32_t dense_resident_mb;
    int32_t dense_window_mb;
    int32_t dense_io_lanes;
    int32_t dense_overlap;
    int32_t dense_two_wave;
} meitte_config;

typedef struct meitte_media {
    const uint8_t * data;
    size_t size;
    const char * name;
    /* 0: detect; 1: image; 2: audio; 3: video. */
    int32_t kind;
} meitte_media;

typedef struct meitte_request {
    size_t struct_size;
    uint32_t abi_version;
    const char * prompt;
    int32_t max_tokens;
    int32_t clear_kv;
    const meitte_media * media;
    size_t media_count;
} meitte_request;

/* The piece is borrowed for this call only. Return nonzero to cancel generation.
 * Do not call generate or destroy on the session from its callback. */
typedef int (*meitte_token_callback)(const char * piece, size_t size, void * user_data);

MEITTE_API meitte_config meitte_default_config(void);
MEITTE_API meitte_request meitte_default_request(void);
MEITTE_API uint32_t meitte_abi_version(void);
MEITTE_API const char * meitte_version(void);
/* Inputs are borrowed for the call. The error buffer belongs to the caller.
 * A struct can end after any field, provided it includes model_path or prompt.
 * Missing trailing fields use the corresponding default. A session permits one operation at a
 * time, except for cancel. */
MEITTE_API meitte_session * meitte_open(const meitte_config * config, char * error, size_t error_capacity);
MEITTE_API meitte_result * meitte_generate(meitte_session * session,
                                           const meitte_request * request,
                                           meitte_token_callback callback,
                                           void * user_data);
MEITTE_API void meitte_cancel(meitte_session * session);
/* Destroy only after generate and cancel calls have returned. */
MEITTE_API void meitte_close(meitte_session * session);
/* Result strings are borrowed until result_free. A null result means allocation failed. */
MEITTE_API int meitte_result_ok(const meitte_result * result);
MEITTE_API int meitte_result_cancelled(const meitte_result * result);
MEITTE_API const char * meitte_result_error(const meitte_result * result);
MEITTE_API const char * meitte_result_text(const meitte_result * result);
MEITTE_API const char * meitte_result_reasoning(const meitte_result * result);
MEITTE_API void meitte_result_free(meitte_result * result);

#ifdef __cplusplus
}
#endif
#endif
