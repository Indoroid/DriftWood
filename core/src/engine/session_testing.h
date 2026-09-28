#pragma once

// Test-only controls for failure paths that the tiny test models cannot produce on their own.
// Production code never calls these. Internal header: not installed, not part of the public API.

namespace meitte::testing {

// Make the next `count` partial KV removals report failure without touching the memory — exactly
// what llama.cpp's recurrent and hybrid caches do when they cannot keep a prefix. The attention-only
// test models never refuse a removal, so rollback after a refused removal is otherwise unreachable.
void fail_next_kv_removals(int count);

// Refusals armed by fail_next_kv_removals() that no removal has consumed yet.
int pending_kv_removal_failures();

} // namespace meitte::testing
