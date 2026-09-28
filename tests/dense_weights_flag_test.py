"""Both frontends must reject a dense-weights policy that nothing would apply.

The policy is applied by the expert streamer, so without --moe-stream it used to be accepted and
silently ignored. `anon` is also the default, so the rule must follow the flag, not the value.
Config validation runs before the model is opened, so no model is needed: the path does not exist.
Usage: dense_weights_flag_test.py CLI SERVER."""
import subprocess
import sys

cli, server = sys.argv[1], sys.argv[2]
failures = 0
MESSAGE = "moe.dense_weights requires moe.enabled"


def run(binary, args):
    proc = subprocess.run([binary, "-m", "/nonexistent/model.gguf", "-p", "hi", "-n", "1"] + args,
                          capture_output=True, timeout=60)
    return proc.returncode, proc.stderr.decode("utf-8", errors="replace")


def check(ok, name, detail=""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n  {detail}" if not ok and detail else ""))
    failures += 0 if ok else 1


for name, binary in (("meitte-cli", cli), ("meitte-server", server)):
    for flag in (["--dense-weights", "anon"], ["--dense-weights", "mmap"], ["--no-warm-dense"], ["--dense-odirect"]):
        rc, err = run(binary, flag)
        check(rc != 0 and MESSAGE in err, f"{name} {' '.join(flag)} without --moe-stream is rejected", err[-300:])
    # With streaming on, validation accepts the policy; the run then fails only on the missing file.
    rc, err = run(binary, ["--moe-stream", "--dense-weights", "anon"])
    check(MESSAGE not in err, f"{name} --moe-stream --dense-weights anon passes validation", err[-300:])
    # The default policy without the flag is not an error either.
    rc, err = run(binary, [])
    check(MESSAGE not in err, f"{name} without --dense-weights passes validation", err[-300:])

sys.exit(1 if failures else 0)
