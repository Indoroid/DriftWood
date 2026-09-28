#include "subprocess.h"

#include <cstdio>
#include <memory>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace meitte::pio {

CaptureResult run_capture(const std::vector<std::string> & argv,
                          const std::vector<uint8_t> & input,
                          size_t max_bytes,
                          std::chrono::milliseconds timeout,
                          const std::function<bool()> & stopped,
                          std::vector<uint8_t> & output) {
    output.clear();
    CaptureResult result;
#ifdef _WIN32
    (void) argv;
    (void) input;
    (void) max_bytes;
    (void) timeout;
    (void) stopped;
    return result;
#else
    auto stop = [&](CaptureStatus status) {
        result.status = status;
        return result;
    };
    if (argv.empty()) return stop(CaptureStatus::SpawnFailed);

    // A private temporary input prevents a full stdin pipe from blocking stdout drainage.
    struct FileCloser {
        void operator()(FILE * value) const {
            if (value) std::fclose(value);
        }
    };
    std::unique_ptr<FILE, FileCloser> file(std::tmpfile());
    if (!file || std::fwrite(input.data(), 1, input.size(), file.get()) != input.size() ||
        std::fflush(file.get()) != 0 || std::fseek(file.get(), 0, SEEK_SET) != 0)
        return stop(CaptureStatus::StageFailed);
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) return stop(CaptureStatus::PipeFailed);
    struct Pipe {
        int read_fd, write_fd;
        ~Pipe() {
            close(read_fd);
            if (write_fd >= 0) close(write_fd);
        }
    } pipe_owner{pipe_fd[0], pipe_fd[1]};
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return stop(CaptureStatus::SpawnInitFailed);
    struct Actions {
        posix_spawn_file_actions_t & value;
        ~Actions() { posix_spawn_file_actions_destroy(&value); }
    } actions_owner{actions};
    int rc = posix_spawn_file_actions_adddup2(&actions, fileno(file.get()), STDIN_FILENO);
    rc |= posix_spawn_file_actions_adddup2(&actions, pipe_fd[1], STDOUT_FILENO);
    rc |= posix_spawn_file_actions_addclose(&actions, pipe_fd[0]);
    rc |= posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    if (rc) return stop(CaptureStatus::SpawnConfigFailed);
    std::vector<std::string> args = argv;
    std::vector<char *> args_c;
    for (auto & arg : args)
        args_c.push_back(arg.data());
    args_c.push_back(nullptr);
    pid_t pid;
    rc = posix_spawnp(&pid, args_c[0], &actions, nullptr, args_c.data(), environ);
    if (rc) {
        result.spawn_error = rc;
        return stop(CaptureStatus::SpawnFailed);
    }
    // Kill and reap the child on every early return.
    struct Child {
        pid_t pid;
        ~Child() {
            if (pid > 0) {
                kill(pid, SIGKILL);
                while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
                }
            }
        }
    } child{pid};
    close(pipe_owner.write_fd);
    pipe_owner.write_fd = -1;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto should_stop = [&] { return (stopped && stopped()) || std::chrono::steady_clock::now() >= deadline; };
    for (;;) {
        if (should_stop()) return stop(CaptureStatus::Stopped);
        pollfd descriptor{pipe_fd[0], POLLIN, 0};
        rc = poll(&descriptor, 1, 100);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0) return stop(CaptureStatus::PollFailed);
        if (!rc) continue;
        uint8_t buffer[16384];
        ssize_t n = read(pipe_fd[0], buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return stop(CaptureStatus::ReadFailed);
        if (!n) break;
        if ((size_t) n > max_bytes - output.size()) return stop(CaptureStatus::LimitExceeded);
        output.insert(output.end(), buffer, buffer + n);
    }
    int status = 0;
    while ((rc = waitpid(pid, &status, WNOHANG)) == 0) {
        if (should_stop()) return stop(CaptureStatus::WaitStopped);
        poll(nullptr, 0, 10);
    }
    if (rc < 0) return stop(CaptureStatus::WaitFailed);
    child.pid = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return stop(CaptureStatus::ExitFailed);
    return stop(CaptureStatus::Ok);
#endif
}

} // namespace meitte::pio
