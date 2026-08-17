#include "media_io.h"

#include <fstream>

namespace bmoe {

bool load_media_file(const std::string & path, MediaInput & out, std::string & error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        error = "cannot open media file: " + path;
        return false;
    }
    const std::streamoff end = f.tellg();
    if (end <= 0) {
        error = "media file is empty: " + path;
        return false;
    }
    out.bytes.resize(static_cast<size_t>(end));
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char *>(out.bytes.data()), static_cast<std::streamsize>(out.bytes.size()))) {
        error = "failed to read media file: " + path;
        return false;
    }
    out.name = path;
    return true;
}

} // namespace bmoe
