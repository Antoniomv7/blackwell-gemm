#pragma once

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bw {

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path, bool append = false) {
        const std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ios::openmode mode = std::ios::out;
        if (append) mode |= std::ios::app;

        file_.open(path, mode);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + path);
        }
    }

    void write_header(std::initializer_list<std::string> cols) {
        write_row(cols);
    }

    void write_row(std::initializer_list<std::string> cols) {
        bool first = true;
        for (const auto& c : cols) {
            if (!first) file_ << ",";
            file_ << escape(c);
            first = false;
        }
        file_ << "\n";
    }

    void write_row(const std::vector<std::string>& cols) {
        bool first = true;
        for (const auto& c : cols) {
            if (!first) file_ << ",";
            file_ << escape(c);
            first = false;
        }
        file_ << "\n";
    }

    void flush() {
        file_.flush();
    }

private:
    static std::string escape(const std::string& s) {
        const bool need_quotes =
            (s.find(',') != std::string::npos) ||
            (s.find('"') != std::string::npos) ||
            (s.find('\n') != std::string::npos);

        if (!need_quotes) return s;

        std::string out = "\"";
        for (char ch : s) {
            if (ch == '"') out += "\"\"";
            else out += ch;
        }
        out += "\"";
        return out;
    }

    std::ofstream file_;
};

template <typename T>
inline std::string to_csv_string(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string to_csv_string(const std::string& value) {
    return value;
}

inline std::string to_csv_string(const char* value) {
    return std::string(value);
}

}  // namespace bw