#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bw {

class CliArgs {
public:
    CliArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            if (arg.rfind("--", 0) == 0) {
                const auto eq = arg.find('=');
                if (eq != std::string::npos) {
                    const std::string key = arg.substr(2, eq - 2);
                    const std::string val = arg.substr(eq + 1);
                    opts_[key] = val;
                } else {
                    const std::string key = arg.substr(2);
                    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                        opts_[key] = std::string(argv[i + 1]);
                        ++i;
                    } else {
                        opts_[key] = "1";
                    }
                }
            } else {
                positional_.push_back(arg);
            }
        }
    }

    bool has(const std::string& key) const {
        return opts_.find(key) != opts_.end();
    }

    std::string get_string(const std::string& key,
                           const std::string& default_value = "") const {
        auto it = opts_.find(key);
        if (it == opts_.end()) return default_value;
        return it->second;
    }

    int get_int(const std::string& key, int default_value) const {
        auto it = opts_.find(key);
        if (it == opts_.end()) return default_value;
        return std::stoi(it->second);
    }

    long long get_ll(const std::string& key, long long default_value) const {
        auto it = opts_.find(key);
        if (it == opts_.end()) return default_value;
        return std::stoll(it->second);
    }

    float get_float(const std::string& key, float default_value) const {
        auto it = opts_.find(key);
        if (it == opts_.end()) return default_value;
        return std::stof(it->second);
    }

    bool get_flag(const std::string& key) const {
        auto it = opts_.find(key);
        if (it == opts_.end()) return false;

        const std::string& v = it->second;
        return (v == "1" || v == "true" || v == "on" || v == "yes");
    }

    const std::vector<std::string>& positional() const {
        return positional_;
    }

private:
    std::unordered_map<std::string, std::string> opts_;
    std::vector<std::string> positional_;
};

}  // namespace bw