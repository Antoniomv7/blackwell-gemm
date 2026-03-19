#include "benchmarks/common/cli.hpp"
#include "benchmarks/common/csv_writer.hpp"
#include "benchmarks/common/device_info.hpp"
#include "benchmarks/common/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        bw::CliArgs args(argc, argv);

        const int device = args.get_int("device", 0);
        const std::string csv_path = args.get_string("csv", "");
        const std::string txt_path = args.get_string("txt", "");
        const bool quiet = args.get_flag("quiet");

        CUDA_CHECK(cudaSetDevice(device));

        const bw::DeviceInfo info = bw::query_device_info(device);

        if (!quiet) {
            std::cout << "# MB3.1-A Device Inventory\n";
            bw::print_device_info_human(info, std::cout);
        }

        if (!csv_path.empty()) {
            const std::filesystem::path p(csv_path);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }

            bw::CsvWriter writer(csv_path, false);
            writer.write_row(bw::device_info_csv_header());
            writer.write_row(bw::device_info_csv_row(info));
            writer.flush();

            if (!quiet) {
                std::cout << "[mb31_inventory] Wrote CSV: " << csv_path << "\n";
            }
        }

        if (!txt_path.empty()) {
            const std::filesystem::path p(txt_path);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }

            std::ofstream ofs(txt_path);
            if (!ofs.is_open()) {
                throw std::runtime_error("Failed to open TXT output: " + txt_path);
            }

            ofs << "# MB3.1-A Device Inventory\n";
            bw::print_device_info_human(info, ofs);
            ofs.close();

            if (!quiet) {
                std::cout << "[mb31_inventory] Wrote TXT: " << txt_path << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mb31_inventory] ERROR: " << e.what() << "\n";
        return 1;
    }
}