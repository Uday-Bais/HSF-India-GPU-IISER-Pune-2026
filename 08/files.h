#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>

inline void fail_file_operation(const char *path, const char *message) {
    std::cerr << message << ": " << path << std::endl;
    std::exit(1);
}

inline void read_values_from_file(const char *path, void *buffer, size_t bytes) {
    std::ifstream binary_stream(path, std::ios::binary | std::ios::ate);
    if (!binary_stream) {
        fail_file_operation(path, "Unable to open input file");
    }

    const std::streamsize size = binary_stream.tellg();
    binary_stream.seekg(0, std::ios::beg);

    if (size == static_cast<std::streamsize>(bytes)) {
        if (!binary_stream.read(static_cast<char *>(buffer), size)) {
            fail_file_operation(path, "Unable to read binary input file");
        }
        return;
    }

    binary_stream.close();

    if (bytes % sizeof(float) != 0) {
        fail_file_operation(path, "Input size is not compatible with float parsing");
    }

    std::ifstream text_stream(path);
    if (!text_stream) {
        fail_file_operation(path, "Unable to reopen input file for text parsing");
    }

    float *values = static_cast<float *>(buffer);
    const size_t count = bytes / sizeof(float);

    for (size_t index = 0; index < count; ++index) {
        if (!(text_stream >> values[index])) {
            fail_file_operation(path, "Input file does not contain enough float values");
        }
    }
}

inline void write_values_to_file(const char *path, const void *buffer, size_t bytes) {
    std::ofstream output_stream(path, std::ios::binary | std::ios::trunc);
    if (!output_stream) {
        fail_file_operation(path, "Unable to open output file");
    }

    output_stream.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(bytes));
    if (!output_stream) {
        fail_file_operation(path, "Unable to write output file");
    }
}