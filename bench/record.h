/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_BENCH_RECORD_H_
#define CARTA_ZARR_BENCH_RECORD_H_

// What a run writes down: one CSV row per operation, and what it needs to know to resume.
//
// The columns are one list in record.cc, the header and the row both made from it, so the two cannot
// drift. Changing what a column means, or which columns there are, bumps kCsvVersion; a run refuses
// to append to a CSV whose header is not its own.

#include "options.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace carta::zarr::bench {

inline constexpr int kCsvVersion = 7;

// What a dataset's bench-manifest.json says, for the columns the library cannot answer. Empty for a
// dataset the generator did not write, which the bench reads as well as any other.
struct DatasetFacts {
    std::string identity_hash;
    std::string codec;
    std::string stripe_effective;
    std::string compression_ratio;
};

DatasetFacts ReadManifest(const std::string& dataset);

struct Row {
    // The run.
    std::string run_id;
    std::string run_key;
    std::string label;
    std::string timestamp_utc;
    std::string host;
    std::string bench_commit;
    std::string build_type;
    // The library's tuning overrides, "default" for none. See cmake/TuningOverrides.cmake.
    std::string tuning;
    // The dataset.
    std::string dataset;
    std::string dataset_identity_hash;
    std::string image_id;
    std::string shape;
    std::string chunk_shape;
    std::string shard_shape;
    std::string codec;
    std::string stripe_effective;
    std::string compression_ratio;
    // The settings.
    std::string mode;
    unsigned processes = 1;
    unsigned io_threads = 0;
    unsigned decode_threads = 0;
    std::string cache_bytes;
    std::size_t read_budget_bytes = 0;
    std::uint64_t seed = 0;
    unsigned ops = 0;
    // Each only for the mode it shapes, and empty for the others.
    std::string region_fraction;
    std::string histogram_method;
    std::string animation_frames;
    std::string animation_fps;
    std::string animation_prefetch;
    long long trial_timeout_s = 0;
    std::string cold_method;
    bool cold_ok = false;
    // The operation.
    unsigned trial = 0;
    unsigned process_index = 0;
    std::optional<unsigned> op_index;
    std::string position;
    bool overlap = false;
    bool shares_chunks = false;
    // What came of it.
    std::string status;
    std::string error;
    std::optional<double> seconds;
    std::optional<double> t_start_s;
    std::optional<double> t_end_s;
    std::uint64_t elements = 0;
    std::uint64_t logical_bytes = 0;
    std::optional<std::uint64_t> storage_read_bytes;
    std::optional<double> user_cpu_s;
    std::optional<double> sys_cpu_s;
    std::optional<std::uint64_t> peak_rss_bytes;
    std::optional<std::uint64_t> checksum;
    // How many frames an animation played, its frames after the first, and the first on its own. See
    // FrameStats.
    std::optional<unsigned> frames_played;
    std::optional<double> frame_first_s;
    std::optional<double> frame_median_s;
    std::optional<double> frame_max_s;
    std::optional<unsigned> late_frames;
    std::optional<double> late_max_s;
    std::optional<unsigned> prefetches;
    std::optional<unsigned> late_prefetches;
};

// The settings columns of a row, from the options and the dataset: everything a row says before
// anything has been read.
Row RowTemplate(const RunOptions& options, Mode mode, ColdMethod cold, const DatasetFacts& facts,
                const std::string& run_id);

std::string CsvHeader();
std::string FormatRow(const Row& row);
std::vector<std::string> ParseCsvLine(const std::string& line);

// The fields of a row this run's own code reads back: to resume, and to summarise a trial.
struct RowSummary {
    std::string run_key;
    unsigned trial = 0;
    std::string status;
    std::optional<double> seconds;
    std::optional<double> t_end_s;
};

std::optional<RowSummary> SummariseRow(const std::string& line);

// Where rows go: appended to a CSV and flushed a trial at a time, or to stdout.
class CsvOutput {
public:
    // An error when the file cannot be opened, or holds a header that is not this version's.
    static std::optional<CsvOutput> Open(const std::string& path, std::string& error);
    CsvOutput(const CsvOutput&) = delete;
    CsvOutput& operator=(const CsvOutput&) = delete;
    CsvOutput(CsvOutput&& other) noexcept;
    CsvOutput& operator=(CsvOutput&&) = delete;
    ~CsvOutput();

    void Write(const std::vector<std::string>& rows);

    // The trials a CSV already holds whole, by run key: those with rows and no error among them.
    // A trial interrupted part-way never wrote any, since rows are written a trial at a time.
    const std::map<std::string, std::set<unsigned>>& completed() const noexcept {
        return _completed;
    }

private:
    explicit CsvOutput(std::FILE* file) : _file(file) {}

    std::FILE* _file = nullptr;
    std::map<std::string, std::set<unsigned>> _completed;
};

}  // namespace carta::zarr::bench

#endif  // CARTA_ZARR_BENCH_RECORD_H_
