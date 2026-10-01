/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// carta-zarr-bench: how fast this library reads a dataset on this storage, the way CARTA reads it.
//
// It reaches the library only through its public headers, as carta-backend does, so what it
// measures is what a backend linked against the same build would see. tools/zarr-bench/sweep.py runs
// it over layouts and settings and turns its CSV into a recommendation; see tools/zarr-bench/README.md.

#include "cold.h"
#include "options.h"
#include "record.h"
#include "trial.h"
#include "workload.h"

#include <carta-zarr/carta_zarr.h>

#include <nlohmann/json.hpp>

#include <csignal>
#include <cstdio>
#include <random>
#include <string>

namespace {

using namespace carta::zarr;
using namespace carta::zarr::bench;

const char* RoleName(AxisRole role) {
    switch (role) {
        case AxisRole::spatial_x:
            return "spatial_x";
        case AxisRole::spatial_y:
            return "spatial_y";
        case AxisRole::spectral:
            return "spectral";
        case AxisRole::polarization:
            return "polarization";
        case AxisRole::time:
            return "time";
        case AxisRole::other:
            return "other";
    }
    return "other";
}

const char* MatchName(SchemaMatchKind kind) {
    switch (kind) {
        case SchemaMatchKind::no_match:
            return "no_match";
        case SchemaMatchKind::match:
            return "match";
        case SchemaMatchKind::invalid:
            return "invalid";
    }
    return "no_match";
}

nlohmann::json Diagnostics(const std::vector<Diagnostic>& diagnostics) {
    auto list = nlohmann::json::array();
    for (const auto& diagnostic : diagnostics) {
        list.push_back({{"code", DiagnosticCodeName(diagnostic.code)},
                        {"message", diagnostic.message},
                        {"node", diagnostic.node_path}});
    }
    return list;
}

// One line of JSON, written whether or not the dataset opens, so that a caller always has something
// to parse; the exit status says whether it did.
int Probe(const ProbeOptions& options) {
    nlohmann::json report{{"dataset", options.dataset}, {"ok", false}};
    const auto finish = [&](const std::string& error) {
        if (!error.empty()) {
            report["error"] = error;
        }
        std::printf("%s\n", report.dump().c_str());
        return report["ok"].get<bool>() ? 0 : 1;
    };

    const auto probe = ProbeSchema(options.dataset, kXradioImageSchema);
    if (!probe) {
        return finish("ProbeSchema: " + std::string(ErrorCodeName(probe.error().code)) + ": " +
                      probe.error().message);
    }
    report["schema"] = {{"match", MatchName(probe->kind)},
                        {"id", probe->schema_id},
                        {"version", probe->schema_version},
                        {"diagnostics", Diagnostics(probe->diagnostics)}};
    if (probe->kind != SchemaMatchKind::match) {
        return finish("not an XRADIO image dataset");
    }

    const auto context = Context::Create();
    if (!context) {
        return finish("Context::Create: " + context.error().message);
    }
    const auto dataset = Dataset::Open(*context, options.dataset);
    if (!dataset) {
        return finish("Dataset::Open: " + std::string(ErrorCodeName(dataset.error().code)) + ": " +
                      dataset.error().message);
    }
    auto images = nlohmann::json::array();
    for (const auto& entry : dataset->descriptor().images) {
        images.push_back({{"id", entry.id}, {"openable", entry.openable}, {"role", entry.image_role}});
    }
    report["images"] = images;
    const auto& default_id = dataset->descriptor().default_image_id;
    report["default_image_id"] = default_id ? nlohmann::json(*default_id) : nlohmann::json();

    const auto id = options.image_id.empty() ? default_id.value_or("") : options.image_id;
    if (id.empty()) {
        return finish("the dataset lists no image that opens");
    }
    const auto image = dataset->OpenImage(id);
    if (!image) {
        return finish("OpenImage: " + std::string(ErrorCodeName(image.error().code)) + ": " + image.error().message);
    }
    const auto& descriptor = image->descriptor();
    const auto& geometry = image->chunk_geometry();
    auto axes = nlohmann::json::array();
    for (std::size_t index = 0; index < descriptor.axes.size(); ++index) {
        const auto& axis = descriptor.axes[index];
        axes.push_back({{"name", axis.name},
                        {"role", RoleName(axis.role)},
                        {"length", axis.length},
                        {"chunk", geometry.chunk_shape.at(index)},
                        {"shard", geometry.shard_shape.at(index)}});
    }
    report["image"] = {{"id", descriptor.id},
                       {"axes", axes},
                       {"sharded", geometry.sharded},
                       {"compressor", geometry.compressor},
                       {"has_pixel_mask", descriptor.has_pixel_mask},
                       {"diagnostics", Diagnostics(descriptor.diagnostics)}};
    if (const auto cube = CubeAxes::Of(descriptor); !cube) {
        return finish(cube.error().message);
    }
    report["ok"] = true;
    return finish("");
}

std::string RunId() {
    std::random_device device;
    const auto high = static_cast<std::uint64_t>(device()) << 32;
    std::array<char, 17> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(high | device()));
    return buffer.data();
}

int Run(const RunOptions& options) {
    const auto cold = ChooseColdMethod(options.cold, options.drop_cache_command);
    if (!cold.error.empty()) {
        std::fprintf(stderr, "error: %s\n", cold.error.c_str());
        return 2;
    }
    if (std::string_view(CARTA_ZARR_BENCH_BUILD_TYPE) != "Release") {
        std::fprintf(stderr,
                     "warning: carta-zarr-bench was built as %s; only a Release build measures what a "
                     "deployed backend would see\n",
                     CARTA_ZARR_BENCH_BUILD_TYPE);
    }
    if (std::string_view(CARTA_ZARR_BENCH_TUNING) != "default") {
        std::fprintf(stderr, "note: the library is built with tuning overrides: %s\n", CARTA_ZARR_BENCH_TUNING);
    }
    std::fprintf(stderr, "caches emptied by: %s\n", ColdMethodName(cold.method));

    std::string error;
    auto output = CsvOutput::Open(options.csv_path, error);
    if (!output) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }

    const auto facts = ReadManifest(options.dataset);
    const auto run_id = RunId();
    bool failed = false;
    for (const auto mode : options.modes) {
        const auto base = RowTemplate(options, mode, cold.method, facts, run_id);
        const auto done = output->completed().find(base.run_key);
        for (unsigned trial = 0; trial < options.trials; ++trial) {
            if (options.resume && done != output->completed().end() && done->second.count(trial) > 0) {
                std::fprintf(stderr, "%s trial %u/%u: already in %s\n", ModeName(mode), trial + 1, options.trials,
                             options.csv_path.c_str());
                continue;
            }
            const auto outcome = RunTrial(options, mode, trial, cold.method, base);
            output->Write(outcome.rows);
            std::fprintf(stderr, "%s trial %u/%u: %u ok, %u timeout, %u error, makespan %.3f s\n", ModeName(mode),
                         trial + 1, options.trials, outcome.ok, outcome.timeouts, outcome.errors, outcome.makespan_s);
            failed = failed || outcome.errors > 0;
        }
    }
    return failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    // A process that dies before its release must not take the parent with it when the parent writes
    // to its pipe.
    std::signal(SIGPIPE, SIG_IGN);

    const auto command = ParseCommandLine(argc, argv);
    if (const auto* usage = std::get_if<Usage>(&command)) {
        std::fprintf(usage->error ? stderr : stdout, "%s", usage->message.c_str());
        return usage->error ? 2 : 0;
    }
    if (const auto* probe = std::get_if<ProbeOptions>(&command)) {
        return Probe(*probe);
    }
    return Run(std::get<RunOptions>(command));
}
