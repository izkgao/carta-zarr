/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "sky.h"

#include "../../zarr/array_metadata.h"
#include "../../zarr/array_view.h"
#include "linear_axis.h"
#include "probe_report.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <optional>
#include <utility>

namespace carta::zarr::internal::xradio {
namespace {

namespace zarr_metadata = ::carta::zarr::internal::zarr;

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

constexpr std::string_view kVersion = "1.2";
constexpr std::array<std::string_view, 5> kSkyAxes{"time", "frequency", "polarization", "l", "m"};
constexpr double kRadToDeg = 180.0 / M_PI;

bool IsFlag(const zarr_metadata::ArrayMetadata& metadata) {
    return metadata.attributes.contains("type") && metadata.attributes.at("type").is_string() &&
           metadata.attributes.at("type").get<std::string>() == "flag";
}

int KnownImageRank(std::string_view image_id) {
    static constexpr std::array<std::string_view, 6> known{
        "SKY", "MODEL", "RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM", "MASK_DECONVOLVE"};
    const auto* const found = std::find(known.begin(), known.end(), image_id);
    return found == known.end() ? static_cast<int>(known.size()) : static_cast<int>(found - known.begin());
}

bool RequireExpectedAxes(ProbeReport& report, const zarr_metadata::ArrayMetadata& metadata, std::string_view node) {
    if (!report.RequireThat(metadata.dimension_names.size() == kSkyAxes.size(), "invalid_metadata",
                            std::string(node) + " must have exactly five dimensions", std::string(node))) {
        return false;
    }
    for (const auto axis : kSkyAxes) {
        if (!report.RequireThat(zarr_metadata::FindDimensionIndex(metadata, axis).has_value(), "invalid_metadata",
                                std::string(node) + " is missing required axis " + std::string(axis),
                                std::string(node))) {
            return false;
        }
    }
    return true;
}

// XRADIO's converters mark an image dataset with a root "type" attribute, but not with a single
// spelling: create_image_xds_from_store() writes "image_dataset" (and image_xds.py matches on that),
// while the synthetic-image factory writes "image". Accept both, and treat any other value as
// undeclared so that a renamed marker falls back to structural detection instead of rejecting a
// store outright.
bool DeclaresImageDataset(const nlohmann::json& attributes) {
    static constexpr std::array<std::string_view, 2> kImageDatasetTypes{"image_dataset", "image"};
    if (!attributes.is_object() || !attributes.contains("type") || !attributes.at("type").is_string()) {
        return false;
    }
    const auto declared = attributes.at("type").get<std::string>();
    return std::find(kImageDatasetTypes.begin(), kImageDatasetTypes.end(), declared) != kImageDatasetTypes.end();
}

bool HasAttribute(const nlohmann::json& attributes, std::string_view name) {
    return attributes.is_object() && attributes.contains(name);
}

// Every axis a declared image carries must have its coordinate array. Both XRADIO readers write one
// unconditionally for all five axes, time included (xds_from_casacore.py and xds_from_fits.py both
// assign coords["time"]), so a missing coordinate means the store is malformed rather than old. The
// structural path below is the lenient one, for stores predating the root type marker.
void RequirePresentCoordinates(ProbeReport& report, const zarr_metadata::ArrayMetadata& image) {
    for (const auto axis : kSkyAxes) {
        report.RequireCoordinateOf(image, axis,
                                   axis == "polarization" ? CoordinateKind::labels : CoordinateKind::numeric);
    }
}

void AddImageDiagnostic(ImageDescriptor& descriptor, std::string code, std::string message, std::string node_path) {
    descriptor.diagnostics.push_back(Diagnostic{std::move(code), std::move(message), std::move(node_path)});
}

void AppendDiagnostics(ImageDescriptor& descriptor, std::vector<Diagnostic> diagnostics) {
    descriptor.diagnostics.insert(descriptor.diagnostics.end(), std::make_move_iterator(diagnostics.begin()),
                                  std::make_move_iterator(diagnostics.end()));
}

std::string AttributeString(const nlohmann::json& attributes, std::string_view name) {
    if (attributes.is_object() && attributes.contains(name) && attributes.at(name).is_string()) {
        return attributes.at(name).get<std::string>();
    }
    return {};
}

std::optional<double> AttributeNumber(const nlohmann::json& value) {
    return value.is_number() ? std::optional<double>(value.get<double>()) : std::nullopt;
}

constexpr std::array<std::string_view, 5> kApertureAxes{"time", "frequency", "polarization", "u", "v"};

// An image carries every axis of its plane. XRADIO writes optional coordinate arrays that share the
// image's spatial axes without being images at all: right_ascension and declination are float64 over
// (l, m) and carry no type attribute, so a rule keyed only on "has l and m" mistakes them for
// openable images. Matching the whole axis set also drops the (time, frequency, polarization)
// normalization variables and the beam fit parameters, and it means the optional coordinates are
// never read.
bool HasAllAxes(const zarr_metadata::ArrayMetadata& metadata, const std::array<std::string_view, 5>& axes) {
    return std::all_of(axes.begin(), axes.end(), [&metadata](const auto axis) {
        return zarr_metadata::FindDimensionIndex(metadata, axis).has_value();
    });
}

}  // namespace

Result<::carta::zarr::internal::ImageDiscovery> DiscoverSkyImages(const Store& store) {
    auto nodes = store.ListNodeMetadata();
    if (!nodes) {
        return nodes.error();
    }

    ::carta::zarr::internal::ImageDiscovery result;
    for (const auto& entry : nodes.value()) {
        const auto& node = entry.first;
        auto array_result = store.ReadArrayMetadata(node);
        if (!array_result) {
            continue;
        }
        const auto& array = array_result.value();
        if (IsFlag(array)) {
            continue;
        }

        if (HasAllAxes(array, kSkyAxes)) {
            result.image_ids.push_back(node);
            if (zarr_metadata::IsRealDataType(array.data_type)) {
                result.openable_image_ids.push_back(node);
            } else {
                result.diagnostics.push_back(
                    Diagnostic{"unsupported_data_type", "Complex sky-plane variables are not openable", node});
            }
        } else if (HasAllAxes(array, kApertureAxes)) {
            result.image_ids.push_back(node);
            result.diagnostics.push_back(
                Diagnostic{"unsupported_coordinate_plane", "Aperture-plane variables are not openable", node});
        }
    }

    auto sort_images = [](std::vector<std::string>& images) {
        std::sort(images.begin(), images.end(), [](const auto& left, const auto& right) {
            const int left_rank = KnownImageRank(left);
            const int right_rank = KnownImageRank(right);
            return left_rank == right_rank ? left < right : left_rank < right_rank;
        });
    };
    sort_images(result.image_ids);
    sort_images(result.openable_image_ids);
    return result;
}

Result<SchemaProbeResult> ProbeSky(const Store& store) {
    ProbeReport report(store, "SKY");

    const auto& root_attributes = store.RootAttributes();
    if (DeclaresImageDataset(root_attributes)) {
        auto discovery = DiscoverSkyImages(store);
        if (!discovery) {
            return discovery.error();
        }
        report.SetDiagnostics(discovery.value().diagnostics);
        if (discovery.value().openable_image_ids.empty()) {
            // A valid XRADIO file we do not serve stays a non-match; the discovery diagnostics
            // already say why.
            return report.Finish(SchemaMatchKind::no_match, std::string(kVersion));
        }

        // Past this point the store has declared itself an image dataset and offered an image, so
        // any further failure is malformed metadata rather than a different schema.
        const auto& first_image = discovery.value().openable_image_ids.front();
        auto array_result = store.ReadArrayMetadata(first_image);
        if (report.RequireArrayMetadata(array_result, first_image)) {
            RequirePresentCoordinates(report, array_result.value());
            if (report.ok() && HasAttribute(root_attributes, "coordinate_system_info")) {
                report.RequireCoordinateSystem(root_attributes);
            }
        }
        return report.Finish(report.ok() ? SchemaMatchKind::match : SchemaMatchKind::invalid, std::string(kVersion));
    }

    // A store with no SKY node is simply not ours, so it stays a non-match. Asking the Store rather
    // than the filesystem matters: ReadNodeMetadata consults consolidated metadata first, so a store
    // that declares SKY only there is found instead of being silently rejected.
    auto sky_metadata_result = store.ReadNodeMetadata("SKY");
    if (!sky_metadata_result) {
        if (sky_metadata_result.error().code == ErrorCode::not_found) {
            return report.Finish(SchemaMatchKind::no_match, std::string(kVersion));
        }
        return sky_metadata_result.error();
    }

    auto sky_result = store.ReadArrayMetadata("SKY");
    if (report.RequireArrayMetadata(sky_result, "SKY") && RequireExpectedAxes(report, sky_result.value(), "SKY")) {
        const auto& sky = sky_result.value();
        report.RequireThat(zarr_metadata::IsRealDataType(sky.data_type), "unsupported_data_type",
                           "SKY must use a real numeric data type", "SKY");
        report.RequireCoordinateSystem(store.RootAttributes());
        // Legacy XRADIO stores identify time by SKY's dimension metadata and do not necessarily
        // contain a separate time coordinate array. The other four are separate arrays.
        report.RequireCoordinateOf(sky, "frequency", CoordinateKind::numeric);
        report.RequireCoordinateOf(sky, "polarization", CoordinateKind::labels);
        report.RequireCoordinateOf(sky, "l", CoordinateKind::numeric);
        report.RequireCoordinateOf(sky, "m", CoordinateKind::numeric);
    }

    if (!report.ok()) {
        return report.Finish(SchemaMatchKind::invalid, std::string(kVersion));
    }
    // A structural match carries no diagnostics: nothing was wrong, it was simply an older store.
    report.ClearDiagnostics();
    return report.Finish(SchemaMatchKind::match, std::string(kVersion));
}

Result<ImageDescriptor> DescribeSky(const Store& store, std::string_view image_id) {
    auto array_result = store.ReadArrayMetadata(image_id);
    if (!array_result) {
        return array_result.error();
    }
    const auto& image = array_result.value();
    if (IsFlag(image) || !zarr_metadata::FindDimensionIndex(image, "l") ||
        !zarr_metadata::FindDimensionIndex(image, "m") || !zarr_metadata::IsRealDataType(image.data_type)) {
        return MakeError(ErrorCode::unsupported_data_type, "Image variable is not an openable sky-plane image",
                         std::string(image_id));
    }

    ImageDescriptor descriptor;
    descriptor.id = std::string(image_id);
    descriptor.stored_type = zarr_metadata::ParseDataType(image.data_type);
    descriptor.unit = AttributeString(image.attributes, "units");
    // XRADIO writes the role on the variable's own "type" attribute, lowercased ("sky", "model",
    // "residual"); the same attribute spells "flag" for pixel masks, which are never images. The v2
    // schema proposes a separate "image_type" attribute that no released XRADIO writes yet, so it is
    // only consulted as a forward-compatible fallback.
    const std::string declared_role = AttributeString(image.attributes, "type");
    descriptor.image_role = declared_role == "flag" ? std::string{} : declared_role;
    if (descriptor.image_role.empty()) {
        descriptor.image_role = AttributeString(image.attributes, "image_type");
    }

    const auto& root_attrs = store.RootAttributes();
    if (root_attrs.is_object() && root_attrs.contains("data_groups") && root_attrs.at("data_groups").is_object()) {
        for (const auto& [group_name, group] : root_attrs.at("data_groups").items()) {
            if (!group.is_object()) {
                continue;
            }
            const auto references_image = std::any_of(group.begin(), group.end(), [&](const nlohmann::json& value) {
                return value.is_string() && value.get<std::string>() == image_id;
            });
            if (references_image) {
                descriptor.data_groups.push_back(group_name);
            }
        }
    }

    const std::array<std::pair<std::string_view, AxisRole>, 5> logical_axes{{{"l", AxisRole::spatial_x},
                                                                             {"m", AxisRole::spatial_y},
                                                                             {"frequency", AxisRole::spectral},
                                                                             {"polarization", AxisRole::polarization},
                                                                             {"time", AxisRole::time}}};
    for (const auto [name, role] : logical_axes) {
        const auto index = zarr_metadata::FindDimensionIndex(image, name);
        if (!index) {
            continue;
        }
        std::string unit;
        auto coordinate_metadata = store.ReadArrayMetadata(name);
        if (coordinate_metadata) {
            unit = AttributeString(coordinate_metadata.value().attributes, "units");
        }
        descriptor.axes.push_back(
            AxisDescriptor{std::string(name), role, image.shape[*index], std::move(unit), *index});
    }

    std::vector<double> l_values;
    std::vector<double> m_values;
    std::vector<double> frequency_values;
    std::vector<double> time_values;
    auto read_double_coordinate = [&](std::string_view name, std::vector<double>& values) -> Result<void> {
        auto coordinate_metadata = store.ReadNodeMetadata(name);
        if (!coordinate_metadata) {
            if (coordinate_metadata.error().code == ErrorCode::not_found) {
                return {};
            }
            return coordinate_metadata.error();
        }
        auto coordinate_values = store.ReadNumericArray(name);
        if (!coordinate_values) {
            return coordinate_values.error();
        }
        values = std::move(coordinate_values.value());
        return {};
    };
    auto l_result = read_double_coordinate("l", l_values);
    if (!l_result) {
        return l_result.error();
    }
    auto m_result = read_double_coordinate("m", m_values);
    if (!m_result) {
        return m_result.error();
    }

    // A direction axis is linear by construction, so its increment is reported even when the samples
    // are not evenly spaced; the fit says so in a diagnostic rather than withholding the value.
    const bool has_coordinate_system = root_attrs.is_object() && root_attrs.contains("coordinate_system_info") &&
                                       root_attrs.at("coordinate_system_info").is_object();
    if (has_coordinate_system || (!l_values.empty() && !m_values.empty())) {
        DirectionCoordinate dir;
        if (has_coordinate_system) {
            const auto& cs_info = root_attrs.at("coordinate_system_info");
            if (cs_info.contains("projection") && cs_info.at("projection").is_string()) {
                dir.projection = cs_info.at("projection").get<std::string>();
            }
            if (cs_info.contains("reference_direction") && cs_info.at("reference_direction").is_object()) {
                const auto& ref_dir = cs_info.at("reference_direction");
                if (ref_dir.contains("data") && ref_dir.at("data").is_array() && ref_dir.at("data").size() >= 2) {
                    dir.reference_value[0] = ref_dir.at("data")[0].get<double>() * kRadToDeg;
                    dir.reference_value[1] = ref_dir.at("data")[1].get<double>() * kRadToDeg;
                }
                if (ref_dir.contains("attrs") && ref_dir.at("attrs").is_object()) {
                    const auto& attrs = ref_dir.at("attrs");
                    if (attrs.contains("frame") && attrs.at("frame").is_string()) {
                        dir.reference_frame = attrs.at("frame").get<std::string>();
                    }
                    if (attrs.contains("equinox")) {
                        if (attrs.at("equinox").is_number()) {
                            dir.equinox = attrs.at("equinox").get<double>();
                        } else if (attrs.at("equinox").is_string()) {
                            std::string eq = attrs.at("equinox").get<std::string>();
                            size_t const pos =
                                (eq.size() > 1 && (eq[0] == 'J' || eq[0] == 'B' || eq[0] == 'j' || eq[0] == 'b')) ? 1
                                                                                                                  : 0;
                            try {
                                dir.equinox = std::stod(eq.substr(pos));
                            } catch (...) {
                            }
                        }
                    }
                }
            }
            if (cs_info.contains("projection_parameters") && cs_info.at("projection_parameters").is_array()) {
                for (const auto& value : cs_info.at("projection_parameters")) {
                    if (value.is_number()) {
                        dir.projection_parameters.push_back(value.get<double>());
                    }
                }
            }
            if (cs_info.contains("native_pole_direction") && cs_info.at("native_pole_direction").is_object() &&
                cs_info.at("native_pole_direction").contains("data") &&
                zarr_metadata::IsNumericVector(cs_info.at("native_pole_direction").at("data"), 2)) {
                const auto& pole = cs_info.at("native_pole_direction").at("data");
                dir.native_pole_direction[0] = pole[0].get<double>() * kRadToDeg;
                dir.native_pole_direction[1] = pole[1].get<double>() * kRadToDeg;
            }
            if (cs_info.contains("pixel_coordinate_transformation_matrix") &&
                cs_info.at("pixel_coordinate_transformation_matrix").is_array() &&
                cs_info.at("pixel_coordinate_transformation_matrix").size() >= 2) {
                const auto& pc = cs_info.at("pixel_coordinate_transformation_matrix");
                if (pc[0].is_array() && pc[0].size() >= 2 && pc[1].is_array() && pc[1].size() >= 2) {
                    dir.transformation_matrix[0][0] = pc[0][0].get<double>();
                    dir.transformation_matrix[0][1] = pc[0][1].get<double>();
                    dir.transformation_matrix[1][0] = pc[1][0].get<double>();
                    dir.transformation_matrix[1][1] = pc[1][1].get<double>();
                }
            }
        }
        const auto set_direction_axis = [&](const std::vector<double>& values, double& increment,
                                            double& reference_pixel, std::string_view name) {
            auto fit = FitLinearAxis(values, std::nullopt, name);
            if (fit.increment) {
                // The samples are direction cosines; the descriptor reports degrees.
                increment = *fit.increment * kRadToDeg;
            }
            if (fit.reference_pixel) {
                reference_pixel = *fit.reference_pixel;
            }
            AppendDiagnostics(descriptor, std::move(fit.diagnostics));
        };
        set_direction_axis(l_values, dir.increment[0], dir.reference_pixel[0], "l");
        set_direction_axis(m_values, dir.increment[1], dir.reference_pixel[1], "m");
        descriptor.direction = std::move(dir);
    }

    auto frequency_result = read_double_coordinate("frequency", frequency_values);
    if (!frequency_result) {
        return frequency_result.error();
    }
    if (!frequency_values.empty()) {
        SpectralCoordinate spectral;
        spectral.channel_frequencies = frequency_values;
        auto freq_meta = store.ReadNodeMetadata("frequency");
        nlohmann::json freq_attrs = nlohmann::json::object();
        if (freq_meta && freq_meta.value().contains("attributes") && freq_meta.value().at("attributes").is_object()) {
            freq_attrs = freq_meta.value().at("attributes");
        }
        spectral.unit = AttributeString(freq_attrs, "units");
        if (freq_attrs.contains("reference_frequency") && freq_attrs.at("reference_frequency").is_object() &&
            freq_attrs.at("reference_frequency").contains("attrs") &&
            freq_attrs.at("reference_frequency").at("attrs").is_object()) {
            const auto& ref_attrs = freq_attrs.at("reference_frequency").at("attrs");
            if (spectral.unit.empty()) {
                spectral.unit = AttributeString(ref_attrs, "units");
            }
            spectral.system = AttributeString(ref_attrs, "observer");
        }
        double reference_value = spectral.channel_frequencies.front();
        if (freq_attrs.contains("reference_frequency") && freq_attrs.at("reference_frequency").is_object() &&
            freq_attrs.at("reference_frequency").contains("data")) {
            if (const auto value = AttributeNumber(freq_attrs.at("reference_frequency").at("data"))) {
                reference_value = *value;
            }
        }
        if (freq_attrs.contains("rest_frequency") && freq_attrs.at("rest_frequency").is_object() &&
            freq_attrs.at("rest_frequency").contains("data") &&
            freq_attrs.at("rest_frequency").at("data").is_number()) {
            spectral.rest_frequency = freq_attrs.at("rest_frequency").at("data").get<double>();
        }
        auto fit = FitLinearAxis(spectral.channel_frequencies, reference_value, "frequency");
        if (fit.uniform) {
            spectral.reference_pixel = fit.reference_pixel;
            spectral.reference_value = fit.reference_value;
            spectral.increment = fit.increment;
            AppendDiagnostics(descriptor, std::move(fit.diagnostics));
        } else {
            // Unevenly spaced channels get no linear description, so a diagnostic about its
            // reference pixel would describe a value the consumer never sees. Only the reason why
            // is worth carrying; per ADR-0002 it is what tells the consumer to build a tabular axis.
            for (auto& diagnostic : fit.diagnostics) {
                if (diagnostic.code == "nonuniform_axis") {
                    descriptor.diagnostics.push_back(std::move(diagnostic));
                }
            }
        }
        descriptor.spectral = std::move(spectral);
    }

    auto polarization_metadata = store.ReadNodeMetadata("polarization");
    if (polarization_metadata) {
        auto pol_labels = store.ReadStringArray1D("polarization");
        if (!pol_labels) {
            return pol_labels.error();
        }
        PolarizationCoordinate pol;
        pol.labels = std::move(pol_labels.value());
        descriptor.polarization = std::move(pol);
    }

    auto time_result = read_double_coordinate("time", time_values);
    if (!time_result) {
        return time_result.error();
    }
    if (!time_values.empty()) {
        TemporalCoordinate temporal;
        temporal.values = std::move(time_values);
        auto time_metadata = store.ReadNodeMetadata("time");
        if (time_metadata && time_metadata.value().contains("attributes") &&
            time_metadata.value().at("attributes").is_object()) {
            const auto& time_attrs = time_metadata.value().at("attributes");
            temporal.unit = AttributeString(time_attrs, "units");
            temporal.scale = AttributeString(time_attrs, "scale");
            temporal.format = AttributeString(time_attrs, "format");
        }
        descriptor.temporal = std::move(temporal);
    }

    // Parse Observation & Telescope Metadata from the selected image's attributes.
    ObservationInfo obs;
    if (image.attributes.contains("object_name") && image.attributes.at("object_name").is_string()) {
        obs.object_name = image.attributes.at("object_name").get<std::string>();
    }
    if (image.attributes.contains("observer") && image.attributes.at("observer").is_string()) {
        obs.observer = image.attributes.at("observer").get<std::string>();
    }
    if (image.attributes.contains("telescope") && image.attributes.at("telescope").is_object()) {
        const auto& telescope = image.attributes.at("telescope");
        if (telescope.contains("name") && telescope.at("name").is_string()) {
            obs.telescope_name = telescope.at("name").get<std::string>();
        }
        if (telescope.contains("direction") && telescope.at("direction").is_object() &&
            telescope.at("direction").contains("data") && telescope.at("direction").at("data").is_array() &&
            telescope.at("direction").at("data").size() >= 2 && telescope.contains("distance") &&
            telescope.at("distance").is_object() && telescope.at("distance").contains("data") &&
            telescope.at("distance").at("data").is_array() && !telescope.at("distance").at("data").empty()) {
            double const lon = telescope.at("direction").at("data")[0].get<double>();
            double const lat = telescope.at("direction").at("data")[1].get<double>();
            double const radius = telescope.at("distance").at("data")[0].get<double>();
            obs.observatory_position = std::array<double, 3>{
                radius * std::cos(lat) * std::cos(lon), radius * std::cos(lat) * std::sin(lon), radius * std::sin(lat)};
        }
    }
    if (image.attributes.contains("obsdate") && image.attributes.at("obsdate").is_object()) {
        const auto& obsdate = image.attributes.at("obsdate");
        if (obsdate.contains("attrs") && obsdate.at("attrs").is_object() && obsdate.at("attrs").contains("scale") &&
            obsdate.at("attrs").at("scale").is_string()) {
            obs.timesys = obsdate.at("attrs").at("scale").get<std::string>();
        }
        if (obsdate.contains("data")) {
            if (obsdate.at("data").is_string()) {
                obs.date_obs = obsdate.at("data").get<std::string>();
            } else if (obsdate.at("data").is_number()) {
                obs.mjd_obs = obsdate.at("data").get<double>();
            }
        }
    }
    descriptor.observation = std::move(obs);

    // Parse Storage Layout
    auto layout_res = store.ReadStorageLayout(image_id);
    if (layout_res) {
        descriptor.storage = std::move(layout_res.value());
    }

    const std::string flag_name = AttributeString(image.attributes, "flag");
    if (!flag_name.empty()) {
        descriptor.has_pixel_mask = true;
    } else {
        auto nodes = store.ListNodeMetadata();
        if (!nodes) {
            return nodes.error();
        }
        std::vector<std::string> matching_flags;
        for (const auto& entry : nodes.value()) {
            const auto& node = entry.first;
            auto flag_array = store.ReadArrayMetadata(node);
            if (!flag_array || !IsFlag(flag_array.value()) || flag_array.value().shape != image.shape) {
                continue;
            }
            matching_flags.push_back(node);
        }
        if (matching_flags.size() == 1) {
            descriptor.has_pixel_mask = true;
        } else if (matching_flags.size() > 1) {
            AddImageDiagnostic(descriptor, "ambiguous_pixel_mask",
                               "More than one flag variable matches the image shape; no pixel mask was selected",
                               std::string(image_id));
        }
    }

    return descriptor;
}

Result<std::vector<Beam>> ReadBeamsSky(const Store& store, std::string_view image_id) {
    auto sky_meta = store.ReadNodeMetadata(image_id);
    if (!sky_meta) {
        return sky_meta.error();
    }
    std::string beam_array_name;
    if (sky_meta.value().contains("attributes") && sky_meta.value().at("attributes").is_object() &&
        sky_meta.value().at("attributes").contains("beam_fit_params") &&
        sky_meta.value().at("attributes").at("beam_fit_params").is_string()) {
        beam_array_name = sky_meta.value().at("attributes").at("beam_fit_params").get<std::string>();
    }
    if (beam_array_name.empty()) {
        // No beam array associated
        return std::vector<Beam>{};
    }

    auto beam_arr_res = store.ReadArrayMetadata(beam_array_name);
    if (!beam_arr_res) {
        return beam_arr_res.error();
    }
    const auto& beam_arr_info = beam_arr_res.value();

    std::string beam_unit;
    if (beam_arr_info.attributes.contains("units") && beam_arr_info.attributes.at("units").is_string()) {
        beam_unit = beam_arr_info.attributes.at("units").get<std::string>();
    }

    auto param_labels_res = store.ReadStringArray1D("beam_params_label");
    std::vector<std::string> param_labels;
    if (param_labels_res) {
        param_labels = param_labels_res.value();
    }

    std::optional<size_t> major_idx;
    std::optional<size_t> minor_idx;
    std::optional<size_t> pa_idx;
    for (size_t i = 0; i < param_labels.size(); ++i) {
        std::string label = param_labels[i];
        std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c) { return std::toupper(c); });
        if (label == "MAJOR") {
            major_idx = i;
        } else if (label == "MINOR") {
            minor_idx = i;
        } else if (label == "PA") {
            pa_idx = i;
        }
    }

    auto beam_data_res = store.ReadNumericArray(beam_array_name);
    if (!beam_data_res) {
        return beam_data_res.error();
    }
    const auto& beam_data = beam_data_res.value();

    const auto freq_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "frequency");
    const auto pol_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "polarization");
    const auto param_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "beam_params_label");
    const auto time_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "time");

    if (!freq_dim || !pol_dim || !param_dim || !major_idx || !minor_idx || !pa_idx) {
        return std::vector<Beam>{};
    }

    const std::uint64_t n_time = time_dim ? beam_arr_info.shape[*time_dim] : 1;
    const std::uint64_t n_chan = beam_arr_info.shape[*freq_dim];
    const std::uint64_t n_pol = beam_arr_info.shape[*pol_dim];

    // Time varies slowest so that a single-plane beam table reads back in the order it always has.
    const zarr_metadata::ArrayView beam_values(beam_arr_info, beam_data);
    std::vector<Beam> beams;
    beams.reserve(static_cast<std::size_t>(n_time * n_chan * n_pol));
    for (std::uint64_t t = 0; t < n_time; ++t) {
        for (std::uint64_t c = 0; c < n_chan; ++c) {
            for (std::uint64_t p = 0; p < n_pol; ++p) {
                Beam beam;
                beam.time = static_cast<std::size_t>(t);
                beam.channel = static_cast<std::size_t>(c);
                beam.polarization = static_cast<std::size_t>(p);
                beam.unit = beam_unit;

                for (const auto& [parameter, field] :
                     {std::pair{*major_idx, &beam.major}, std::pair{*minor_idx, &beam.minor},
                      std::pair{*pa_idx, &beam.position_angle}}) {
                    // A beam table need not carry a time dimension; naming one it lacks is an error,
                    // not a way of asking for its single implicit plane.
                    std::vector<zarr_metadata::ArrayView::NamedIndex> indices{
                        {"frequency", c}, {"polarization", p}, {"beam_params_label", parameter}};
                    if (time_dim) {
                        indices.emplace_back("time", t);
                    }
                    auto value = beam_values.At(indices);
                    if (!value) {
                        return value.error();
                    }
                    *field = value.value();
                }
                beams.push_back(std::move(beam));
            }
        }
    }

    return beams;
}

}  // namespace carta::zarr::internal::xradio
