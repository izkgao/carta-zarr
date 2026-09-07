/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "image.h"

#include "../../zarr/array_metadata.h"
#include "../../zarr/array_view.h"
#include "linear_axis.h"
#include "probe_report.h"

#include <algorithm>
#include <array>
#include <cctype>
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

std::string Upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
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

bool HasAttribute(const nlohmann::json& attributes, std::string_view name) {
    return attributes.is_object() && attributes.contains(name);
}

// Every image carries a coordinate array for each axis it uses. Both XRADIO readers write all five
// unconditionally (xds_from_casacore.py and xds_from_fits.py both assign coords["time"]), so a
// missing coordinate means the store is malformed.
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

const nlohmann::json* ObjectMember(const nlohmann::json& object, std::string_view name) {
    if (!object.is_object() || !object.contains(name)) {
        return nullptr;
    }
    return &object.at(name);
}

std::vector<std::string> FindDataGroups(const nlohmann::json& root_attributes, std::string_view image_id) {
    std::vector<std::string> data_groups;
    const auto* groups = ObjectMember(root_attributes, "data_groups");
    if (groups == nullptr || !groups->is_object()) {
        return data_groups;
    }

    for (const auto& [group_name, group] : groups->items()) {
        if (!group.is_object()) {
            continue;
        }
        const auto references_image = std::any_of(group.begin(), group.end(), [&](const nlohmann::json& value) {
            return value.is_string() && value.get<std::string>() == image_id;
        });
        if (references_image) {
            data_groups.push_back(group_name);
        }
    }
    return data_groups;
}

std::vector<AxisDescriptor> DescribeAxes(const Store& store, const zarr_metadata::ArrayMetadata& image) {
    const std::array<std::pair<std::string_view, AxisRole>, 5> logical_axes{{{"l", AxisRole::spatial_x},
                                                                             {"m", AxisRole::spatial_y},
                                                                             {"frequency", AxisRole::spectral},
                                                                             {"polarization", AxisRole::polarization},
                                                                             {"time", AxisRole::time}}};
    std::vector<AxisDescriptor> axes;
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
        axes.push_back(AxisDescriptor{std::string(name), role, image.shape[*index], std::move(unit), *index});
    }
    return axes;
}

Result<std::vector<double>> ReadNumericCoordinate(const Store& store, std::string_view name) {
    auto metadata = store.ReadNodeMetadata(name);
    if (!metadata) {
        if (metadata.error().code == ErrorCode::not_found) {
            return std::vector<double>{};
        }
        return metadata.error();
    }
    return store.ReadNumericArray(name);
}

std::optional<DirectionCoordinate> DescribeDirection(const nlohmann::json& root_attributes,
                                                     const std::vector<double>& l_values,
                                                     const std::vector<double>& m_values, ImageDescriptor& descriptor) {
    const auto* coordinate_system = ObjectMember(root_attributes, "coordinate_system_info");
    const bool has_coordinate_system = coordinate_system != nullptr && coordinate_system->is_object();
    if (!has_coordinate_system && (l_values.empty() || m_values.empty())) {
        return std::nullopt;
    }

    DirectionCoordinate direction;
    if (has_coordinate_system) {
        const auto& cs_info = *coordinate_system;
        if (const auto* projection = ObjectMember(cs_info, "projection");
            projection != nullptr && projection->is_string()) {
            direction.projection = Upper(projection->get<std::string>());
        }
        if (const auto* reference_direction = ObjectMember(cs_info, "reference_direction");
            reference_direction != nullptr && reference_direction->is_object()) {
            if (const auto* data = ObjectMember(*reference_direction, "data");
                data != nullptr && data->is_array() && data->size() >= 2) {
                direction.reference_value[0] = data->at(0).get<double>() * kRadToDeg;
                direction.reference_value[1] = data->at(1).get<double>() * kRadToDeg;
            }
            if (const auto* attributes = ObjectMember(*reference_direction, "attrs");
                attributes != nullptr && attributes->is_object()) {
                direction.reference_frame = Upper(AttributeString(*attributes, "frame"));
                if (const auto* equinox = ObjectMember(*attributes, "equinox"); equinox != nullptr) {
                    if (equinox->is_number()) {
                        direction.equinox = equinox->get<double>();
                    } else if (equinox->is_string()) {
                        const std::string value = equinox->get<std::string>();
                        const std::size_t position = (value.size() > 1 && (value[0] == 'J' || value[0] == 'B' ||
                                                                           value[0] == 'j' || value[0] == 'b'))
                                                       ? 1
                                                       : 0;
                        try {
                            direction.equinox = std::stod(value.substr(position));
                        } catch (...) {
                        }
                    }
                }
            }
        }
        if (const auto* parameters = ObjectMember(cs_info, "projection_parameters");
            parameters != nullptr && parameters->is_array()) {
            for (const auto& value : *parameters) {
                if (value.is_number()) {
                    direction.projection_parameters.push_back(value.get<double>());
                }
            }
        }
        if (const auto* native_pole = ObjectMember(cs_info, "native_pole_direction");
            native_pole != nullptr && native_pole->is_object()) {
            const auto* data = ObjectMember(*native_pole, "data");
            if (data != nullptr && zarr_metadata::IsNumericVector(*data, 2)) {
                direction.native_pole_direction[0] = data->at(0).get<double>() * kRadToDeg;
                direction.native_pole_direction[1] = data->at(1).get<double>() * kRadToDeg;
            }
        }
        if (const auto* matrix = ObjectMember(cs_info, "pixel_coordinate_transformation_matrix");
            matrix != nullptr && matrix->is_array() && matrix->size() >= 2) {
            if (matrix->at(0).is_array() && matrix->at(0).size() >= 2 && matrix->at(1).is_array() &&
                matrix->at(1).size() >= 2) {
                direction.transformation_matrix[0][0] = matrix->at(0).at(0).get<double>();
                direction.transformation_matrix[0][1] = matrix->at(0).at(1).get<double>();
                direction.transformation_matrix[1][0] = matrix->at(1).at(0).get<double>();
                direction.transformation_matrix[1][1] = matrix->at(1).at(1).get<double>();
            }
        }
    }

    const auto set_direction_axis = [&](const std::vector<double>& values, double& increment, double& reference_pixel,
                                        std::string_view name) {
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
    set_direction_axis(l_values, direction.increment[0], direction.reference_pixel[0], "l");
    set_direction_axis(m_values, direction.increment[1], direction.reference_pixel[1], "m");
    return direction;
}

std::optional<SpectralCoordinate> DescribeSpectralCoordinate(const Store& store,
                                                             const std::vector<double>& frequency_values,
                                                             ImageDescriptor& descriptor) {
    if (frequency_values.empty()) {
        return std::nullopt;
    }

    SpectralCoordinate spectral;
    spectral.channel_frequencies = frequency_values;
    nlohmann::json frequency_attributes = nlohmann::json::object();
    if (auto frequency_metadata = store.ReadNodeMetadata("frequency"); frequency_metadata) {
        if (const auto* attributes = ObjectMember(frequency_metadata.value(), "attributes");
            attributes != nullptr && attributes->is_object()) {
            frequency_attributes = *attributes;
        }
    }

    spectral.unit = AttributeString(frequency_attributes, "units");
    if (const auto* reference_frequency = ObjectMember(frequency_attributes, "reference_frequency");
        reference_frequency != nullptr && reference_frequency->is_object()) {
        if (const auto* attributes = ObjectMember(*reference_frequency, "attrs");
            attributes != nullptr && attributes->is_object()) {
            if (spectral.unit.empty()) {
                spectral.unit = AttributeString(*attributes, "units");
            }
            spectral.system = Upper(AttributeString(*attributes, "observer"));
        }
    }

    double reference_value = spectral.channel_frequencies.front();
    if (const auto* reference_frequency = ObjectMember(frequency_attributes, "reference_frequency");
        reference_frequency != nullptr && reference_frequency->is_object()) {
        if (const auto* data = ObjectMember(*reference_frequency, "data"); data != nullptr) {
            if (const auto value = AttributeNumber(*data)) {
                reference_value = *value;
            }
        }
    }
    if (const auto* rest_frequency = ObjectMember(frequency_attributes, "rest_frequency");
        rest_frequency != nullptr && rest_frequency->is_object()) {
        if (const auto* data = ObjectMember(*rest_frequency, "data"); data != nullptr && data->is_number()) {
            spectral.rest_frequency = data->get<double>();
        }
    }

    auto fit = FitLinearAxis(spectral.channel_frequencies, reference_value, "frequency");
    if (fit.uniform) {
        spectral.reference_pixel = fit.reference_pixel;
        spectral.reference_value = fit.reference_value;
        spectral.increment = fit.increment;
        AppendDiagnostics(descriptor, std::move(fit.diagnostics));
    } else {
        // Unevenly spaced channels get no linear description, so a diagnostic about its reference
        // pixel would describe a value the consumer never sees. Only the reason why is worth
        // carrying; it tells the consumer to build a tabular axis.
        for (auto& diagnostic : fit.diagnostics) {
            if (diagnostic.code == "nonuniform_axis") {
                descriptor.diagnostics.push_back(std::move(diagnostic));
            }
        }
    }
    return spectral;
}

std::optional<TemporalCoordinate> DescribeTemporalCoordinate(const Store& store, std::vector<double> values) {
    if (values.empty()) {
        return std::nullopt;
    }

    TemporalCoordinate temporal;
    temporal.values = std::move(values);
    if (auto metadata = store.ReadNodeMetadata("time"); metadata) {
        if (const auto* attributes = ObjectMember(metadata.value(), "attributes");
            attributes != nullptr && attributes->is_object()) {
            temporal.unit = AttributeString(*attributes, "units");
            temporal.scale = Upper(AttributeString(*attributes, "scale"));
            temporal.format = Upper(AttributeString(*attributes, "format"));
        }
    }
    return temporal;
}

ObservationInfo DescribeObservation(const zarr_metadata::ArrayMetadata& image) {
    ObservationInfo observation;
    if (const auto* object_name = ObjectMember(image.attributes, "object_name");
        object_name != nullptr && object_name->is_string()) {
        observation.object_name = object_name->get<std::string>();
    }
    if (const auto* observer = ObjectMember(image.attributes, "observer");
        observer != nullptr && observer->is_string()) {
        observation.observer = observer->get<std::string>();
    }
    if (const auto* telescope = ObjectMember(image.attributes, "telescope");
        telescope != nullptr && telescope->is_object()) {
        if (const auto* name = ObjectMember(*telescope, "name"); name != nullptr && name->is_string()) {
            observation.telescope_name = name->get<std::string>();
        }
        const auto* direction = ObjectMember(*telescope, "direction");
        const auto* distance = ObjectMember(*telescope, "distance");
        const nlohmann::json* direction_data = nullptr;
        if (direction != nullptr) {
            direction_data = ObjectMember(*direction, "data");
        }
        const nlohmann::json* distance_data = nullptr;
        if (distance != nullptr) {
            distance_data = ObjectMember(*distance, "data");
        }
        if (direction != nullptr && direction->is_object() && direction_data != nullptr && direction_data->is_array() &&
            direction_data->size() >= 2 && distance != nullptr && distance->is_object() && distance_data != nullptr &&
            distance_data->is_array() && !distance_data->empty()) {
            const double lon = direction_data->at(0).get<double>();
            const double lat = direction_data->at(1).get<double>();
            const double radius = distance_data->at(0).get<double>();
            observation.observatory_position = std::array<double, 3>{
                radius * std::cos(lat) * std::cos(lon), radius * std::cos(lat) * std::sin(lon), radius * std::sin(lat)};
        }
    }
    if (const auto* obsdate = ObjectMember(image.attributes, "obsdate"); obsdate != nullptr && obsdate->is_object()) {
        if (const auto* attributes = ObjectMember(*obsdate, "attrs");
            attributes != nullptr && attributes->is_object()) {
            observation.timesys = Upper(AttributeString(*attributes, "scale"));
        }
        if (const auto* data = ObjectMember(*obsdate, "data"); data != nullptr) {
            if (data->is_string()) {
                observation.date_obs = data->get<std::string>();
            } else if (data->is_number()) {
                observation.mjd_obs = data->get<double>();
            }
        }
    }
    return observation;
}

// Returns the flag variable supplying this image's pixel mask, or an empty name when it has none.
Result<std::string> DeterminePixelMask(const Store& store, const zarr_metadata::ArrayMetadata& image,
                                       std::string_view image_id, ImageDescriptor& descriptor) {
    if (auto declared = AttributeString(image.attributes, "flag"); !declared.empty()) {
        return declared;
    }

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
        return matching_flags.front();
    }
    if (matching_flags.size() > 1) {
        AddImageDiagnostic(descriptor, "ambiguous_pixel_mask",
                           "More than one flag variable matches the image shape; no pixel mask was selected",
                           std::string(image_id));
    }
    return std::string{};
}

struct BeamParameterIndices {
    std::optional<std::size_t> major;
    std::optional<std::size_t> minor;
    std::optional<std::size_t> position_angle;
};

BeamParameterIndices FindBeamParameterIndices(const std::vector<std::string>& labels) {
    BeamParameterIndices indices;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const std::string label = Upper(labels[index]);
        if (label == "MAJOR") {
            indices.major = index;
        } else if (label == "MINOR") {
            indices.minor = index;
        } else if (label == "PA") {
            indices.position_angle = index;
        }
    }
    return indices;
}

Result<double> ReadBeamValue(const zarr_metadata::ArrayView& values, std::uint64_t channel, std::uint64_t polarization,
                             std::uint64_t parameter, bool has_time_dimension, std::uint64_t time) {
    std::vector<zarr_metadata::ArrayView::NamedIndex> indices{
        {"frequency", channel}, {"polarization", polarization}, {"beam_params_label", parameter}};
    if (has_time_dimension) {
        indices.emplace_back("time", time);
    }
    return values.At(indices);
}

}  // namespace

Result<::carta::zarr::internal::ImageDiscovery> DiscoverImages(const Store& store) {
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

Result<SchemaProbeResult> ProbeImage(const Store& store) {
    ProbeReport report(store, "image dataset");

    const auto& root_attributes = store.RootAttributes();
    auto discovery = DiscoverImages(store);
    if (!discovery) {
        return discovery.error();
    }
    report.SetDiagnostics(discovery.value().diagnostics);
    if (discovery.value().openable_image_ids.empty()) {
        // A valid Zarr store without an image that this profile can open is a non-match. The
        // discovery diagnostics still explain why variables such as complex or aperture-plane
        // arrays were not openable.
        return report.Finish(SchemaMatchKind::no_match, std::string(kVersion));
    }

    // Once discovery found an openable image, validate the metadata needed by the image reader.
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

Result<ImageDescriptor> DescribeImage(const Store& store, std::string_view image_id) {
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
    descriptor.data_groups = FindDataGroups(root_attrs, image_id);
    descriptor.axes = DescribeAxes(store, image);

    std::vector<double> l_values;
    std::vector<double> m_values;
    std::vector<double> frequency_values;
    std::vector<double> time_values;
    auto l_result = ReadNumericCoordinate(store, "l");
    if (!l_result) {
        return l_result.error();
    }
    l_values = std::move(l_result.value());
    auto m_result = ReadNumericCoordinate(store, "m");
    if (!m_result) {
        return m_result.error();
    }
    m_values = std::move(m_result.value());

    // A direction axis is linear by construction, so its increment is reported even when the samples
    // are not evenly spaced; the fit says so in a diagnostic rather than withholding the value.
    if (auto direction = DescribeDirection(root_attrs, l_values, m_values, descriptor); direction) {
        descriptor.direction = std::move(direction);
    }

    auto frequency_result = ReadNumericCoordinate(store, "frequency");
    if (!frequency_result) {
        return frequency_result.error();
    }
    frequency_values = std::move(frequency_result.value());
    if (auto spectral = DescribeSpectralCoordinate(store, frequency_values, descriptor); spectral) {
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

    auto time_result = ReadNumericCoordinate(store, "time");
    if (!time_result) {
        return time_result.error();
    }
    time_values = std::move(time_result.value());
    if (auto temporal = DescribeTemporalCoordinate(store, std::move(time_values)); temporal) {
        descriptor.temporal = std::move(temporal);
    }

    // Parse Observation & Telescope Metadata from the selected image's attributes.
    descriptor.observation = DescribeObservation(image);

    auto layout_res = store.ReadStorageLayout(image_id);
    if (layout_res) {
        descriptor.storage = std::move(layout_res.value());
    }

    auto pixel_mask = DeterminePixelMask(store, image, image_id, descriptor);
    if (!pixel_mask) {
        return pixel_mask.error();
    }
    descriptor.pixel_mask_id = pixel_mask.value();
    descriptor.has_pixel_mask = !descriptor.pixel_mask_id.empty();

    return descriptor;
}

Result<std::vector<Beam>> ReadBeams(const Store& store, std::string_view image_id) {
    auto sky_meta = store.ReadNodeMetadata(image_id);
    if (!sky_meta) {
        return sky_meta.error();
    }
    const auto& sky_metadata = sky_meta.value();
    std::string beam_array_name;
    if (const auto* attributes = ObjectMember(sky_metadata, "attributes"); attributes != nullptr) {
        beam_array_name = AttributeString(*attributes, "beam_fit_params");
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

    const std::string beam_unit = AttributeString(beam_arr_info.attributes, "units");

    auto param_labels_res = store.ReadStringArray1D("beam_params_label");
    std::vector<std::string> param_labels;
    if (param_labels_res) {
        param_labels = param_labels_res.value();
    }

    const auto parameter_indices = FindBeamParameterIndices(param_labels);

    auto beam_data_res = store.ReadNumericArray(beam_array_name);
    if (!beam_data_res) {
        return beam_data_res.error();
    }
    const auto& beam_data = beam_data_res.value();

    const auto freq_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "frequency");
    const auto pol_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "polarization");
    const auto param_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "beam_params_label");
    const auto time_dim = zarr_metadata::FindDimensionIndex(beam_arr_info, "time");

    if (!freq_dim || !pol_dim || !param_dim || !parameter_indices.major || !parameter_indices.minor ||
        !parameter_indices.position_angle) {
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
                     {std::pair{*parameter_indices.major, &beam.major},
                      std::pair{*parameter_indices.minor, &beam.minor},
                      std::pair{*parameter_indices.position_angle, &beam.position_angle}}) {
                    auto value = ReadBeamValue(beam_values, c, p, parameter, time_dim.has_value(), t);
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
