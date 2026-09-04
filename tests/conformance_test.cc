/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Conformance: assert what this library believes about XRADIO against a store that XRADIO itself
// produced. The fixture comes from tests/data/generate_conformance_fixtures.py, which converts a
// FITS image with a pinned XRADIO. When that pin is bumped and one of these fails, the failure names
// the assumption the new XRADIO version broke.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <carta-zarr/carta_zarr.h>

namespace {

using carta::zarr::AxisRole;

const std::filesystem::path kFixture{CARTA_ZARR_CONFORMANCE_FIXTURE};

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireClose(double actual, double expected, double tolerance, const std::string& message) {
    Require(std::abs(actual - expected) <= tolerance,
            message + " (expected " + std::to_string(expected) + ", got " + std::to_string(actual) + ")");
}

carta::zarr::Image OpenSky() {
    Require(std::filesystem::exists(kFixture),
            "the conformance fixture is missing; run tests/data/generate_conformance_fixtures.py");

    const auto matched = carta::zarr::IsXradioImage(kFixture.string());
    Require(matched && matched.value(), "XRADIO's own output was not recognized as an image dataset");

    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    auto dataset = carta::zarr::Dataset::Open(context.value(), kFixture.string());
    Require(static_cast<bool>(dataset), "Dataset::Open failed on XRADIO's own output");

    // XRADIO writes right_ascension, declination and velocity as coordinates alongside SKY. Only
    // SKY carries the full sky axis set, so only SKY is an image.
    Require(dataset.value().descriptor().image_ids == std::vector<std::string>{"SKY"},
            "discovery did not report SKY as the only image of XRADIO's own output");
    auto image = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(image), "SKY could not be opened");
    return image.value();
}

void TestIdentityAndAxes(const carta::zarr::ImageDescriptor& sky) {
    Require(sky.image_role == "sky", "SKY's role is written on its own type attribute");
    Require(sky.unit == "Jy/beam", "BUNIT did not survive the conversion as the image unit");
    Require(sky.data_groups == std::vector<std::string>{"base"}, "SKY's data group membership changed");
    Require(!sky.has_pixel_mask, "an image converted from a FITS with no mask reported a pixel mask");

    // The FITS was (RA 5, DEC 4, STOKES 3, FREQ 2); the library reports logical order.
    const std::vector<std::pair<AxisRole, std::uint64_t>> expected{{AxisRole::spatial_x, 5},
                                                                   {AxisRole::spatial_y, 4},
                                                                   {AxisRole::spectral, 2},
                                                                   {AxisRole::polarization, 3},
                                                                   {AxisRole::time, 1}};
    Require(sky.axes.size() == expected.size(), "SKY did not report five axes");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        Require(sky.axes[index].role == expected[index].first, "axis " + std::to_string(index) + " has the wrong role");
        Require(sky.axes[index].length == expected[index].second,
                "axis " + std::to_string(index) + " has the wrong length");
    }
}

void TestDirection(const carta::zarr::ImageDescriptor& sky) {
    Require(sky.direction.has_value(), "no direction coordinate was reported");
    const auto& direction = *sky.direction;
    Require(direction.projection == "SIN", "projection did not survive the conversion");
    Require(direction.projection_parameters == std::vector<double>{0.0, 0.0}, "projection parameters were dropped");
    Require(direction.reference_frame == "fk5", "RADESYS did not survive as the reference frame");
    Require(direction.equinox.has_value(), "EQUINOX was dropped");
    RequireClose(*direction.equinox, 2000.0, 1.0e-9, "equinox");
    // LONPOLE 180 and LATPOLE 30 are stored in radians and reported in degrees.
    RequireClose(direction.native_pole_direction[0], 180.0, 1.0e-9, "native pole longitude");
    RequireClose(direction.native_pole_direction[1], 30.0, 1.0e-9, "native pole latitude");
    RequireClose(direction.reference_value[0], 45.0, 1.0e-9, "reference longitude");
    RequireClose(direction.reference_value[1], 30.0, 1.0e-9, "reference latitude");
}

void TestSpectralAndPolarization(const carta::zarr::ImageDescriptor& sky) {
    Require(sky.spectral.has_value(), "no spectral coordinate was reported");
    const auto& spectral = *sky.spectral;
    Require(spectral.channel_frequencies.size() == 2, "the channel table lost a channel");
    RequireClose(spectral.channel_frequencies[0], 1.4e9, 1.0, "first channel frequency");
    RequireClose(spectral.channel_frequencies[1], 1.401e9, 1.0, "second channel frequency");
    Require(spectral.unit == "Hz", "the frequency unit was not taken from reference_frequency's attrs");
    Require(spectral.system == "lsrk", "SPECSYS did not survive as the spectral system");
    Require(spectral.rest_frequency.has_value(), "RESTFRQ was dropped");
    RequireClose(*spectral.rest_frequency, 1.420405751e9, 1.0, "rest frequency");
    // These channels are evenly spaced, so the optional linear description must be present.
    Require(spectral.increment.has_value(), "evenly spaced channels reported no increment");
    RequireClose(*spectral.increment, 1.0e6, 1.0, "channel increment");

    // The polarization labels are a zstd-compressed fixed_length_utf32 array, which TensorStore
    // cannot read: this is the custom string decoder running against XRADIO's real output.
    Require(sky.polarization.has_value(), "no polarization coordinate was reported");
    Require(sky.polarization->labels == std::vector<std::string>({"I", "Q", "U"}),
            "the compressed polarization label array did not decode to the FITS Stokes ordering");
}

void TestTemporalAndStorage(const carta::zarr::ImageDescriptor& sky) {
    // DATE-OBS 2020-05-31T12:00:00 is MJD 59000.5. The FITS path writes MJD days, not the unix
    // seconds the schema document describes.
    Require(sky.temporal.has_value(), "no time coordinate was reported");
    Require(sky.temporal->values.size() == 1, "the time coordinate lost its only value");
    RequireClose(sky.temporal->values[0], 59000.5, 1.0e-6, "time value");
    Require(sky.temporal->unit == "d", "the time unit changed");
    Require(sky.temporal->scale == "utc", "the time scale changed");
    Require(sky.temporal->format == "mjd", "the time format changed");

    Require(sky.storage.has_value(), "no storage layout was reported");
    Require(!sky.storage->sharded, "XRADIO's zarr writer started sharding");
    Require(sky.storage->compressor == "zstd", "XRADIO's zarr writer changed compressor");
    Require((sky.storage->chunk_shape == std::vector<std::uint64_t>{1, 2, 3, 5, 4}),
            "the chunk shape changed; it is reported in stored axis order");
}

}  // namespace

int main() {
    try {
        const auto image = OpenSky();
        const auto& sky = image.descriptor();
        TestIdentityAndAxes(sky);
        TestDirection(sky);
        TestSpectralAndPolarization(sky);
        TestTemporalAndStorage(sky);

        const auto beams = image.ReadBeams();
        Require(static_cast<bool>(beams), "BMAJ/BMIN/BPA did not survive as a readable beam");
        Require(!beams.value().empty(), "the beam table decoded to no beams");

        std::cout << "carta-zarr XRADIO conformance tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr XRADIO conformance tests failed: " << error.what() << '\n';
        return 1;
    }
}
