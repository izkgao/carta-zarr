/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_RESULT_H_
#define CARTA_ZARR_RESULT_H_

#include "carta-zarr/error.h"

#include <optional>
#include <utility>
#include <variant>

namespace carta::zarr {

template <typename T> class Result {
public:
    Result(T value) : _value(std::move(value)) {}
    Result(Error error) : _value(std::move(error)) {}

    bool has_value() const noexcept {
        return std::holds_alternative<T>(_value);
    }
    explicit operator bool() const noexcept {
        return has_value();
    }

    T& value() & {
        return std::get<T>(_value);
    }
    const T& value() const& {
        return std::get<T>(_value);
    }
    T&& value() && {
        return std::get<T>(std::move(_value));
    }

    Error& error() & {
        return std::get<Error>(_value);
    }
    const Error& error() const& {
        return std::get<Error>(_value);
    }

private:
    std::variant<T, Error> _value;
};

template <> class Result<void> {
public:
    Result() noexcept = default;
    Result(Error error) : _error(std::move(error)) {}

    bool has_value() const noexcept {
        return !_error.has_value();
    }
    explicit operator bool() const noexcept {
        return has_value();
    }

    const Error& error() const& {
        return *_error;
    }
    Error& error() & {
        return *_error;
    }

private:
    std::optional<Error> _error;
};

} // namespace carta::zarr

#endif // CARTA_ZARR_RESULT_H_
