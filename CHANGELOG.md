# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-22

### Added

- GitHub Actions CI pipeline with checks, clippy, tests, formatting, and docs
- GitHub Pages documentation deployment
- Comprehensive unit tests for all `From` implementations
- Integration tests for full model workflows
- Proper `std::error::Error` implementation for `Error` type
- `From` implementations for error conversion
- Documentation for all public items
- MSRV badge and documentation

### Changed

- **Breaking**: Updated to Rust Edition 2024 (requires Rust 1.85+)
- Updated ONNX submodule from v1.6.1 to v1.20.1 (opset 21)
- Improved `build.rs` with better error messages and modern prost-build configuration
- Enhanced README with comprehensive usage examples
- Fixed `Attribute::Floats` Display implementation (was incorrectly using "ints_" prefix)
- Fixed `Attribute::Graph` Display implementation (was using Debug format)

### Fixed

- Display implementation for `Attribute::Floats` variant
- Display implementation for `Attribute::Graph` variant

## [0.2.0] - 2024-xx-xx

### Changed

- Updated prost to 0.13
- Updated bytes to 1.x
- Fixed RUSTSEC-2021-0073 security advisory

## [0.1.1] - Previous

### Added

- Exposed `make_attribute` function
- Moved helpers to this crate
- Convert data type to type proto

## [0.1.0] - Initial Release

### Added

- Initial ONNX protocol buffer bindings
- Basic model I/O utilities
- Attribute helpers
