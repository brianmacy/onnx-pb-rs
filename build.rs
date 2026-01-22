//! Build script for compiling ONNX protobuf files.

use std::path::Path;

fn main() {
    let proto_file = "onnx/onnx.proto3";
    let include_dir = "third_party/onnx";

    // Verify proto file exists
    let full_path = Path::new(include_dir).join(proto_file);
    if !full_path.exists() {
        panic!(
            "ONNX proto file not found at {:?}. \
             Did you initialize the git submodule? \
             Run: git submodule update --init --recursive",
            full_path
        );
    }

    // Configure prost-build
    let mut config = prost_build::Config::new();
    config.btree_map(["."]);

    // Compile the proto files
    config
        .compile_protos(&[proto_file], &[include_dir])
        .expect("Failed to compile ONNX protobuf files");

    // Rerun if proto files change
    println!("cargo::rerun-if-changed={}", full_path.display());
    println!("cargo::rerun-if-changed=build.rs");
}
