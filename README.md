# onnx-pb

[![CI](https://github.com/brianmacy/onnx-pb-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/brianmacy/onnx-pb-rs/actions/workflows/ci.yml)
[![Crate](https://img.shields.io/crates/v/onnx-pb.svg)](https://crates.io/crates/onnx-pb)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://brianmacy.github.io/onnx-pb-rs/onnx_pb/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

ONNX protocol buffer bindings for Rust.

This crate provides Rust bindings for the [ONNX](https://onnx.ai/) (Open Neural Network Exchange) protocol buffer format, enabling you to read, create, and manipulate ONNX models in Rust.

## Features

- Load and save ONNX models
- Create tensors from Rust primitives
- Build ONNX graphs programmatically
- Full access to all ONNX protobuf types
- ONNX v1.20.1 support (opset 21)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
onnx-pb = "0.3"
```

## Usage

### Loading an ONNX Model

```rust
use onnx_pb::open_model;

let model = open_model("model.onnx").expect("Failed to load model");
println!("IR version: {}", model.ir_version);
println!("Producer: {}", model.producer_name);

if let Some(graph) = &model.graph {
    println!("Graph name: {}", graph.name);
    println!("Nodes: {}", graph.node.len());
}
```

### Creating Tensors

```rust
use onnx_pb::{TensorProto, tensor_proto::DataType};

// From a vector
let tensor: TensorProto = vec![1.0f32, 2.0, 3.0, 4.0].into();
assert_eq!(tensor.dims, vec![4]);
assert_eq!(tensor.data_type, DataType::Float as i32);

// From a scalar
let scalar: TensorProto = 42i64.into();
assert_eq!(scalar.dims, vec![1]);

// Various types supported: f32, f64, i32, i64, u64
let double_tensor: TensorProto = vec![1.0f64, 2.0].into();
let int_tensor: TensorProto = vec![1i32, 2, 3].into();
```

### Creating Attributes

```rust
use onnx_pb::make_attribute;

// Float attribute
let alpha = make_attribute("alpha", 0.5f32);

// Integer attribute
let axis = make_attribute("axis", 1i64);

// String attribute
let mode = make_attribute("mode", "nearest");

// Boolean attribute (converted to int)
let keepdims = make_attribute("keepdims", true);

// List of integers
let pads = make_attribute("pads", vec![1i64, 2, 1, 2]);
```

### Building a Model

```rust
use onnx_pb::{
    save_model, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
};

// Create a node
let add_node = NodeProto {
    input: vec!["A".to_string(), "B".to_string()],
    output: vec!["C".to_string()],
    name: "add_node".to_string(),
    op_type: "Add".to_string(),
    ..NodeProto::default()
};

// Create graph
let graph = GraphProto {
    name: "my_graph".to_string(),
    node: vec![add_node],
    input: vec!["A".into(), "B".into()],
    output: vec!["C".into()],
    ..GraphProto::default()
};

// Create model
let model = ModelProto {
    ir_version: 9,
    graph: Some(graph),
    opset_import: vec![OperatorSetIdProto {
        domain: "".to_string(),
        version: 21,
    }],
    producer_name: "my_app".to_string(),
    ..ModelProto::default()
};

// Save
save_model("output.onnx", &model).expect("Failed to save model");
```

### Working with Shapes

```rust
use onnx_pb::{TensorShapeProto, tensor_shape_proto::dimension};

// Create shape from dimensions
let shape: TensorShapeProto = vec![1i64, 3, 224, 224].into();

// Dynamic dimensions with symbolic names
let dim_val: dimension::Value = "batch_size".to_string().into();
```

## API Overview

### Types

- `ModelProto` - Top-level ONNX model container
- `GraphProto` - Computation graph with nodes, inputs, outputs
- `NodeProto` - Individual operation node
- `TensorProto` - Tensor data
- `ValueInfoProto` - Type and shape information
- `AttributeProto` - Node attribute
- `TypeProto` - Type descriptor

### Functions

- `open_model(path)` - Load an ONNX model from file
- `save_model(path, model)` - Save an ONNX model to file
- `make_attribute(name, value)` - Create an attribute

### Error Handling

```rust
use onnx_pb::{open_model, Error};

match open_model("model.onnx") {
    Ok(model) => println!("Loaded model"),
    Err(Error::Io(e)) => eprintln!("IO error: {}", e),
    Err(Error::Decode(e)) => eprintln!("Invalid ONNX file: {}", e),
    Err(Error::Encode(e)) => eprintln!("Encoding error: {}", e),
}
```

## Minimum Supported Rust Version

This crate requires Rust 1.85 or later (Edition 2024).

## License

MIT license, same as [ONNX](https://github.com/onnx/onnx).
