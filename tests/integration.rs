//! Integration tests for onnx-pb crate.
//!
//! These tests verify full workflows: creating models, saving, loading, and verification.

use onnx_pb::{
    GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
    make_attribute, open_model, save_model, tensor_proto::DataType, type_proto::Value,
};
use tempfile::tempdir;

/// Creates a minimal valid ONNX model with a single Add operation.
fn create_test_model() -> ModelProto {
    // Create input value infos
    let input_a = ValueInfoProto {
        name: "A".to_string(),
        r#type: Some(TypeProto {
            denotation: "".to_string(),
            value: Some(Value::TensorType(onnx_pb::type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(3)),
                        },
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(4)),
                        },
                    ],
                }),
            })),
        }),
        doc_string: "".to_string(),
        metadata_props: vec![],
    };

    let input_b = ValueInfoProto {
        name: "B".to_string(),
        r#type: Some(TypeProto {
            denotation: "".to_string(),
            value: Some(Value::TensorType(onnx_pb::type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(3)),
                        },
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(4)),
                        },
                    ],
                }),
            })),
        }),
        doc_string: "".to_string(),
        metadata_props: vec![],
    };

    let output = ValueInfoProto {
        name: "C".to_string(),
        r#type: Some(TypeProto {
            denotation: "".to_string(),
            value: Some(Value::TensorType(onnx_pb::type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(3)),
                        },
                        onnx_pb::tensor_shape_proto::Dimension {
                            denotation: "".to_string(),
                            value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(4)),
                        },
                    ],
                }),
            })),
        }),
        doc_string: "".to_string(),
        metadata_props: vec![],
    };

    // Create Add node using Default
    let add_node = NodeProto {
        input: vec!["A".to_string(), "B".to_string()],
        output: vec!["C".to_string()],
        name: "add_node".to_string(),
        op_type: "Add".to_string(),
        ..Default::default()
    };

    // Create graph
    let graph = GraphProto {
        node: vec![add_node],
        name: "test_graph".to_string(),
        initializer: vec![],
        sparse_initializer: vec![],
        doc_string: "Test graph for integration tests".to_string(),
        input: vec![input_a, input_b],
        output: vec![output],
        value_info: vec![],
        quantization_annotation: vec![],
        metadata_props: vec![],
    };

    // Create model using Default
    ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_pb::OperatorSetIdProto {
            domain: "".to_string(),
            version: 21,
        }],
        producer_name: "onnx-pb-rs".to_string(),
        producer_version: "0.3.0".to_string(),
        domain: "".to_string(),
        model_version: 1,
        doc_string: "Integration test model".to_string(),
        graph: Some(graph),
        ..Default::default()
    }
}

#[test]
fn test_create_and_save_model() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("test_model.onnx");

    let model = create_test_model();
    save_model(&model_path, &model).expect("Failed to save model");

    assert!(model_path.exists());
    let metadata = std::fs::metadata(&model_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_full_roundtrip() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("roundtrip_model.onnx");

    // Create and save
    let original = create_test_model();
    save_model(&model_path, &original).expect("Failed to save model");

    // Load
    let loaded = open_model(&model_path).expect("Failed to load model");

    // Verify model properties
    assert_eq!(loaded.ir_version, 9);
    assert_eq!(loaded.producer_name, "onnx-pb-rs");
    assert_eq!(loaded.producer_version, "0.3.0");
    assert_eq!(loaded.model_version, 1);
    assert_eq!(loaded.doc_string, "Integration test model");

    // Verify opset
    assert_eq!(loaded.opset_import.len(), 1);
    assert_eq!(loaded.opset_import[0].version, 21);

    // Verify graph
    let graph = loaded.graph.expect("Model should have a graph");
    assert_eq!(graph.name, "test_graph");
    assert_eq!(graph.input.len(), 2);
    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.node.len(), 1);

    // Verify node
    let node = &graph.node[0];
    assert_eq!(node.op_type, "Add");
    assert_eq!(node.input, vec!["A", "B"]);
    assert_eq!(node.output, vec!["C"]);
}

#[test]
fn test_tensor_conversions() {
    // Test f32 tensor
    let f32_tensor: TensorProto = vec![1.0f32, 2.0, 3.0, 4.0].into();
    assert_eq!(f32_tensor.dims, vec![4]);
    assert_eq!(f32_tensor.data_type, DataType::Float as i32);
    assert_eq!(f32_tensor.float_data, vec![1.0, 2.0, 3.0, 4.0]);

    // Test i64 tensor
    let i64_tensor: TensorProto = vec![10i64, 20, 30].into();
    assert_eq!(i64_tensor.dims, vec![3]);
    assert_eq!(i64_tensor.data_type, DataType::Int64 as i32);
    assert_eq!(i64_tensor.int64_data, vec![10, 20, 30]);

    // Test scalar
    let scalar: TensorProto = 42.0f64.into();
    assert_eq!(scalar.dims, vec![1]);
    assert_eq!(scalar.data_type, DataType::Double as i32);
    assert_eq!(scalar.double_data, vec![42.0]);
}

#[test]
fn test_attribute_creation() {
    // Float attribute
    let float_attr = make_attribute("alpha", 0.5f32);
    assert_eq!(float_attr.name, "alpha");
    assert!((float_attr.f - 0.5f32).abs() < f32::EPSILON);

    // Int attribute
    let int_attr = make_attribute("axis", 1i64);
    assert_eq!(int_attr.name, "axis");
    assert_eq!(int_attr.i, 1);

    // String attribute
    let string_attr = make_attribute("mode", "linear");
    assert_eq!(string_attr.name, "mode");
    assert_eq!(string_attr.s, b"linear");

    // Ints attribute
    let ints_attr = make_attribute("pads", vec![1i64, 2, 1, 2]);
    assert_eq!(ints_attr.name, "pads");
    assert_eq!(ints_attr.ints, vec![1, 2, 1, 2]);
}

#[test]
fn test_model_with_initializers() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model_with_init.onnx");

    // Create weight tensor
    let weights: TensorProto = TensorProto {
        dims: vec![2, 3],
        float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        data_type: DataType::Float as i32,
        name: "weights".to_string(),
        ..TensorProto::default()
    };

    // Create a simple model with initializer
    let graph = GraphProto {
        name: "graph_with_init".to_string(),
        initializer: vec![weights],
        ..GraphProto::default()
    };

    let model = ModelProto {
        ir_version: 9,
        graph: Some(graph),
        ..ModelProto::default()
    };

    // Save and reload
    save_model(&model_path, &model).expect("Failed to save model");
    let loaded = open_model(&model_path).expect("Failed to load model");

    // Verify initializer
    let graph = loaded.graph.expect("Model should have graph");
    assert_eq!(graph.initializer.len(), 1);
    let init = &graph.initializer[0];
    assert_eq!(init.name, "weights");
    assert_eq!(init.dims, vec![2, 3]);
    assert_eq!(init.float_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_type_proto_conversions() {
    let type_proto: TypeProto = DataType::Float.into();
    match type_proto.value {
        Some(Value::TensorType(tensor)) => {
            assert_eq!(tensor.elem_type, DataType::Float as i32);
        }
        _ => panic!("Expected TensorType"),
    }
}

#[test]
fn test_tensor_shape_from_vec() {
    let shape: TensorShapeProto = vec![1i64, 3, 224, 224].into();
    assert_eq!(shape.dim.len(), 4);

    // Verify each dimension
    let expected_dims = [1i64, 3, 224, 224];
    for (i, expected) in expected_dims.iter().enumerate() {
        match &shape.dim[i].value {
            Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                assert_eq!(v, expected);
            }
            _ => panic!("Expected DimValue at index {}", i),
        }
    }
}

#[test]
fn test_multiple_nodes_graph() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("multi_node_model.onnx");

    // Create a graph with multiple operations: Add -> Relu
    let add_node = NodeProto {
        input: vec!["A".to_string(), "B".to_string()],
        output: vec!["add_out".to_string()],
        name: "add_node".to_string(),
        op_type: "Add".to_string(),
        ..Default::default()
    };

    let relu_node = NodeProto {
        input: vec!["add_out".to_string()],
        output: vec!["C".to_string()],
        name: "relu_node".to_string(),
        op_type: "Relu".to_string(),
        ..Default::default()
    };

    let graph = GraphProto {
        name: "multi_node_graph".to_string(),
        node: vec![add_node, relu_node],
        input: vec!["A".into(), "B".into()],
        output: vec!["C".into()],
        ..GraphProto::default()
    };

    let model = ModelProto {
        ir_version: 9,
        graph: Some(graph),
        opset_import: vec![onnx_pb::OperatorSetIdProto {
            domain: "".to_string(),
            version: 21,
        }],
        ..Default::default()
    };

    // Save and reload
    save_model(&model_path, &model).expect("Failed to save model");
    let loaded = open_model(&model_path).expect("Failed to load model");

    // Verify
    let graph = loaded.graph.expect("Model should have graph");
    assert_eq!(graph.node.len(), 2);
    assert_eq!(graph.node[0].op_type, "Add");
    assert_eq!(graph.node[1].op_type, "Relu");
}
