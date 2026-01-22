//! ONNX protocol buffer bindings for Rust.
//!
//! This crate provides Rust bindings for the [ONNX](https://onnx.ai/) (Open Neural Network Exchange)
//! protocol buffer format, enabling you to read, create, and manipulate ONNX models in Rust.
//!
//! # Features
//!
//! - Load and save ONNX models
//! - Create tensors from Rust primitives
//! - Build ONNX graphs programmatically
//! - Full access to all ONNX protobuf types
//!
//! # Example
//!
//! ```rust
//! use onnx_pb::{TensorProto, tensor_proto::DataType};
//!
//! // Create a tensor from a vector
//! let tensor: TensorProto = vec![1.0f32, 2.0, 3.0].into();
//! assert_eq!(tensor.dims, vec![3]);
//! assert_eq!(tensor.data_type, DataType::Float as i32);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod attrs;
mod util;

pub use self::attrs::*;
pub use self::util::*;

// Allow doc formatting issues in generated protobuf code
#[allow(clippy::doc_overindented_list_items)]
#[allow(missing_docs)]
mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
pub use onnx_proto::*;

use self::{
    tensor_proto::DataType,
    tensor_shape_proto::{Dimension, dimension},
    type_proto::{Tensor, Value},
};

/// Converts an `i64` into a dimension value.
impl From<i64> for dimension::Value {
    fn from(v: i64) -> Self {
        dimension::Value::DimValue(v)
    }
}

/// Converts a `String` into a dimension parameter.
impl From<String> for dimension::Value {
    fn from(v: String) -> Self {
        dimension::Value::DimParam(v)
    }
}

/// Converts a vector of dimensions into a `TensorShapeProto`.
impl<T: Into<Dimension>> From<Vec<T>> for TensorShapeProto {
    fn from(v: Vec<T>) -> Self {
        TensorShapeProto {
            dim: v.into_iter().map(Into::into).collect(),
        }
    }
}

/// Converts a dimension value into a `Dimension`.
impl<V: Into<dimension::Value>> From<V> for Dimension {
    fn from(v: V) -> Self {
        tensor_shape_proto::Dimension {
            denotation: String::default(),
            value: Some(v.into()),
        }
    }
}

/// Converts a key-value tuple into a `StringStringEntryProto`.
impl<K: Into<String>, V: Into<String>> From<(K, V)> for StringStringEntryProto {
    fn from((k, v): (K, V)) -> Self {
        StringStringEntryProto {
            key: k.into(),
            value: v.into(),
        }
    }
}

/// Converts a string into a `ValueInfoProto` with just a name.
impl<T: Into<String>> From<T> for ValueInfoProto {
    fn from(name: T) -> Self {
        ValueInfoProto {
            name: name.into(),
            ..ValueInfoProto::default()
        }
    }
}

/// Creates a scalar f32 tensor.
impl From<f32> for TensorProto {
    fn from(data: f32) -> TensorProto {
        TensorProto {
            dims: vec![1],
            float_data: vec![data],
            data_type: DataType::Float as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a 1D f32 tensor from a vector.
impl From<Vec<f32>> for TensorProto {
    fn from(data: Vec<f32>) -> TensorProto {
        TensorProto {
            dims: vec![data.len() as i64],
            float_data: data,
            data_type: DataType::Float as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a scalar i32 tensor.
impl From<i32> for TensorProto {
    fn from(data: i32) -> TensorProto {
        TensorProto {
            dims: vec![1],
            int32_data: vec![data],
            data_type: DataType::Int32 as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a 1D i32 tensor from a vector.
impl From<Vec<i32>> for TensorProto {
    fn from(data: Vec<i32>) -> TensorProto {
        TensorProto {
            dims: vec![data.len() as i64],
            int32_data: data,
            data_type: DataType::Int32 as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a scalar i64 tensor.
impl From<i64> for TensorProto {
    fn from(data: i64) -> TensorProto {
        TensorProto {
            dims: vec![1],
            int64_data: vec![data],
            data_type: DataType::Int64 as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a 1D i64 tensor from a vector.
impl From<Vec<i64>> for TensorProto {
    fn from(data: Vec<i64>) -> TensorProto {
        TensorProto {
            dims: vec![data.len() as i64],
            int64_data: data,
            data_type: DataType::Int64 as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a scalar f64 tensor.
impl From<f64> for TensorProto {
    fn from(data: f64) -> TensorProto {
        TensorProto {
            dims: vec![1],
            double_data: vec![data],
            data_type: DataType::Double as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a 1D f64 tensor from a vector.
impl From<Vec<f64>> for TensorProto {
    fn from(data: Vec<f64>) -> TensorProto {
        TensorProto {
            dims: vec![data.len() as i64],
            double_data: data,
            data_type: DataType::Double as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a scalar u64 tensor.
impl From<u64> for TensorProto {
    fn from(data: u64) -> TensorProto {
        TensorProto {
            dims: vec![1],
            uint64_data: vec![data],
            data_type: DataType::Uint64 as i32,
            ..TensorProto::default()
        }
    }
}

/// Creates a 1D u64 tensor from a vector.
impl From<Vec<u64>> for TensorProto {
    fn from(data: Vec<u64>) -> TensorProto {
        TensorProto {
            dims: vec![data.len() as i64],
            uint64_data: data,
            data_type: DataType::Uint64 as i32,
            ..TensorProto::default()
        }
    }
}

/// Converts a `DataType` into a `TypeProto`.
impl From<DataType> for TypeProto {
    fn from(typ: DataType) -> TypeProto {
        TypeProto {
            denotation: "".to_owned(),
            value: Some(Value::TensorType(Tensor {
                elem_type: typ as i32,
                shape: None,
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_value_from_i64() {
        let dim_val: dimension::Value = 42i64.into();
        match dim_val {
            dimension::Value::DimValue(v) => assert_eq!(v, 42),
            _ => panic!("Expected DimValue"),
        }
    }

    #[test]
    fn test_dimension_value_from_string() {
        let dim_val: dimension::Value = "batch".to_string().into();
        match dim_val {
            dimension::Value::DimParam(v) => assert_eq!(v, "batch"),
            _ => panic!("Expected DimParam"),
        }
    }

    #[test]
    fn test_tensor_shape_from_vec() {
        let shape: TensorShapeProto = vec![1i64, 3, 224, 224].into();
        assert_eq!(shape.dim.len(), 4);
        for (i, expected) in [1i64, 3, 224, 224].iter().enumerate() {
            match &shape.dim[i].value {
                Some(dimension::Value::DimValue(v)) => assert_eq!(v, expected),
                _ => panic!("Expected DimValue at index {}", i),
            }
        }
    }

    #[test]
    fn test_string_string_entry_from_tuple() {
        let entry: StringStringEntryProto = ("key", "value").into();
        assert_eq!(entry.key, "key");
        assert_eq!(entry.value, "value");
    }

    #[test]
    fn test_value_info_from_string() {
        let info: ValueInfoProto = "input_tensor".into();
        assert_eq!(info.name, "input_tensor");
    }

    #[test]
    fn test_tensor_proto_from_f32() {
        let tensor: TensorProto = 1.23f32.into();
        assert_eq!(tensor.dims, vec![1]);
        assert_eq!(tensor.float_data, vec![1.23f32]);
        assert_eq!(tensor.data_type, DataType::Float as i32);
    }

    #[test]
    fn test_tensor_proto_from_vec_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor: TensorProto = data.clone().into();
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.float_data, data);
        assert_eq!(tensor.data_type, DataType::Float as i32);
    }

    #[test]
    fn test_tensor_proto_from_i32() {
        let tensor: TensorProto = 42i32.into();
        assert_eq!(tensor.dims, vec![1]);
        assert_eq!(tensor.int32_data, vec![42i32]);
        assert_eq!(tensor.data_type, DataType::Int32 as i32);
    }

    #[test]
    fn test_tensor_proto_from_vec_i32() {
        let data = vec![1i32, 2, 3, 4];
        let tensor: TensorProto = data.clone().into();
        assert_eq!(tensor.dims, vec![4]);
        assert_eq!(tensor.int32_data, data);
        assert_eq!(tensor.data_type, DataType::Int32 as i32);
    }

    #[test]
    fn test_tensor_proto_from_i64() {
        let tensor: TensorProto = 100i64.into();
        assert_eq!(tensor.dims, vec![1]);
        assert_eq!(tensor.int64_data, vec![100i64]);
        assert_eq!(tensor.data_type, DataType::Int64 as i32);
    }

    #[test]
    fn test_tensor_proto_from_vec_i64() {
        let data = vec![10i64, 20, 30];
        let tensor: TensorProto = data.clone().into();
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.int64_data, data);
        assert_eq!(tensor.data_type, DataType::Int64 as i32);
    }

    #[test]
    fn test_tensor_proto_from_f64() {
        let tensor: TensorProto = 1.234f64.into();
        assert_eq!(tensor.dims, vec![1]);
        assert_eq!(tensor.double_data, vec![1.234f64]);
        assert_eq!(tensor.data_type, DataType::Double as i32);
    }

    #[test]
    fn test_tensor_proto_from_vec_f64() {
        let data = vec![1.1f64, 2.2, 3.3];
        let tensor: TensorProto = data.clone().into();
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.double_data, data);
        assert_eq!(tensor.data_type, DataType::Double as i32);
    }

    #[test]
    fn test_tensor_proto_from_u64() {
        let tensor: TensorProto = 999u64.into();
        assert_eq!(tensor.dims, vec![1]);
        assert_eq!(tensor.uint64_data, vec![999u64]);
        assert_eq!(tensor.data_type, DataType::Uint64 as i32);
    }

    #[test]
    fn test_tensor_proto_from_vec_u64() {
        let data = vec![100u64, 200, 300];
        let tensor: TensorProto = data.clone().into();
        assert_eq!(tensor.dims, vec![3]);
        assert_eq!(tensor.uint64_data, data);
        assert_eq!(tensor.data_type, DataType::Uint64 as i32);
    }

    #[test]
    fn test_type_proto_from_data_type() {
        let type_proto: TypeProto = DataType::Float.into();
        assert_eq!(type_proto.denotation, "");
        match type_proto.value {
            Some(Value::TensorType(tensor)) => {
                assert_eq!(tensor.elem_type, DataType::Float as i32);
                assert!(tensor.shape.is_none());
            }
            _ => panic!("Expected TensorType"),
        }
    }

    #[test]
    fn test_empty_tensor() {
        let tensor: TensorProto = Vec::<f32>::new().into();
        assert_eq!(tensor.dims, vec![0]);
        assert!(tensor.float_data.is_empty());
    }
}
