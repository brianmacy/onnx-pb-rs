//! ONNX node attribute helpers.
//!
//! This module provides convenient types and functions for creating ONNX node attributes.

use crate::{AttributeProto, GraphProto, TensorProto, attribute_proto::AttributeType};

/// Wrapper type for axis indices.
///
/// Used to represent axis/axes parameters for ONNX operations.
///
/// # Example
///
/// ```rust
/// use onnx_pb::Axes;
///
/// // Single axis
/// let axis: Axes = 0i64.into();
///
/// // Multiple axes
/// let axes: Axes = vec![0i64, 1, 2].into();
/// ```
pub struct Axes(pub Vec<i64>);

impl From<i64> for Axes {
    fn from(axes: i64) -> Self {
        Axes(vec![axes])
    }
}

impl From<Vec<i64>> for Axes {
    fn from(axes: Vec<i64>) -> Self {
        Axes(axes)
    }
}

/// Enum representing all possible ONNX attribute types.
///
/// This provides a type-safe way to construct ONNX attributes before
/// converting them to `AttributeProto`.
///
/// # Example
///
/// ```rust
/// use onnx_pb::{Attribute, make_attribute};
///
/// // Create attributes of different types
/// let float_attr = make_attribute("alpha", Attribute::Float(0.5));
/// let int_attr = make_attribute("axis", Attribute::Int(1));
/// let string_attr = make_attribute("mode", "constant");
/// ```
#[derive(Clone, Debug)]
pub enum Attribute {
    /// Single float value.
    Float(f32),
    /// List of float values.
    Floats(Vec<f32>),
    /// Single integer value.
    Int(i64),
    /// List of integer values.
    Ints(Vec<i64>),
    /// Raw bytes (stored as string in ONNX).
    Bytes(Vec<u8>),
    /// String value.
    String(String),
    /// List of string values.
    Strings(Vec<String>),
    /// Tensor value.
    Tensor(TensorProto),
    /// List of tensor values.
    Tensors(Vec<TensorProto>),
    /// Graph value (for control flow ops).
    Graph(GraphProto),
    /// List of graph values.
    Graphs(Vec<GraphProto>),
}

macro_rules! attr_converter {
    ( $a:ident, $b:ty ) => {
        impl From<$b> for Attribute {
            fn from(v: $b) -> Self {
                Attribute::$a(v)
            }
        }
    };
}

impl From<bool> for Attribute {
    fn from(v: bool) -> Self {
        Attribute::Int(if v { 1 } else { 0 })
    }
}

impl From<Axes> for Attribute {
    fn from(axes: Axes) -> Self {
        Attribute::Ints(axes.0)
    }
}

attr_converter!(Float, f32);
attr_converter!(Floats, Vec<f32>);
attr_converter!(Int, i64);
attr_converter!(Bytes, Vec<u8>);
attr_converter!(String, String);
attr_converter!(Strings, Vec<String>);
attr_converter!(Ints, Vec<i64>);
attr_converter!(Tensor, TensorProto);
attr_converter!(Tensors, Vec<TensorProto>);
attr_converter!(Graph, GraphProto);
attr_converter!(Graphs, Vec<GraphProto>);

impl From<&str> for Attribute {
    fn from(v: &str) -> Self {
        v.to_owned().into()
    }
}

impl From<Vec<&str>> for Attribute {
    fn from(v: Vec<&str>) -> Self {
        v.into_iter()
            .map(|s| s.to_owned())
            .collect::<Vec<_>>()
            .into()
    }
}

/// Creates a new ONNX attribute from a name and value.
///
/// # Arguments
///
/// * `name` - The attribute name
/// * `attribute` - The attribute value (any type that implements `Into<Attribute>`)
///
/// # Example
///
/// ```rust
/// use onnx_pb::make_attribute;
///
/// // Float attribute
/// let attr = make_attribute("alpha", 0.5f32);
///
/// // Integer attribute
/// let attr = make_attribute("axis", 1i64);
///
/// // String attribute
/// let attr = make_attribute("mode", "nearest");
///
/// // Boolean attribute (converted to int)
/// let attr = make_attribute("keepdims", true);
/// ```
pub fn make_attribute<S: Into<String>, A: Into<Attribute>>(
    name: S,
    attribute: A,
) -> AttributeProto {
    let mut attr_proto = AttributeProto {
        name: name.into(),
        ..AttributeProto::default()
    };
    match attribute.into() {
        Attribute::Float(val) => {
            attr_proto.f = val;
            attr_proto.r#type = AttributeType::Float as i32;
        }
        Attribute::Floats(vals) => {
            attr_proto.floats = vals;
            attr_proto.r#type = AttributeType::Floats as i32;
        }
        Attribute::Int(val) => {
            attr_proto.i = val;
            attr_proto.r#type = AttributeType::Int as i32;
        }
        Attribute::Ints(vals) => {
            attr_proto.ints = vals;
            attr_proto.r#type = AttributeType::Ints as i32;
        }
        Attribute::Bytes(val) => {
            attr_proto.s = val;
            attr_proto.r#type = AttributeType::String as i32;
        }
        Attribute::String(val) => {
            attr_proto.s = val.into();
            attr_proto.r#type = AttributeType::String as i32;
        }
        Attribute::Strings(vals) => {
            attr_proto.strings = vals.into_iter().map(Into::into).collect();
            attr_proto.r#type = AttributeType::Strings as i32;
        }
        Attribute::Graph(val) => {
            attr_proto.g = Some(val);
            attr_proto.r#type = AttributeType::Graph as i32;
        }
        Attribute::Graphs(vals) => {
            attr_proto.graphs = vals;
            attr_proto.r#type = AttributeType::Graphs as i32;
        }
        Attribute::Tensor(val) => {
            attr_proto.t = Some(val);
            attr_proto.r#type = AttributeType::Tensor as i32;
        }
        Attribute::Tensors(vals) => {
            attr_proto.tensors = vals;
            attr_proto.r#type = AttributeType::Tensors as i32;
        }
    };
    attr_proto
}

impl std::fmt::Display for Attribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Attribute::Float(v) => write!(f, "float_{:.2}", v),
            Attribute::Floats(v) => write!(
                f,
                "floats_{}",
                v.iter()
                    .map(|t| format!("{:.2}", t))
                    .collect::<Vec<_>>()
                    .join("_")
            ),
            Attribute::Int(v) => write!(f, "int_{}", v),
            Attribute::Ints(v) => write!(
                f,
                "ints_{}",
                v.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join("_")
            ),
            Attribute::Bytes(v) => write!(f, "bytes_{:?}", v),
            Attribute::String(v) => write!(f, "string_{}", v),
            Attribute::Strings(v) => write!(f, "strings_{}", v.join("_")),
            Attribute::Tensor(v) => write!(f, "tensor_{}", v.name),
            Attribute::Tensors(v) => write!(
                f,
                "tensors_{}",
                v.iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join("_")
            ),
            Attribute::Graph(v) => write!(f, "graph_{}", v.name),
            Attribute::Graphs(v) => write!(
                f,
                "graphs_{}",
                v.iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join("_")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axes_from_i64() {
        let axes: Axes = 5i64.into();
        assert_eq!(axes.0, vec![5]);
    }

    #[test]
    fn test_axes_from_vec() {
        let axes: Axes = vec![0i64, 1, 2].into();
        assert_eq!(axes.0, vec![0, 1, 2]);
    }

    #[test]
    fn test_attribute_from_bool_true() {
        let attr: Attribute = true.into();
        match attr {
            Attribute::Int(v) => assert_eq!(v, 1),
            _ => panic!("Expected Int attribute"),
        }
    }

    #[test]
    fn test_attribute_from_bool_false() {
        let attr: Attribute = false.into();
        match attr {
            Attribute::Int(v) => assert_eq!(v, 0),
            _ => panic!("Expected Int attribute"),
        }
    }

    #[test]
    fn test_attribute_from_axes() {
        let axes = Axes(vec![1, 2, 3]);
        let attr: Attribute = axes.into();
        match attr {
            Attribute::Ints(v) => assert_eq!(v, vec![1, 2, 3]),
            _ => panic!("Expected Ints attribute"),
        }
    }

    #[test]
    fn test_attribute_from_f32() {
        let attr: Attribute = 1.23f32.into();
        match attr {
            Attribute::Float(v) => assert!((v - 1.23f32).abs() < f32::EPSILON),
            _ => panic!("Expected Float attribute"),
        }
    }

    #[test]
    fn test_attribute_from_vec_f32() {
        let attr: Attribute = vec![1.0f32, 2.0, 3.0].into();
        match attr {
            Attribute::Floats(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
            _ => panic!("Expected Floats attribute"),
        }
    }

    #[test]
    fn test_attribute_from_i64() {
        let attr: Attribute = 42i64.into();
        match attr {
            Attribute::Int(v) => assert_eq!(v, 42),
            _ => panic!("Expected Int attribute"),
        }
    }

    #[test]
    fn test_attribute_from_vec_i64() {
        let attr: Attribute = vec![1i64, 2, 3].into();
        match attr {
            Attribute::Ints(v) => assert_eq!(v, vec![1, 2, 3]),
            _ => panic!("Expected Ints attribute"),
        }
    }

    #[test]
    fn test_attribute_from_string() {
        let attr: Attribute = "test".to_string().into();
        match attr {
            Attribute::String(v) => assert_eq!(v, "test"),
            _ => panic!("Expected String attribute"),
        }
    }

    #[test]
    fn test_attribute_from_str() {
        let attr: Attribute = "test".into();
        match attr {
            Attribute::String(v) => assert_eq!(v, "test"),
            _ => panic!("Expected String attribute"),
        }
    }

    #[test]
    fn test_attribute_from_vec_str() {
        let attr: Attribute = vec!["a", "b", "c"].into();
        match attr {
            Attribute::Strings(v) => assert_eq!(v, vec!["a", "b", "c"]),
            _ => panic!("Expected Strings attribute"),
        }
    }

    #[test]
    fn test_make_attribute_float() {
        let attr = make_attribute("alpha", 0.5f32);
        assert_eq!(attr.name, "alpha");
        assert_eq!(attr.r#type, AttributeType::Float as i32);
        assert!((attr.f - 0.5f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_make_attribute_int() {
        let attr = make_attribute("axis", 1i64);
        assert_eq!(attr.name, "axis");
        assert_eq!(attr.r#type, AttributeType::Int as i32);
        assert_eq!(attr.i, 1);
    }

    #[test]
    fn test_make_attribute_string() {
        let attr = make_attribute("mode", "nearest");
        assert_eq!(attr.name, "mode");
        assert_eq!(attr.r#type, AttributeType::String as i32);
        assert_eq!(attr.s, b"nearest");
    }

    #[test]
    fn test_make_attribute_ints() {
        let attr = make_attribute("pads", vec![1i64, 2, 3, 4]);
        assert_eq!(attr.name, "pads");
        assert_eq!(attr.r#type, AttributeType::Ints as i32);
        assert_eq!(attr.ints, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_make_attribute_bool() {
        let attr = make_attribute("keepdims", true);
        assert_eq!(attr.name, "keepdims");
        assert_eq!(attr.r#type, AttributeType::Int as i32);
        assert_eq!(attr.i, 1);
    }

    #[test]
    fn test_attribute_display_float() {
        let attr = Attribute::Float(1.23);
        assert_eq!(format!("{}", attr), "float_1.23");
    }

    #[test]
    fn test_attribute_display_floats() {
        let attr = Attribute::Floats(vec![1.0, 2.0]);
        assert_eq!(format!("{}", attr), "floats_1.00_2.00");
    }

    #[test]
    fn test_attribute_display_int() {
        let attr = Attribute::Int(42);
        assert_eq!(format!("{}", attr), "int_42");
    }

    #[test]
    fn test_attribute_display_ints() {
        let attr = Attribute::Ints(vec![1, 2, 3]);
        assert_eq!(format!("{}", attr), "ints_1_2_3");
    }

    #[test]
    fn test_attribute_display_string() {
        let attr = Attribute::String("test".to_string());
        assert_eq!(format!("{}", attr), "string_test");
    }

    #[test]
    fn test_attribute_display_strings() {
        let attr = Attribute::Strings(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(format!("{}", attr), "strings_a_b");
    }

    #[test]
    fn test_attribute_clone() {
        let attr = Attribute::Int(42);
        let cloned = attr.clone();
        match cloned {
            Attribute::Int(v) => assert_eq!(v, 42),
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_attribute_debug() {
        let attr = Attribute::Int(42);
        let debug_str = format!("{:?}", attr);
        assert!(debug_str.contains("Int"));
        assert!(debug_str.contains("42"));
    }
}
