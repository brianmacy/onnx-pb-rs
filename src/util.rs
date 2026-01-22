//! Model saving and opening utilities.
//!
//! This module provides functions for reading and writing ONNX model files.

use std::path::Path;

use prost::Message;

use crate::ModelProto;

/// Error type for model I/O operations.
#[derive(Debug)]
pub enum Error {
    /// IO error (file not found, permission denied, etc.).
    Io(std::io::Error),

    /// Protobuf decode error (invalid or corrupted model file).
    Decode(prost::DecodeError),

    /// Protobuf encode error.
    Encode(prost::EncodeError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Decode(e) => write!(f, "Decode error: {}", e),
            Error::Encode(e) => write!(f, "Encode error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Decode(e) => Some(e),
            Error::Encode(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<prost::DecodeError> for Error {
    fn from(e: prost::DecodeError) -> Self {
        Error::Decode(e)
    }
}

impl From<prost::EncodeError> for Error {
    fn from(e: prost::EncodeError) -> Self {
        Error::Encode(e)
    }
}

/// Opens an ONNX model from a file.
///
/// # Arguments
///
/// * `path` - Path to the ONNX model file
///
/// # Returns
///
/// The parsed `ModelProto` on success, or an `Error` on failure.
///
/// # Example
///
/// ```rust,no_run
/// use onnx_pb::open_model;
///
/// let model = open_model("model.onnx").expect("Failed to load model");
/// println!("Model IR version: {}", model.ir_version);
/// ```
pub fn open_model<P: AsRef<Path>>(path: P) -> Result<ModelProto, Error> {
    let body = std::fs::read(path)?;
    Ok(ModelProto::decode(body.as_slice())?)
}

/// Saves an ONNX model to a file.
///
/// # Arguments
///
/// * `path` - Path where the model should be saved
/// * `model` - Reference to the model to save
///
/// # Returns
///
/// `Ok(())` on success, or an `Error` on failure.
///
/// # Example
///
/// ```rust,no_run
/// use onnx_pb::{save_model, ModelProto};
///
/// let model = ModelProto::default();
/// save_model("output.onnx", &model).expect("Failed to save model");
/// ```
pub fn save_model<P: AsRef<Path>>(path: P, model: &ModelProto) -> Result<(), Error> {
    let mut body = Vec::new();
    model.encode(&mut body)?;
    std::fs::write(path, body)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;
    use std::io::ErrorKind;
    use tempfile::tempdir;

    #[test]
    fn test_open_model_not_found() {
        let result = open_model("/nonexistent/path/model.onnx");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Io(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_open_model_invalid_data() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid.onnx");
        std::fs::write(&file_path, b"not a valid protobuf").unwrap();

        let result = open_model(&file_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Decode(_) => {}
            _ => panic!("Expected Decode error"),
        }
    }

    #[test]
    fn test_save_and_open_model_roundtrip() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.onnx");

        // Create a minimal model
        let model = ModelProto {
            ir_version: 9,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: "test.domain".to_string(),
            model_version: 1,
            doc_string: "Test model".to_string(),
            ..ModelProto::default()
        };

        // Save
        save_model(&file_path, &model).expect("Failed to save model");

        // Verify file exists
        assert!(file_path.exists());

        // Load
        let loaded = open_model(&file_path).expect("Failed to load model");

        // Verify contents
        assert_eq!(loaded.ir_version, 9);
        assert_eq!(loaded.producer_name, "test");
        assert_eq!(loaded.producer_version, "1.0");
        assert_eq!(loaded.domain, "test.domain");
        assert_eq!(loaded.model_version, 1);
        assert_eq!(loaded.doc_string, "Test model");
    }

    #[test]
    fn test_save_model_permission_denied() {
        // Try to write to a directory that doesn't exist
        let result = save_model("/nonexistent/dir/model.onnx", &ModelProto::default());
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Io(_) => {}
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_error_display_io() {
        let io_err = std::io::Error::new(ErrorKind::NotFound, "file not found");
        let err = Error::Io(io_err);
        let display = format!("{}", err);
        assert!(display.contains("IO error"));
        assert!(display.contains("file not found"));
    }

    #[test]
    fn test_error_display_decode() {
        // Create a decode error by trying to decode invalid data
        let result = ModelProto::decode(&[0xff, 0xff, 0xff][..]);
        if let Err(decode_err) = result {
            let err = Error::Decode(decode_err);
            let display = format!("{}", err);
            assert!(display.contains("Decode error"));
        }
    }

    #[test]
    fn test_error_source() {
        let io_err = std::io::Error::new(ErrorKind::NotFound, "test");
        let err = Error::Io(io_err);
        assert!(StdError::source(&err).is_some());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(ErrorKind::NotFound, "test");
        let err: Error = io_err.into();
        match err {
            Error::Io(_) => {}
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_error_debug() {
        let io_err = std::io::Error::new(ErrorKind::NotFound, "test");
        let err = Error::Io(io_err);
        let debug = format!("{:?}", err);
        assert!(debug.contains("Io"));
    }
}
