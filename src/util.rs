//! Model saving and opening.

use std::path::Path;

use prost::Message;

use crate::ModelProto;

/// File utils error.
#[derive(Debug)]
pub enum Error {
    /// IO error.
    Io(std::io::Error),

    /// Decode error.
    Decode(prost::DecodeError),

    /// Encode error.
    Encode(prost::EncodeError),
}

/// Opens model from a file.
pub fn open_model<P: AsRef<Path>>(path: P) -> Result<ModelProto, Error> {
    let body = std::fs::read(path).map_err(Error::Io)?;
    ModelProto::decode(body.as_slice()).map_err(Error::Decode)
}

/// Saves model to a file.
pub fn save_model<P: AsRef<Path>>(path: P, model: &ModelProto) -> Result<(), Error> {
    let mut body = Vec::new();
    model.encode(&mut body).map_err(Error::Encode)?;
    std::fs::write(path, body).map_err(Error::Io)?;
    Ok(())
}
