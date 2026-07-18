use std::fmt::{Display, Formatter};

pub type AtlasResult<T> = Result<T, AtlasError>;

#[derive(Debug)]
pub enum AtlasError {
    Coordinate(String),
    InvalidFormat(String),
    LimitExceeded(String),
    MixedSchema { expected: u16, found: u16 },
    DigestMismatch,
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl Display for AtlasError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coordinate(message) => write!(formatter, "invalid Atlas coordinate: {message}"),
            Self::InvalidFormat(message) => write!(formatter, "invalid Atlas format: {message}"),
            Self::LimitExceeded(message) => write!(formatter, "Atlas limit exceeded: {message}"),
            Self::MixedSchema { expected, found } => {
                write!(
                    formatter,
                    "mixed Atlas schema: expected {expected}, found {found}"
                )
            }
            Self::DigestMismatch => formatter.write_str("Atlas payload digest mismatch"),
            Self::Io(error) => write!(formatter, "Atlas I/O error: {error}"),
            Self::Json(error) => write!(formatter, "Atlas manifest JSON error: {error}"),
        }
    }
}

impl std::error::Error for AtlasError {}

impl From<std::io::Error> for AtlasError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for AtlasError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
