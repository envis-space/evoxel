use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    EvoxelError(#[from] evoxel::Error),
    #[error(transparent)]
    EvoxelIoError(#[from] evoxel::io::Error),
    #[error(transparent)]
    EvoxelTransformError(#[from] evoxel::transform::Error),

    #[error(transparent)]
    StdIoError(#[from] std::io::Error),
}
