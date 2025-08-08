use evoxel_core::{VoxelDataColumnType, VoxelGrid};
use nalgebra::Vector3;

pub fn translate(voxel_grid: &VoxelGrid, translation: Vector3<i64>) -> VoxelGrid {
    let mut translated_data = voxel_grid.voxel_data().clone();
    translated_data
        .apply(VoxelDataColumnType::X.as_str(), |x| x + translation.x)
        .expect("TODO: panic message");
    translated_data
        .apply(VoxelDataColumnType::Y.as_str(), |y| y + translation.y)
        .expect("TODO: panic message");
    translated_data
        .apply(VoxelDataColumnType::Z.as_str(), |z| z + translation.z)
        .expect("TODO: panic message");

    let info = voxel_grid.info().clone();
    let frames = voxel_grid.reference_frames().clone();

    VoxelGrid::from_data_frame(translated_data, info, frames).unwrap()
}
