use crate::data_frame_utils;
use crate::error::Error;
use crate::info::VoxelGridInfo;
use chrono::{DateTime, Utc};
use ecoord::{FrameId, ReferenceFrames, TransformId};
use nalgebra::Point3;
use polars::datatypes::PlSmallStr;
use polars::prelude::DataFrame;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    pub(crate) voxel_data: DataFrame,
    pub(crate) info: VoxelGridInfo,
    pub(crate) reference_frames: ReferenceFrames,
}

impl VoxelGrid {
    pub fn new(
        voxel_data: DataFrame,
        info: VoxelGridInfo,
        reference_frames: ReferenceFrames,
    ) -> Result<Self, Error> {
        data_frame_utils::check_data_integrity(&voxel_data, &info, &reference_frames)?;

        Ok(Self {
            voxel_data,
            info,
            reference_frames,
        })
    }

    pub fn from_data_frame(
        voxel_data: DataFrame,
        info: VoxelGridInfo,
        reference_frames: ReferenceFrames,
    ) -> Result<Self, Error> {
        /*assert!(
            frames.contains_frame(&info.frame_id),
            "Reference frames must contain frame id '{}' of point cloud data.",
            info.frame_id
        );*/

        data_frame_utils::check_data_integrity(&voxel_data, &info, &reference_frames)?;
        Ok(Self {
            voxel_data,
            info,
            reference_frames,
        })
    }

    pub fn voxel_data(&self) -> &DataFrame {
        &self.voxel_data
    }

    pub fn info(&self) -> &VoxelGridInfo {
        &self.info
    }
    pub fn reference_frames(&self) -> &ReferenceFrames {
        &self.reference_frames
    }
    pub fn set_reference_frames(&mut self, reference_frames: ReferenceFrames) {
        self.reference_frames = reference_frames;
    }

    pub fn size(&self) -> usize {
        self.voxel_data.height()
    }

    pub fn min_index(&self) -> Point3<i64> {
        let index_x: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::X.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .min()
            .unwrap()
            .unwrap();
        let index_y: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Y.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .min()
            .unwrap()
            .unwrap();
        let index_z: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Z.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .min()
            .unwrap()
            .unwrap();

        Point3::new(index_x, index_y, index_z)
    }

    pub fn min_local_center_point(&self) -> Point3<f64> {
        let min_index = self.min_index();
        Point3::new(min_index.x as f64, min_index.y as f64, min_index.z as f64)
            * self.info.resolution
    }

    pub fn min_center_point(&self, frame_id: FrameId) -> Result<Point3<f64>, Error> {
        let min_point = self.min_local_center_point();
        let isometry_graph = self.reference_frames.derive_transform_graph(&None, &None)?;
        let transform_id = TransformId::new(frame_id, self.info.frame_id.clone());
        let isometry = isometry_graph.get_isometry(&transform_id)?;

        Ok(isometry * min_point)
    }

    pub fn max_index(&self) -> Point3<i64> {
        let index_x: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::X.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .max()
            .unwrap()
            .unwrap();
        let index_y: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Y.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .max()
            .unwrap()
            .unwrap();
        let index_z: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Z.as_str())
            .unwrap()
            .as_series()
            .unwrap()
            .max()
            .unwrap()
            .unwrap();

        Point3::new(index_x, index_y, index_z)
    }

    pub fn max_local_center_point(&self) -> Point3<f64> {
        let max_index = self.max_index();
        Point3::new(max_index.x as f64, max_index.y as f64, max_index.z as f64)
            * self.info.resolution
    }

    pub fn max_center_point(&self, frame_id: FrameId) -> Result<Point3<f64>, Error> {
        let max_point = self.max_local_center_point();
        let isometry_graph = self.reference_frames.derive_transform_graph(&None, &None)?;
        let transform_id = TransformId::new(frame_id, self.info.frame_id.clone());
        let isometry = isometry_graph.get_isometry(&transform_id)?;

        Ok(isometry * max_point)
    }

    /// Returns all cell indices as a vector in the local coordinate frame.
    ///
    ///
    pub fn get_all_cell_indices_in_local_frame(&self) -> Vec<Point3<i64>> {
        let x_series = self
            .voxel_data
            .column(VoxelDataColumnType::X.as_str())
            .unwrap()
            .i64()
            .unwrap();
        let y_series = self
            .voxel_data
            .column(VoxelDataColumnType::Y.as_str())
            .unwrap()
            .i64()
            .unwrap();
        let z_series = self
            .voxel_data
            .column(VoxelDataColumnType::Z.as_str())
            .unwrap()
            .i64()
            .unwrap();

        let all_indices: Vec<Point3<i64>> = (0..self.size())
            .into_par_iter()
            .map(|i: usize| {
                Point3::new(
                    x_series.get(i).unwrap(),
                    y_series.get(i).unwrap(),
                    z_series.get(i).unwrap(),
                )
            })
            .collect();

        all_indices
    }

    pub fn get_all_center_points_in_local_frame(&self) -> Vec<Point3<f64>> {
        let all_indices = self.get_all_cell_indices_in_local_frame();

        all_indices
            .par_iter()
            .map(|c| {
                Point3::new(
                    self.info.resolution * c.x as f64,
                    self.info.resolution * c.y as f64,
                    self.info.resolution * c.z as f64,
                )
            })
            .collect()
    }

    pub fn get_cell_index(&self, row_index: usize) -> Point3<i64> {
        let index_x: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::X.as_str())
            .unwrap()
            .i64()
            .unwrap()
            .get(row_index)
            .unwrap();
        let index_y: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Y.as_str())
            .unwrap()
            .i64()
            .unwrap()
            .get(row_index)
            .unwrap();
        let index_z: i64 = self
            .voxel_data
            .column(VoxelDataColumnType::Z.as_str())
            .unwrap()
            .i64()
            .unwrap()
            .get(row_index)
            .unwrap();

        Point3::new(index_x, index_y, index_z)
    }

    /// Returns the frame id of the voxel grid coordinates
    ///
    pub fn get_local_frame_id(&self) -> FrameId {
        //let t: &Transformation = self.frames.transforms().get();
        self.info.frame_id.clone()
    }

    pub fn get_local_center_point(&self, row_idx: usize) -> Point3<f64> {
        let index: Point3<i64> = self.get_cell_index(row_idx);
        let x: f64 = self.info.resolution * index.x as f64;
        let y: f64 = self.info.resolution * index.y as f64;
        let z: f64 = self.info.resolution * index.z as f64;

        Point3::new(x, y, z)
    }

    pub fn get_center_point(
        &self,
        row_idx: usize,
        frame_id: &FrameId,
        timestamp: DateTime<Utc>,
    ) -> Result<Point3<f64>, Error> {
        let local_center_point = self.get_local_center_point(row_idx);

        let transform_id = TransformId::new(frame_id.clone(), self.get_local_frame_id());

        let isometry = self
            .reference_frames
            .derive_transform_graph(&None, &Some(timestamp))?
            .get_isometry(&transform_id)?;

        // let isometry = self.pose.isometry();
        Ok(isometry * local_center_point)
    }

    pub fn get_all_center_points(
        &self,
        frame_id: &FrameId,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<Point3<f64>>, Error> {
        let center_points: Vec<Point3<f64>> = (0..self.size() as i64)
            .map(|i: i64| self.get_center_point(i as usize, frame_id, timestamp))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(center_points)
    }
}

const COLUMN_NAME_X_STR: &str = "x";
const COLUMN_NAME_Y_STR: &str = "y";
const COLUMN_NAME_Z_STR: &str = "z";
const COLUMN_NAME_COUNT_STR: &str = "count";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VoxelDataColumnType {
    /// X index (mandatory)
    X,
    /// Y index (mandatory)
    Y,
    /// Z index (mandatory)
    Z,
    /// Count
    Count,
}

impl VoxelDataColumnType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VoxelDataColumnType::X => COLUMN_NAME_X_STR,
            VoxelDataColumnType::Y => COLUMN_NAME_Y_STR,
            VoxelDataColumnType::Z => COLUMN_NAME_Z_STR,
            VoxelDataColumnType::Count => COLUMN_NAME_COUNT_STR,
        }
    }
}

impl From<VoxelDataColumnType> for PlSmallStr {
    fn from(value: VoxelDataColumnType) -> Self {
        value.as_str().into()
    }
}
