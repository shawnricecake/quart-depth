
# Data Preparation

## Datasets

### NYU Depth Dataset V2
Refer to this [repo](https://github.com/tjqansthd/LapDepth-release)

#### For the Metric3D:

1. copy `metric3d/data/nyu_demo/organize_data.py` into `NYU_Depth_V2/official_split/test/`

2. `python3 organize_data.py` to generate `rgb/` and `depth/`

3. revise the path in `metric3d/data/gene_annos_nyu_demo.py` 

4. `python3 gene_annos_nyu_demo.py` to generate new `test_annotations.json`


### SUN RGB-D Dataset
Refer to [repo](https://github.com/ankurhanda/sunrgbd-meta-data?tab=readme-ov-file)

Download test dataset (5050 jpg images) is available from [SUNRGBD-test_images.tgz](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz)

Download test depth dataset (5050 depth images) [test_data_depth](https://www.doc.ic.ac.uk/~ahanda/sunrgb_test_depth.tgz) (550MB)

### IBims-1 Dataset
Download from [here](https://dataserv.ub.tum.de/index.php/s/m1455541)

`wget https://dataserv.ub.tum.de/s/m1455541/download?path=%2F&files=ibims1_core_raw.zip`


### KITTI Dataset

`wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip`

Remove `train/` and `val/` folders in `data_depth_annotated`, make all data together

Refer to this [repo](https://github.com/tjqansthd/LapDepth-release)

#### For the Metric3D:
1. revise the path in `gene_annos_kitti_demo-mine.py`
2. `python3 gene_annos_kitti_demo-mine.py` to generate `test_annotations.json`

### Virtual KITTI Dataset

Download from [here](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/)

RGB images: `wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_rgb.tar`

Depth images: `wget http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_depthgt.tar`

### Hypersim Dataset

Refer to `metric_depth/dataset/splits/hypersim/val.txt` in Depth Anything V2

Run `python3 hypersim_download.py`


## Data Tree

```bazaar
dataset
|-- diode
|   `-- val
|       |-- indoors
|       |   |-- scene_00019
|       |   |   `-- scan_00183
|       |   |-- scene_00020
|       |   |   |-- scan_00184
|       |   |   |-- scan_00185
|       |   |   |-- scan_00186
|       |   |   `-- scan_00187
|       |   `-- scene_00021
|       |       |-- scan_00188
|       |       |-- scan_00189
|       |       |-- scan_00190
|       |       |-- scan_00191
|       |       `-- scan_00192
|       `-- outdoor
|           |-- scene_00022
|           |   |-- scan_00193
|           |   |-- scan_00194
|           |   |-- scan_00195
|           |   |-- scan_00196
|           |   `-- scan_00197
|           |-- scene_00023
|           |   |-- scan_00198
|           |   |-- scan_00199
|           |   `-- scan_00200
|           `-- scene_00024
|               |-- scan_00201
|               `-- scan_00202
|-- ibims
|   `-- ibims1_core_raw
|       |-- calib
|       |-- depth
|       |-- edges
|       |-- mask_floor
|       |-- mask_invalid
|       |-- mask_table
|       |-- mask_transp
|       |-- mask_wall
|       `-- rgb
|-- kitti
|   |-- data_depth_annotated
|   |   |-- train
|   |   |   |-- 2011_09_26_drive_0001_sync
|   |   |   |   `-- proj_depth
|   |   |   |       `-- groundtruth
|   |   |   |           |-- image_02
|   |   |   |           `-- image_03
|   |   |   `-- 2011_10_03_drive_0042_sync
|   |   |       `-- proj_depth
|   |   |           `-- groundtruth
|   |   |               |-- image_02
|   |   |               `-- image_03
|   |   `-- val
|   |       |-- 2011_09_26_drive_0002_sync
|   |       |   `-- proj_depth
|   |       |       `-- groundtruth
|   |       |           |-- image_02
|   |       |           `-- image_03
|   |       `-- 2011_10_03_drive_0047_sync
|   |           `-- proj_depth
|   |               `-- groundtruth
|   |                   |-- image_02
|   |                   `-- image_03
|   `-- raw_data
|       |-- 2011_09_26
|       |   |-- 2011_09_26_drive_0001_sync
|       |   |   |-- image_00
|       |   |   |   `-- data
|       |   |   |-- image_01
|       |   |   |   `-- data
|       |   |   |-- image_02
|       |   |   |   `-- data
|       |   |   |-- image_03
|       |   |   |   `-- data
|       |   |   |-- oxts
|       |   |   |   `-- data
|       |   |   `-- velodyne_points
|       |   |       `-- data
|       `-- 2011_10_03_drive_0047_sync
|           |-- image_00
|           |   `-- data
|           |-- image_01
|           |   `-- data
|           |-- image_02
|           |   `-- data
|           |-- image_03
|           |   `-- data
|           |-- oxts
|           |   `-- data
|           `-- velodyne_points
|               `-- data
|-- nyuv2
|   |-- NYU_Depth_V2
|   |   `-- official_splits
|   |       |-- rgb
|   |       |   |-- rgb_00000.jpg
|   |       |   |-- ...
|   |       |-- depth
|   |       |   |-- sync_depth_00000.png
|   |       |   |-- ...
|   |       |-- test
|   |       |   |-- bathroom
|   |       |   |   `-- dense
|   |       |   |-- bedroom
|   |       |   |   `-- dense
|   |       |   |-- bookstore
|   |       |   |   `-- dense
|   |       |   |-- classroom
|   |       |   |   `-- dense
|   |       |   |-- computer_lab
|   |       |   |   `-- dense
|   |       |   |-- dining_room
|   |       |   |   `-- dense
|   |       |   |-- foyer
|   |       |   |   `-- dense
|   |       |   |-- home_office
|   |       |   |   `-- dense
|   |       |   |-- kitchen
|   |       |   |   `-- dense
|   |       |   |-- living_room
|   |       |   |   `-- dense
|   |       |   |-- office
|   |       |   |   `-- dense
|   |       |   |-- office_kitchen
|   |       |   |   `-- dense
|   |       |   |-- playroom
|   |       |   |   `-- dense
|   |       |   |-- reception_room
|   |       |   |   `-- dense
|   |       |   |-- study
|   |       |   |   `-- dense
|   |       |   `-- study_room
|   |           `-- dense
|-- hypersim
|   |-- ai_003_010
|   |   `-- detail
|   |       |-- cam_00
|   |       |   |-- camera_keyframe_frame_indices.hdf5
|   |       |   |-- ...
|   |       |-- cam_01
|   |       |   |-- camera_keyframe_frame_indices.hdf5
|   |       |   |-- ...
|   |       |-- mesh
|   |       |   |-- metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5
|   |       |   |-- ...
|   |       |-- metadata_cameras.csv
|   |       |-- metadata_node_strings.csv
|   |       |-- metadata_node.csv
|   |       |-- metadata_scene.csv
|   |   `-- images
|   |       |   |-- ...
|   |   `-- scene_cam_00_final_preview
|   |       |   |-- ...
|   |   `-- scene_cam_00_geometry_hdf5
|   |       |   |-- ...
|   |   `-- ...
|-- sunrgbd
|   |-- depth
|   `-- rgb
`-- vkitti2
    |-- depth
    |   |-- Scene01
    |   |   |-- 15-deg-left
    |   |   |   `-- frames
    |   |   |       `-- depth
    |   |   |           |-- Camera_0
    |   |   |           `-- Camera_1
    |   `-- sunset
    |       `-- frames
    |           `-- depth
    |               |-- Camera_0
    |               `-- Camera_1
    |-- Scene02
    |-- Scene20
    `-- rgb
        |-- Scene01
        |   |-- 15-deg-left
        |   |   `-- frames
        |   |       `-- rgb
        |   |           |-- Camera_0
        |   |           `-- Camera_1
        |   `-- sunset
        |       `-- frames
        |           `-- rgb
        |               |-- Camera_0
        |               `-- Camera_1
        |-- Scene02
        `-- Scene20

```




