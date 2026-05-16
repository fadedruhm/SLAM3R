# Tips for running SLAM3R on self-captured data


## Image name format

Your images should be consecutive frames (e.g., sampled from a video)with numbered filenames to indicate their sequential order, such as `frame-0031.color.png, output_0414.jpg, ...` with zero padding. 


## Guidance on parameters in the [script](../scripts/demo_wild.sh)

`KEYFRAME_STRIDE`: The selection of keyframe stride is crucial. A small stride may lack sufficient camera motion, while a large stride can cause insufficient overlap, both of which hinder the performance. We offer an automatic setting option, but you can also configure it manually. 

`CONF_THRES_I2P`: It helps to filter out points predicted by the Image-to-Points model before input to the Local-to-World model. Consider increasing this threshold if there are many areas in your image with unpredictable depth (e.g., sky or black-masked regions).

`INITIAL_WINSIZE`: The maximum window size for initializing scene reconstruction. Note that if initial confidence scores are too low, the window size will automatically reduce, with a minimum size of 3.

`BUFFER_SIZE`: The maximum size of the buffering set. The buffer size should be large enough to store the required scene frames, but an excessively large buffer can impede retrieval efficiency and degrade retrieval performance.

`BUFFER_STRATEGY`: We provide two strategies for maintaining the scene frame buffering set under a limited buffer size. "reservoir" is suitable for single-room scenarios, while "fifo" performs more effectively on larger scenes. As discussed in our paper, SLAM3R suffers from drift issues in very large scenes. 


## Failure cases

SLAM3R's performance can degrade when processing sparse views with large viewpoint changes, low overlaps, or motion blur. In such cases, the default window size may include images that don't overlap enough, leading to reduced performance. This can also lead to incorrect retrievals, causing some frames to be registered outside the main reconstruction.

Due to limited training data (currently only ScanNet++, Aria Synthetic Environments, and CO3D-v2), SLAM3R cannot process images with unfamiliar camera distortion and has poor performance with wide-angle cameras and panoramas. The system also struggles with dynamic scenes. 


## Cesium visualization tip (glTF axis correction)

When visualizing `recon.glb` and a camera trajectory (e.g., points from `scene_traj.txt`) in CesiumJS, you may see a confusing situation:

- The model looks oriented “correctly”, but the plotted trajectory points are far away / rotated / drifting.

In practice, this is often *not* a SLAM or export bug. It is commonly caused by Cesium applying an internal **glTF axis correction** during model loading (glTF is typically Y-up, while Cesium internally uses a different convention). If your points are plotted without applying the same correction, the model and points will not share the same transform chain.

Recommendations:

- Ensure the **same transform chain** is applied to both the model and the points (axis correction + user rotations + modelMatrix/anchor).
- Avoid relying on private Cesium fields (e.g., `_axisCorrectionMatrix`), as they may not exist or may change across versions.
- Avoid double-compensating (e.g., applying both a manual `X=-90°` *and* an axis correction that already covers the same rotation).
