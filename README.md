# VirEqPhy

Implementation accompanying **Geometric Dynamic Equivalence Method for Digital Twin-Driven Embodied Agent Systems** by Xin Li, Shiyuan Zhang, Yanmin Zhou, Ping Lu, Zhipeng Wang, Rong Jiang, Hongrui Sang, and Bin He. [Project page](https://forbiddenname.github.io/VirEqPhyDT/)

VirEqPhy uses a shared occupancy voxel representation for CAD retrieval and combines category orientation priors with weighted Procrustes refinement for 9-DoF alignment. The optional W-MR module learns CAD rankings from appearance, semantic, and geometric similarities. Given successive RGB observations and camera poses, the demo exports the estimated CAD ID, rotation, translation, scale, and transform for each frame.

This release adapts the available laboratory code and implements algorithms specified by the November 22 manuscript where the available source differs. The examples use synthetic tensors and freshly trained weights. Reproducing the paper's numbers requires the original data splits and preprocessing, trained weights, and benchmark evaluation resources. See [implementation and reproduction status](#implementation-and-reproduction-status).

## Installation

Use Python 3.10–3.12. From the repository root, activate a virtual environment and install the dependencies.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU training, install the corresponding CUDA wheels first. Mesh voxelization also requires `python -m pip install -r requirements-renderer.txt`. [PyTorch wheel options](https://docs.pytorch.org/get-started/previous-versions/)

## Running the demo

The following CPU example generates a tiny voxel catalog, fits orientation priors, takes two supervised optimization steps, and exports one scene snapshot. The synthetic feature tensors exercise the geometry pipeline.

```bash
python -m examples.make_data
python -m network.train supervised --samples outputs/example/supervised.pt --catalog outputs/example/catalog.pt --steps 2 --batch-size 2 --output outputs/example/supervised-model.pt
python -m network.demo --frames outputs/example/frame.pt --catalog outputs/example/catalog.pt --checkpoint outputs/example/supervised-model.pt --output outputs/example/scene.json
```

For RGB inference, use a checkpoint trained with the RGB branch and a catalog with the same voxel resolution and category mapping. Run [Mask Transfiner](https://github.com/SysCV/transfiner) with an appropriate R50-FPN config and detector weights to obtain refined instance masks. The frame preparation script can call the official fork through `--transfiner-config CONFIG --transfiner-weights WEIGHTS` when installed in the same environment. For environments with different dependency versions, export `boxes`, `masks`, and `categories` arrays from Mask Transfiner into an NPZ file using NumPy. Masks are at the original image resolution.

```bash
python -m renderer.prepare_frame --image frame.png --detections detections.npz --intrinsics 577.9 577.9 319.5 239.5 --category-map detector_to_vireqphy.json --output data/frame.pt
python -m network.demo --frames data/frame.pt --catalog data/catalog.pt --checkpoint checkpoints/supervised.pt --output outputs/scene.json
```

Replace the example intrinsics with the camera's calibrated `fx fy cx cy`. The category map is a JSON object from detector IDs to repository category IDs, such as `{"3": 2}` for a detector that assigns category 3 to chairs. Supply a rigid 4×4 camera-to-world matrix through `--camera-to-world` for world-frame output. Pass multiple frame bundles in temporal order for sequential scene snapshots. The script uses independent per-frame updates. A consumer can use supplied `track_ids` in each frame bundle for persistent instance identity. Add `--weak-checkpoint checkpoints/weak.pt` to replace category-restricted supervised retrieval with W-MR over the supplied catalog.

## Preparing data

The code uses tensor dictionaries saved with `torch.save`. Keep datasets, checkpoints, and generated outputs separate from the source tree. Obtain the official [ScanNet](https://github.com/ScanNet/ScanNet), [Scan2CAD](https://github.com/skanti/Scan2CAD), and [ShapeNet](https://shapenet.org/) assets under their respective terms. Section VI-A uses the ScanNet25K split with 19,387 training and 5,436 validation images and Scan2CAD object–CAD associations. The supplemental weak data expansion requires the corresponding sample list and rendered views.

### CAD catalog

Prepare a JSON list of records with a unique string `id`, integer `category`, and mesh path `mesh` relative to the manifest. See [the supervised categories](metadata/categories.json). Use CADs with consistent canonical axes.

```bash
python -m renderer.prepare catalog --manifest data/cads.json --size 32 --output data/catalog.pt
```

The catalog contains `ids`, `categories [C]`, and occupancy `voxels [C,1,D,D,D]`. Spatial tensor axes are `z,y,x`. Coordinates are stored in `x,y,z` order, with voxel centers in the centered unit cube. The preprocessor centers the mesh by its bounds and normalizes by its longest axis. Pose annotations must use the same origin and scale convention. If you have original preprocessed grids, convert them directly to this tensor schema to preserve their voxelization.

### Supervised training and validation bundles

Each training row is one object instance. Put the following tensors into a bundle with the same number of rows.

| Keys | Shape and convention |
| --- | --- |
| `categories`, `cad_indices` | `[N]` zero-based class ID and row of the matched CAD in the supplied catalog |
| `rotation`, `scale`, `translation` | `[N,3,3]`, `[N,3]`, `[N,3]`. Rotations map canonical axes to camera axes. Scale applies along the canonical axes. Translation gives the object centroid in meters. |
| `rgb`, `boxes`, `masks`, `intrinsics` | RGB float `[N,3,H,W]` in `[0,1]`, pixel boxes `[N,4]` in `xyxy` order, refined masks `[N,H,W]`, and camera matrices `[N,3,3]`. Repeat RGB and intrinsics for multiple objects from one image. |
| `depth_target` | `[N,1,H,W]` during RGB training. Nonpositive and nonfinite depth is excluded from BerHu. For `--rgbd`, also provide `measured_depth` with valid meter-valued observations and invalid samples filled with zero. |
| `roi`, `mask`, `depth_xyz` | Alternative feature input. Shapes `[N,256,h,w]`, `[N,1,h,w]`, `[N,3,h,w]` with camera-space points in meters. |
| `target_categories`, `symmetry_order` | Optional validation fields. Use 1 for no symmetry, 2 or 4 for the canonical up-axis symmetries, and 0 for continuous symmetry. |

Fit rotation medoids from training annotations only. Each trainable category needs at least eight rotations. Use contiguous category IDs starting at zero. Validation samples must share the training categories and catalog ID conventions. The demo supports one RGB image with several instances by supplying `image_indices`; see [frame preparation](renderer/prepare_frame.py).

### Weak association labels

Start from object crops and CAD views rendered offline. As described in Section V-A, extract DINOv2 appearance descriptors and CLIP image descriptors. Save the descriptors as float tensors `object_appearance [Q,256]`, `cad_appearance [C,R,256]`, `object_semantic [Q,512]`, and `cad_semantic [C,R,512]` with ordered string lists `query_ids` and `cad_ids`. The manuscript does not specify how its DINOv2 output was reduced to 256 dimensions. To match the experiment, supply the original descriptors or use the same saved projection for query and CAD images. The implementation accepts other descriptor dimensions with a matching width within each modality. For the Section VI-B settings, select the highest-confidence detected object per training image before exporting the sample bundle and use the paper's training-category exclusions for the unseen-category run.

Export pretrained supervised voxel embeddings and construct the weak-label matrices.

```bash
python -m renderer.weak_features --samples data/weak_samples.pt --catalog data/catalog.pt --checkpoint checkpoints/supervised.pt --visual data/visual.pt --queries data/weak_queries.pt --descriptors data/descriptors.pt
python -m renderer.prepare labels --descriptors data/descriptors.pt --output data/weak_labels.pt
```

Here `weak_samples.pt` uses the RGB or feature fields above with ordered `query_ids`. No matched CAD IDs or pose annotations are required for weak training. Appearance and semantic best-view indices are selected independently before concatenation. The raw appearance–semantic and geometric cosines are blended with weights 0.4 and 0.6. The blend and individual components are normalized per object across the catalog.

## Training and evaluation

```bash
python -m network.train supervised --samples data/train.pt --catalog data/catalog.pt --output checkpoints/supervised.pt --device cuda
python -m network.evaluate alignment --samples data/val.pt --catalog data/catalog.pt --checkpoint checkpoints/supervised.pt --device cuda

python -m network.train weak --samples data/weak_queries.pt --labels data/weak_labels.pt --catalog data/catalog.pt --output checkpoints/weak.pt --device cuda
python -m network.evaluate weak --samples data/weak_val.pt --catalog data/catalog.pt --checkpoint checkpoints/weak.pt --device cuda
```

Supervised training defaults to SGD momentum 0.9, learning rate 0.001, batch size 32, and 90,000 iterations. DWA starts at iteration 10,000. Train on catalogs at resolutions 16, 32, and 64 for the three voxel-size configurations. The `--rgbd` flag enables the measured-depth ablation with a supported input bundle. `--cad-candidates` sets the sampled CAD budget per training category before hard-negative mining.

Weak training defaults to K=5 and a negative-to-positive sample ratio of 1. Use `--k 1 --negative-ratio 0` for positive-only Top-1 and `--k 5 --negative-ratio 0` for positive-only Top-5. `--negative-ratio 2` selects all bottom 2K candidates. `--positive-pool` caps the number of high-similarity candidates used for training, before adding negatives. Its default is 32. Use `--positive-pool 0` to include every candidate outside the bottom 2K. For weak evaluation, use a held-out dictionary with `voxels` and `cad_indices` and the full evaluation catalog. The evaluator reports Accu1/5 and IoU1/5. Alignment evaluation reports matched-instance class and instance averages at 20 cm, 20°, and 20% scale error, with a separate retrieval-aware score requiring exact CAD ID.

## Implementation and reproduction status

| Paper | Source and review outcome |
| --- | --- |
| IV-A, MI | [ResNet-50 FPN, multi-scale depth, and masked ROI input](network/features.py). The local ROCA path has coarse mask prediction. This release accepts refined masks from the external Mask Transfiner fork and trains the FPN/depth branch on prepared object instances. The joint detector and refinement loss `L2D` requires the original integration. |
| IV-B, MV and Eq. (3) | [Dual occupancy and coordinate decoder](network/model.py). The local Pix2Vox-derived code decodes occupancy, while the ROCA alignment path uses NOC maps. The release shares four 3-D upsampling layers and uses separate occupancy and observed-coordinate outputs, trained with BCE and coordinate MSE. |
| IV-C, MR and Eq. (4) | [Shared residual 3-D voxel encoder and cosine retrieval](network/model.py). The release feeds predicted occupancies directly into retrieval. It uses category-constrained nearest-neighbor search, hard same-class negatives for NCE and voxel cube occlusion during training. |
| IV-D, MA and Eq. (2) | [Category-wise k-medoids, 9-DoF regression, and weighted rotation](network/alignment.py). The original ROCA alignment path refines rotation from depth/NOC correspondences. The release classifies eight rotation medoids per category and solves a centered, occupancy-weighted rotation on the retrieved CAD grid. |
| IV-E and Eq. (5) | [BerHu](network/features.py), [geometric objectives](network/model.py), and [DWA](network/train.py). The RGB training path weighs four available tasks initially at 0.5, 1, 1, and 2 for depth, MV, MR, and MA. The precomputed-feature path weighs the three geometry tasks. |
| V-A and V-B, W-MR and Eqs. (6)–(10) | [Similarity labels, shared DenseNet-style encoder, and perturbed Top-k](network/weak.py). The local weak-retrieval code uses perceptual distance, masked IoU, and a positive-only objective. The release uses pretrained voxel geometry, independent appearance/semantic view selection, cumulative rank indicators, and penalties on sampled bottom-tail candidates. |
| III and VI-C, scene updating | [CAD poses and scene snapshots](network/demo.py). The optional camera transform maps poses into world space. Original Open3D, RealMan, RealSense, Vicon, and Habitat experiment integration is external to this release. |

The manuscript leaves several implementation choices unspecified. This release pairs CAD and observed coordinates at shared canonical voxel indices, thresholds occupancy at 0.5 and falls back to the coarse rotation when correspondence support is degenerate. The refinement uses `(R0 x) ⊙ s` as written in Eq. (2), whereas the exported TRS matrix applies scale in the canonical frame as described in Section IV-D. For anisotropic scale those two operations generally differ. The weak-label normalization uses row-wise min–max, and perturbation scores use cosine mapped to `[0,1]`. Negative sampling is uniform without replacement from the bottom 2K.

The CPU example checks the executable workflow. Run `python -m unittest discover -s tests -v` to check rotation recovery and gradients, orientation priors, weak-label construction, perturbed Top-k gradients, and the RGB feature path. The matched-instance evaluator uses prepared query/ground-truth pairs. Scene-level benchmark matching, Mesh R-CNN retrieval AP, published baseline ablations, and the paper's tables require their original experiment assets or evaluation tools. The repository contains code and a synthetic generator. The original datasets, weights, rendered views, and robot recordings must be obtained separately.

## Citation

```bibtex
@unpublished{li_vireqphy,
  title = {Geometric Dynamic Equivalence Method for Digital Twin-Driven Embodied Agent Systems},
  author = {Li, Xin and Zhang, Shiyuan and Zhou, Yanmin and Lu, Ping and Wang, Zhipeng and Jiang, Rong and Sang, Hongrui and He, Bin},
  note = {Manuscript}
}
```

## Acknowledgments and licensing

The 3-D residual block and BerHu implementation draw on [ROCA](https://github.com/cangumeli/ROCA). Its original notice is included in [licenses/ROCA.txt](licenses/ROCA.txt). Further method references include [Pix2Vox++](https://gitlab.com/hzxie/Pix2VoxPlusPlus), [Mask Transfiner](https://github.com/SysCV/transfiner), [DINOv2](https://github.com/facebookresearch/dinov2), [CLIP](https://github.com/openai/CLIP), and [Weakly-Supervised End-to-End CAD Retrieval to Scan Objects](https://github.com/TimFelixBeyer/weakly-supervised-cad-retrieval). The weak retrieval code here is written from the manuscript equations.
