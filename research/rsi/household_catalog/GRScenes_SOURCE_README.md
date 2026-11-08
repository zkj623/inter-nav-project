---
license: cc-by-nc-sa-4.0
---

## 🏙️ GRScenes Data

![diverse scenes](./scene.png)

### ✨ Features

1. **Large-Scale & Dynamic & Realistic**: GRScenes is built from over 100k high-quality scene prototypes and now includes 99 environments (69 home scenes and 30 commercial scenes). Each scene has been carefully designed and accurately modeled to ensure realism.

2. **Fine-grained Interactive Objects**: All the objects have full internal modeling, enabling robots to perform actions such as opening doors or sliding drawers in a realistic way. These assets facilitate fine-grained manipulation tasks and include part-level annotations in Omniverse X-form level.

3. **Rich Semantic Information**: We set object-level semantic labels in each scene, which support tasks such as local navigation and object searching.

### 📁 Directory Structure

```
GRScenes-100/commercial_scenes.zip --(unzip)--> target_30_new
└── ...
GRScenes-100/home_scenes.zip --(unzip)--> target_69_new
├── Materials
│   └── ... (material mdl files and texture pictures)
├── models
│   ├── layout
│   │   ├── articulated
│   │   │   └── ... ( window, door, etc.)
│   │   └── others
│   │       └── ... (ceiling, wall, ground, etc.)
│   └── object
│       ├── articulated
│       │   └── ... (microwave, refrigerator, etc.)
│       └── others
│           └── ... (bed, bottle, cup, etc.)
└── scenes
    ├── MV7J6NIKTKJZ2AABAAAAADA8_usd
    │   ├── Materials -> ../../Materials
    │   ├── models    -> ../../models
    │   ├── metadata.json (records the referenced model and material paths)
    │   └── start_result_xxx.usd (scene usd files)
    └── ... (other scene folders)
```

- **Materials** folder contains mdl files and texture pictures. The mdl files, which are Material Definition Language files commonly used by rendering engines such as NVIDIA Omniverse. These mdl files are used with texture pictures to define the physically based material properties such as color, reflectivity, and transparency that can be applied to 3D objects.

- **models** folder contains 3D object models, where layouts objects under `layout/` and interactive objects under `object/`. Subdirectories are further categorized according to the model semantic labels such as `door` and `oven`.

- **scenes** folder (e.g., `MV7J6NIKTKJZ2AABAAAAADA8_usd/`) contains the following files:
  - **Scene USD Files**

      We provides three usd files.
      - **raw scene**, named as `start_result_raw.usd`, which defines the layout of the scene.
      - **navigation scene**, named as `start_result_navigation.usd`, which used for navigation tasks.
      - **interaction scene**, named as `start_result_interaction.usd`, which used for manipulation tasks.

  - **metadata.json**

      This file records the metadata information of the models and materials referenced in the raw scene.

  - **interactive_obj_list.json**

      This file records the prim paths of the interactive objects in the interaction scene.

## 📚 Getting Started

### Prerequisites

**Configure MDL Material Search Path**

If loading scenes in Isaac Sim, we recommend to configure an environment variable named `MDL_SYSTEM_PATH` according to this [document](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/mdl_search_path.html). Here are the steps:

```bash
# step 1. Find the Materials folder path, such as `./target_69_new/Materials`
# step 2. Configure the environment variable `MDL_SYSTEM_PATH` (saved to `~/.bashrc` is recommended).
echo 'export MDL_SYSTEM_PATH=$MDL_SYSTEM_PATH:</path/to/your_downloaded_materials_folder>' >> ~/.bashrc
source ~/.bashrc
```

**Install Dependencies**

Our tool scripts depends on OpenUSD and IsaacSim Python SDK, users need to install these dependencies as follows:

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install usd-core==24.11
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
```

### Usage

We provide some [scripts](https://github.com/OpenRobotLab/GRUtopia/tree/main/toolkits/grscenes_scripts) for GRScenes.

- **preprocess.py** is used to bind physics properties (rigid body, collider etc.) with objects in one or several scenes.

```bash
## use `-i/--interaction` option to preprocess scenes for interaction.
python preprocess.py -i/--interaction -f/--files [</path/to/raw_scene_usd_file>...]
## use `-n/--navigation` option to preprocess scenes for navigation
python preprocess.py -n/--navigation -f/--files [</path/to/raw_scene_usd_file>...]
## besides, use `-d/--dirs` option to preprocess all scenes under the scenes folder such as `/ssd/$USER/target_69_new/scenes`
python preprocess.py -i/--interaction -n/--navigation -d/--dirs [</path/to/scene_root_folder>...]
```

- **warmup.py** is used to warmup the simulation process of the given scenes.

```bash
## warmup the specific scenes
python warmup.py -f/--files [</path/to/scene_usd_file>...]
## warmup all scenes
python warmup.py -d/--dirs [</path/to/scene_root_folder>...]
```

- **play_scene.py** is used to load and play the given scene.
```shell
python play_scene.py -f/--file </path/to/scene_usd_file>
```

- **export_scenes.py** is used to export the specified one or more scenes with its related objects and material files.
```bash
python export_scenes.py -i/--input </path/to/source_scene_root_folder> -o/--output </path/to/target_scene_root_folder> -n/--names [<scene_id1>...]
```

- **get_metadata.py** is used to get the metadata information of the models and materials referenced in given model instance or scene usd files.
```bash
python get_metadata.py -f/--files [</path/to/single_instance_or_scene_usd>...]
python get_metadata.py -d/--dirs [</path/to/instance_or_scene_root_folder>...]
```

- **extract_objaverse.py** is used to extract model objects from objaverse. It can convert the data type of 3D models from glb format to usd format, and detach the raw models and its materials.
```bash
python extract_objaverse.py --usd_path </path/to/objavers_usd_file> --material_path </path/to/output_material_files_path>
```

</details>

## 📄 License

GRUtopia's simulation platform is [MIT licensed](LICENSE). The open-sourced GRScenes are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
