# Ocular Region Reconstruction Based on FLAME

3D scan can bu used to generates high quality 3D data. However, the scan result for eyes, including eyeballs and eyebrow usually has large distortion. This code use [FLAME](http://flame.is.tue.mpg.de/) repository to reconstruct ocular region where 3D scan data is unreliable based on ocular region excluding the distorted parts. 

## Setup:
(forked from FLAME:https://github.com/Rubikplayer/flame-fitting/blob/master/README.md)

We also provide [Tensorflow FLAME](https://github.com/TimoBolkart/TF_FLAME) and [PyTorch FLAME](https://github.com/HavenFeng/photometric_optimization) frameworks, and code to [convert from Basel Face Model to FLAME](https://github.com/TimoBolkart/BFM_to_FLAME).

<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. FLAME combines a linear identity shape space (trained from head scans of 3800 subjects) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)

```
Learning a model of facial shape and expression from 4D scans
Tianye Li*, Timo Bolkart*, Michael J. Black, Hao Li, and Javier Romero
ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 2017
```
and the [supplementary video](https://youtu.be/36rPTkhiJTM).

This codebase demonstrates
 * **Sampling:** Load and evaluate FLAME model for random parameters
 * **Landmark fitting:** Fit FLAME to 3D landmarks
 * **Scan fitting:** Fit FLAME to a 3D scan

<p align="center"> 
<img src="gifs/fitting_scan.gif" width="60%">
</p>

### Set-up

The code has been tested with Python 3.6.9.

Clone the git project:
```
git clone https://github.com/Rubikplayer/flame-fitting.git
```

Install pip and virtualenv

```
sudo apt-get install python3-pip python3-venv
```

Set up virtual environment:
```
mkdir <your_home_dir>/.virtualenvs
python3 -m venv <your_home_dir>/.virtualenvs/flame-fitting
```

Activate virtual environment:
```
cd flame-fitting
source <your_home_dir>/.virtualenvs/flame-fitting/bin/activate
```

Make sure your pip version is up-to-date:
```
pip install -U pip
```

Somerequirements can be installed using:
```
pip install -r requirements.txt
```

Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh) within the virtual environment.

The scan-to-mesh distance used for fitting a scan depends on Eigen. Either download Eigen for [here](http://eigen.tuxfamily.org/index.php?title=Main_Page) OR clone the repository:
```
git clone https://gitlab.com/libeigen/eigen.git
```
After downloading Eigen, you need to compile the code in the directory 'sbody/alignment/mesh_distance'. To do this go to the directory:
```
cd sbody/alignment/mesh_distance
```
Edit the file setup.py to set EIGEN_DIR to the location of Eigen. Then type:
```
make
```

### Data

Download the FLAME model [MPI-IS/FLAME](https://flame.is.tue.mpg.de/downloads). You need to sign up and agree to the model license for access to the model.
Download RCHigh_trimmed_ocular_region.obj, RCHigh_trimmed_color.ply, RCHigh_trimmed.obj from pan.baidu.com following the link bellow. Put them into ./data file
链接：https://pan.baidu.com/s/1d5sR7DP6q0VptLLclcImDA 
提取码：8en9 
复制这段内容后打开百度网盘手机App，操作更方便哦
Note: data for this project must be preprocessed because we use dlib to automatically annotate landmarks on projection of 3D face mesh. The figure in scan must face the +z axis. You can check the projection.png to see whether the scan orient in a wrong direction or not. The two .obj files should contain only vertices (no color). The .pkl file is for generating landmarks, so, it must contain color. For all three scans, the resolution of the scan should be in "cm"

### Demo

 * you can run example.py, which will generate sample_index.npy, landmarks_3d_51_points.npy and reconstruction output fit_region_scan_result.obj under ./output 
 * you can run find_sampler_indexes.py to generate sample_index.npy under the ./data file
 * you can run generate_landmarks.py to generate landmarks_3d_51_points.npy under ./data file
 * after the two file has been generated, you can run fit_scan_region.py to generate fit the region specified by sample_index.npy.

Fitting a scan requires scan and FLAME model to be in the same local coordiante systems. The `fit_scan.py` script provides different options by specifying the variable `scale_unit` to convert from Meters [m] (default), Centimeters [cm], or Milimieters [mm]. Please specify the right unit when running `fit_scan.py`. If the unit of the measurement unit is unknown, choose `scale_unit = 'NA'`.

<p align="center"> 
<img src="output/snapshot03.png" width="50%">
</p>

<p align="center"> 
<img src="output/snapshot07.png" width="50%">
</p>

<p align="center"> 
<img src="output/snapshot08.png" width="50%">
</p>

<p align="center"> 
<img src="output/snapshot09.png" width="50%">
</p>

### Landmarks

<p align="center"> 
<img src="data/landmarks_51_annotated.png" width="50%">
</p>

We project 3D scan into a 2D image and use dlib to automatically annotate landmarks on the projected image. Then, we traceback to the closest point in the scan data and mark it as a landmark.

### Citing

When using this code in a scientific publication, please cite FLAME 
```
@article{FLAME:SiggraphAsia2017,
  title = {Learning a model of facial shape and expression from {4D} scans},
  author = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  year = {2017},
  url = {https://doi.org/10.1145/3130800.3130813}
}
```

### License

The FLAME model is under a Creative Commons Attribution license. By using this code, you acknowledge that you have read the terms and conditions (https://flame.is.tue.mpg.de/modellicense), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code. You further agree to cite the FLAME paper when reporting results with this model.

### Supported projects

FLAME supports several projects such as
* [CoMA: Convolutional Mesh Autoencoders](https://github.com/anuragranj/coma)
* [RingNet: 3D Face Shape and Expression Reconstruction from an Image without 3D Supervision](https://github.com/soubhiksanyal/RingNet)
* [VOCA: Voice Operated Character Animation](https://github.com/TimoBolkart/voca)
* [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://github.com/vchoutas/smplify-x)
* [ExPose: Monocular Expressive Body Regression through Body-Driven Attention](https://github.com/vchoutas/expose)
* [GIF: Generative Interpretable Faces](https://github.com/ParthaEth/GIF)
* [DECA: Detailed Expression Capture and Animation](https://github.com/YadiraF/DECA)

FLAME is part of [SMPL-X: : A new joint 3D model of the human body, face and hands together](https://github.com/vchoutas/smplx)


### Acknowledgement

Code in `smpl_webuser` originates from [SMPL Python code](http://smpl.is.tue.mpg.de/), and code in `sbody` originates from [SMALR](https://github.com/silviazuffi/smalr_online). We thank the authors for pushing these code packages. 
