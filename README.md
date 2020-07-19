# Computer Pointer Controller

The Computer Pointer Controller main fucntionaly is moving the computer mouse using gazes of a person, to acheive that four deep learnnig models used: face detection model, landmarks estimation, head pose estimation and finally the gaze estimation.

## Project Set Up and Installation




The project is organized in folders, the `src` folder is for the source code, the `bin` folder is for videos and images to use a input, the `models` folder is for the IR models
used in the project.

The application is developed and test using openvino 2020.4, in order to install the toolkit in linux refer to the [official documentataion](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html).

In order to get started with the application, you should first download the necessary models:


```
> source /opt/intel/openvino/bin/setupvars.sh

> cd /opt/intel/openvino/deployment_tools/tools/model_downloader/

> sudo pip3 install -r requirements.in

> sudo python3 downloader.py --name face-detection-adas-binary-0001

> sudo python3 downloader.py --name landmarks-regression-retail-0009

> sudo python3 downloader.py --name head-pose-estimation-adas-0001

> sudo python3 downloader.py --name gaze-estimation-adas-0002

```

After downloading the needed IR models for the execution of the application, yo are ready to install the application dependencies:

```
> sudo pip3 install -r requirements.txt
```

## Demo

The following shows a demonstartion about the application:

![Demo](img/demo.png)

The following the link to visualize the video:

https://youtu.be/1SN6S4XHxig


## Documentation

In order to run the application, you can keep it simple and take the arguments as default by using the below command, in this case the models used are `FP32`, the visualization of the models outputs is activated and the `CPU` is used as the device for all the models:

```
> python3 src/main.py -i bin/demo.mp4  
```

If you want to know more you could use `--help` to get all the possible options:

```
usage: main.py [-h] [-i INPUT] [-m_fd MODEL_FD] [-m_ld MODEL_LD]
               [-m_hpe MODEL_HPE] [-m_ge MODEL_GE] [-d_fd {CPU,GPU,FPGA,VPU}]
               [-d_ld {CPU,GPU,FPGA,VPU}] [-d_hpe {CPU,GPU,FPGA,VPU}]
               [-d_ge {CPU,GPU,FPGA,VPU}] [-e_fd EXT_FD] [-e_ld EXT_LD]
               [-e_hpe EXT_HPE] [-e_ge EXT_GE] [-v_fd] [-v_ld] [-v_hpe]
               [-v_ge]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input video
  -m_fd MODEL_FD, --model_fd MODEL_FD
                        Face detection model name path
  -m_ld MODEL_LD, --model_ld MODEL_LD
                        Landmarks detection model name path
  -m_hpe MODEL_HPE, --model_hpe MODEL_HPE
                        Head pose estimation model name path
  -m_ge MODEL_GE, --model_ge MODEL_GE
                        Gaze estimation model name path
  -d_fd {CPU,GPU,FPGA,VPU}, --device_fd {CPU,GPU,FPGA,VPU}
                        Face detection device
  -d_ld {CPU,GPU,FPGA,VPU}, --device_ld {CPU,GPU,FPGA,VPU}
                        Landmarks detection device
  -d_hpe {CPU,GPU,FPGA,VPU}, --device_hpe {CPU,GPU,FPGA,VPU}
                        Head pose estimation device
  -d_ge {CPU,GPU,FPGA,VPU}, --device_ge {CPU,GPU,FPGA,VPU}
                        Gaze estimation device
  -e_fd EXT_FD, --ext_fd EXT_FD
                        Face detection model extension
  -e_ld EXT_LD, --ext_ld EXT_LD
                        Landmarks detection model extension
  -e_hpe EXT_HPE, --ext_hpe EXT_HPE
                        Head pose estimation model extension
  -e_ge EXT_GE, --ext_ge EXT_GE
                        Gaze estimation model extension
  -v_fd, --vis_fd       Face detection visualization
  -v_ld, --vis_ld       Landmarks detection visualization
  -v_hpe, --vis_hpe     Head pose estimation visualization
  -v_ge, --vis_ge       Gaze estimation visualization

```

Documentation of the used models:


* [Face Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

* [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

* [Landmarks Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Benchmarks

The below results collected in an Ubuntu 18.04 64 bits virtual machine running in VirtualBox with 4GB of RAM and 2 CPU cores.


Inference time with preprocessing of input and output for each model depending on the precision of the model using CPU:


| Model                         |     FP32      |   FP16    |  FP16-INT8 |
|-------------------------------|---------------|-----------|------------|
|Face detection(FP32-INT1)      |    0.0557ms   | 0.0667ms  | 0.0526ms   |
|Landmarks estimation           |    0.0025ms   | 0.0021ms  | 0.0030ms   |
|Head pose estimation           |    0.0043ms   | 0.0037ms  | 0.0030ms   |
|Gaze estimation                |    0.0042ms   | 0.0068ms  | 0.0035ms   |





Loading time for each model depending on the precision of the model using the CPU:

| Model                         |     FP32      |   FP16     |  FP16-INT8  |
|-------------------------------|---------------|------------|-------------|
|Face detection(FP32-INT1)      |  0.1822 ms    | 0.1811 ms  |  0.1903 ms  |
|Landmarks estimation           |  0.0737 ms    | 0.0924 ms  |  0.1101 ms  |
|Head pose estimation           |  0.0813 ms    | 0.1162 ms  |  0.2133 ms  |
|Gaze estimation                |  0.1069 ms    | 0.1317 ms  |  0.2138 ms  |




Note: For face detection the model available used is of precision FP32-INT1 in all the cases, as this is the only available model.

## Results

We notice the models with low precisions generally tend to give better latency, but it still difficult to give an exact measures as the time spent depend of the performance of the machine used in that given that when running the application.  Also we notice that there isn't a big difference between the same model with different precisions.

The models with low precisions are more lightweight than the models with high precisons, so this makes the exexution of the network more fast. 

As the above collected results shows that the models with low precisons take much time to load than models with higher precisons with a difference that could reach 0.1 ms.

## Stand Out Suggestions
Coming...

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases

The application is designed to handle an input video with multiple persons and will still function as expected, the application is getting the information from all the persons and it uses the gaze of the first person to mouve the mouse.

The application is also designed for robust and safe failing, even if some detections are missed in frames, this will not cause an issue, but it keeps going untill the end. so even if there is an issue caused by lighting, it won't cause the application for working.