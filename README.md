# Openpilot supercomo model deployment

Using comma.ai pretrained self-driving car model to predict lane lines.
![output](https://user-images.githubusercontent.com/43088163/120663559-cab41180-c492-11eb-940d-c58e9b5983f7.png)




# Installation

Use Python version >= 3.6 
1. Install requirements
```sh
$ pip3 install -r requirements.txt
```
2. Use your own video or download my sample video from [HERE](https://drive.google.com/file/d/10CFyMSEY_w5ZjzWsYClFxYIdpY62PG31/view?usp=sharing).
```sh
$ mkdir data
$ cd data
```
3. Download pre-trained model (onnx) from [comma-ai gituhub](https://github.com/commaai/openpilot/tree/master/models)

4. Run the program

Note: Specify the video feed location
```sh
$ python3 openpilot_onnx.py
```

### Credits

Thank You comma.ai for making your research open source.




## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

## Supercombo
### Supercombo input format (Full size: 799906 x float32)
* **image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **wide image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **desire**
  * one-hot encoded buffer to command model to execute certain actions, bit needs to be sent for the past 5 seconds (at 20FPS) : 100 * 8
* **traffic convention**
  * one-hot encoded vector to tell model whether traffic is right-hand or left-hand traffic : 2
* **feature buffer**
  * A buffer of intermediate features that gets appended to the current feature to form a 5 seconds temporal context (at 20FPS) : 99 * 512


### Supercombo output format (Full size: 6472 x float32)
* **plan**
  * 5 potential desired plan predictions : 4955 = 5 * 991
    * predicted mean and standard deviation of the following values at 33 timesteps : 990 = 2 * 33 * 15
      * x,y,z position in current frame (meters)
      * x,y,z velocity in local frame (meters/s)
      * x,y,z acceleration local frame (meters/(s*s))
      * roll, pitch , yaw in current frame (radians)
      * roll, pitch , yaw rates in local frame (radians/s)
    * probability[^1] of this plan hypothesis being the most likely: 1
* **lanelines**
  * 4 lanelines (outer left, left, right, and outer right): 528 = 4 * 132
    * predicted mean and standard deviation for the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
* **laneline probabilties**
  * 2 probabilities[^1] that each of the 4 lanelines exists : 8 = 4 * 2
    * deprecated probability
    * used probability
* **road-edges**
  * 2 road-edges (left and right): 264 = 2 * 132
    * predicted mean and standard deviation for the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
* **leads**
  * 2 hypotheses for potential lead cars : 102 = 2 * 51
    * predicted mean and stadard deviation for the following values at 0,2,4,6,8,10s : 48 = 2 * 6 * 4
      * x position of lead in current frame (meters)
      * y position of lead in current frame (meters)
      * speed of lead (meters/s)
      * acceleration of lead(meters/(s*s))
    * probabilities[^1] this hypothesis is the most likely hypothesis at 0s, 2s or 4s from now : 3
* **lead probabilities**
  * probability[^1] that there is a lead car at 0s, 2s, 4s from now : 3 = 1 * 3
* **desire state**
  * probability[^1] that the model thinks it is executing each of the 8 potential desire actions : 8
* **meta** [^2]
  * Various metadata about the scene : 80 = 1 + 35 + 12 + 32
    * Probability[^1] that openpilot is engaged : 1
    * Probabilities[^1] of various things happening between now and 2,4,6,8,10s : 35 = 5 * 7
      * Disengage of openpilot with gas pedal
      * Disengage of openpilot with brake pedal
      * Override of openpilot steering
      * 3m/(s*s) of deceleration
      * 4m/(s*s) of deceleration
      * 5m/(s*s) of deceleration
    * Probabilities[^1] of left or right blinker being active at 0,2,4,6,8,10s : 12 = 6 * 2
    * Probabilities[^1] that each of the 8 desires is being executed at 0,2,4,6s : 32 = 4 * 8

* **pose** [^2]
  * predicted mean and standard deviation of current translation and rotation rates : 12 = 2 * 6
    * x,y,z velocity in current frame (meters/s)
    * roll, pitch , yaw rates in current frame (radians/s)
* **recurrent state**
  * The recurrent state vector that is fed back into the GRU for temporal context : 512

[^1]: All probabilities are in logits, so you need to apply sigmoid or softmax functions to get actual probabilities
[^2]: These outputs come directly from the vision blocks, they do not have access to temporal state or the desire input


## Driver Monitoring Model
* .onnx model can be run with onnx runtimes
* .dlc file is a pre-quantized model and only runs on qualcomm DSPs

### input format
* single image W = 1440 H = 960 luminance channel (Y) from the planar YUV420 format:
  * full input size is 1440 * 960 = 1382400
  * normalized ranging from 0.0 to 1.0 in float32 (onnx runner) or ranging from 0 to 255 in uint8 (snpe runner)
* camera calibration angles (roll, pitch, yaw) from liveCalibration: 3 x float32 inputs

### output format
* 84 x float32 outputs = 2 + 41 * 2 ([parsing example](https://github.com/commaai/openpilot/blob/22ce4e17ba0d3bfcf37f8255a4dd1dc683fe0c38/selfdrive/modeld/models/dmonitoring.cc#L33))
  * for each person in the front seats (2 * 41)
    * face pose: 12 = 6 + 6
      * face orientation [pitch, yaw, roll] in camera frame: 3
      * face position [dx, dy] relative to image center: 2
      * normalized face size: 1
      * standard deviations for above outputs: 6
    * face visible probability: 1
    * eyes: 20 = (8 + 1) + (8 + 1) + 1 + 1
      * eye position and size, and their standard deviations: 8
      * eye visible probability: 1
      * eye closed probability: 1
    * wearing sunglasses probability: 1
    * face occluded probability: 1
    * touching wheel probability: 1
    * paying attention probability: 1
    * (deprecated) distracted probabilities: 2
    * using phone probability: 1
    * distracted probability: 1
  * common outputs 2
    * poor camera vision probability: 1
    * left hand drive probability: 1

| Credit | Link |
| ------ | ------ |
| Comma Ai | [https://comma.ai/] |
| GitHub | [https://github.com/commaai] |
| Trained models | [https://github.com/commaai/openpilot/tree/master/models] |


