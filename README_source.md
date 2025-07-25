# YOLO Series TensorRT Python/C++ 

## Support
[YOLOv11, YOLOV12](https://docs.ultralytics.com/)、[YOLOv10](https://github.com/THU-MIG/yolov10)、[YOLOv9](https://github.com/WongKinYiu/yolov9)、[YOLOv8](https://v8docs.ultralytics.com/)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv6](https://github.com/meituan/YOLOv6)、 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOv3](https://github.com/ultralytics/yolov3)

- [x] YOLOv12
- [x] YOLOv11
- [x] YOLOv10
- [x] YOLOv9
- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 

## Update
- 2025.07.07 重新工程化
- 2025.03.14 Support YOLOv12
- 2024.11.24 Support YOLOv11, fix the bug causing YOLOv8 accuracy misalignment 
- 2024.06.16 Support YOLOv9, YOLOv10, changing the TensorRT version to 10.0 
- 2023.08.15 Support cuda-python
- 2023.05.12 Update
- 2023.01.07 support YOLOv8
- 2022.11.29 fix some bug thanks @[JiaPai12138](https://github.com/JiaPai12138)
- 2022.08.13 rename reop、 public new version、 **C++ for end2end**
- 2022.08.11 nms plugin support ==> Now you can set --end2end flag while use `export.py` get a engine file  
- 2022.07.08 support YOLOv7 
- 2022.07.03 support TRT int8  post-training quantization 

##  Prepare TRT Env 
`Install via Python`
```
pip install tensorrt
pip install cuda-python
```
`Install via  C++`

[By Docker](https://github.com/NVIDIA/TensorRT/blob/main/docker/ubuntu-20.04.Dockerfile)

## YOLO12
### Export ONNX
```shell
pip install ultralytics
```

```Python
from ultralytics import YOLO
model = YOLO("yolo12n.pt")
model.export(format='onnx')
```

### Generate TRT File 
```shell
python export.py  -o yolo112n.onnx -e yolo12n.trt --end2end --v8 -p fp32
```
### Inference 
```shell
python trt.py -e yolo12n.trt  -i src/1.jpg -o yolo12-1.jpg --end2end 
```


## YOLO11
### Export ONNX
```shell
pip install ultralytics
```

```Python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.export(format='onnx')
```

### Generate TRT File 
```shell
python export.py  -o yolo11n.onnx -e yolov11n.trt --end2end --v8 -p fp32
```
### Inference 
```shell
python trt.py -e yolov11n.trt  -i src/1.jpg -o yolov11-1.jpg --end2end 
```

## YOLOv10
### Generate TRT File 
```shell
python export.py  -o yolov10n.onnx -e yolov10.trt --end2end --v10 -p fp32
```
### Inference 
```shell
python trt.py -e yolov10.trt  -i src/1.jpg -o yolov10-1.jpg --end2end 
```

## YOLOv9
### Generate TRT File 
```shell
python export.py  -o yolov9-c.onnx -e yolov9.trt --end2end --v8 -p fp32
```
### Inference 
```shell
python trt.py -e yolov9.trt  -i src/1.jpg -o yolov9-1.jpg --end2end 
```

## Python Demo
<details><summary> <b>Expand</b> </summary>

1. [YOLOv5](##YOLOv5)
2. [YOLOx](##YOLOX)
3. [YOLOv6](##YOLOV6)
4. [YOLOv7](##YOLOv7)
5. [YOLOv8](##YOLOv8)

## YOLOv8

### Install && Download [Weights](https://github.com/ultralytics/assets/)
```shell
pip install ultralytics
```
### Export ONNX
```Python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.fuse()  
model.info(verbose=False)  # Print model information
model.export(format='onnx')  # TODO: 
```
### Generate TRT File 
```shell
python export.py -o yolov8n.onnx -e yolov8n.trt --end2end --v8 --fp32
```
### Inference 
```shell
python trt.py -e yolov8n.trt  -i src/1.jpg -o yolov8n-1.jpg --end2end 
```


## YOLOv5


```python
!git clone https://github.com/ultralytics/yolov5.git
```

```python
!wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```


```python
!python yolov5/export.py --weights yolov5n.pt --include onnx --simplify --inplace 
```

### include  NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt --end2end
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt 
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg 
```

## YOLOX 


```python
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```


```python
!wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```


```python
!cd YOLOX && pip3 install -v -e . --user
```


```python
!cd YOLOX && python tools/export_onnx.py --output-name ../yolox_s.onnx -n yolox-s -c ../yolox_s.pth --decode_in_inference
```

### include  NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt --end2end
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt 
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg 
```

## YOLOv6 


```python
!wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx
```

### include  NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt --end2end
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt 
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg 
```

## YOLOv7


```python
!git clone https://github.com/WongKinYiu/yolov7.git
```


```python
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```


```python
!pip install -r yolov7/requirements.txt
```


```python
!python yolov7/export.py --weights yolov7-tiny.pt --grid  --simplify
```

### include  NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny.trt --end2end
```


```python
!python trt.py -e yolov7-tiny.trt  -i src/1.jpg -o yolov7-tiny-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny-norm.trt
```


```python
!python trt.py -e yolov7-tiny-norm.trt  -i src/1.jpg -o yolov7-tiny-norm-1.jpg
```
</details>

### C++ Demo

support **NMS plugin**
show in [C++ Demo](cpp/README.MD)


## Citing 

If you use this repo in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{yolotrt2022,
  author =       {Jian Lin},
  title =        {YOLOTRT: tensorrt for yolo series},
  howpublished = {\url{[https://github.com/Linaom1214/TensorRT-For-YOLO-Series]}},
  year =         {2022}
}
```
# YOLO-Series-TRT

https://github.com/Linaom1214/TensorRT-For-YOLO-Series
