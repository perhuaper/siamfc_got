### SiamFC

modified from (https://github.com/got-10k/siamfc).
>add test-and-record while training

>add multiprocess to test.py

>modified show_image() to visualize

>support changing backbone to vgg (siamvgg)

### Dependencies

install pytorch, opencv-python:

```bash
pip install torch opencv-python
```

### Train

1. Download pretrained `model_siamfc.pth` from [Baidu Yun](https://pan.baidu.com/s/1VoV79ZLmCq4J6NZLin8qcA).
sqrb 

2. Download pretrained `model_siamvgg.pth` from [Baidu Yun](链接：https://pan.baidu.com/s/1vskjVbyaZrKQVZJm0zSa3g).
vskn

3. Download GOT10k dataset and OTB2015 dataset.

4. Change your datasets folder dir and model dir (in `test.py`):

5. Run:

```
python test.py --tracker siamfc
```
```
python test.py --tracker siamvgg
```
### Train

1. Change dir to your datasets folder (in end of `train.py`):

2. Run:

```
python train.py --tracker siamfc
```
```
python train.py --tracker siamvgg
```
By default, tracker will train in GOT10k and test over OTB2015 dataset.

#### SiamFC Results in OTB 2015
| epoch| Success Score | Precision Score |FPS|
|:----:|:----:|:----:|:----:|
|  5 | 0.590 | 0.791 |120+|
| 10 | 0.595 | 0.802 |
| 15 | 0.596 | 0.805 |
| 20 | 0.574 | 0.770 |
| 25 | 0.580 | 0.785 |
| 30 | 0.567 | 0.759 |
| 35 | 0.564 | 0.757 |
| 40 | 0.533 | 0.710 |
| 45 | 0.544 | 0.726 |
| 50 | 0.554 | 0.742 |

#### SiamVGG Results in OTB 2015
| epoch| Success Score | Precision Score |FPS|
|:----:|:----:|:----:|:----:|
| 1 | 0.583 | 0.771 |60+|
| 2 | 0.609 | 0.803 |
| 3 | 0.601 | 0.805 |
| 4 | 0.620 | 0.823 |
| 5 | 0.608 | 0.799 |
| 6 | 0.616 | 0.827 |
| 7 | 0.602 | 0.804 |
| 8 | 0.615 | 0.815 |
| 9 | 0.594 | 0.788 |
| 10 | 0.585 | 0.775 |

####combine epoch15 of SiamFC and epoch4 of SiamVGG (only test)
| | Success Score | Precision Score |FPS|
|:----:|:----:|:----:|:----:|
|test|0.633|0.834|40+|
