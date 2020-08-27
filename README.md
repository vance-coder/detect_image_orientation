# detect_image_orientation
detect image orientation with CNN

### 使用CNN检测图像方向
为了提高OCR中文字检测模型的准确率，想着训练一个模型来检测文字方向，然后根据判断的方向先提前把图像给调正了。

因为我这边都是一些票据、账单之类的图片（类似images文件夹下的图片），而且文字检测模型也有一定的角度适应能力，所以这里只是对四个角度0,90,180,270做检测。

在我的训练集下训练好的model可以从这里下载（我的验证集下accuracy是95%）： https://www.yun.cn/s/4120dc888b5d454ba2197ec4705f04e7

### 环境说明：
- Python - 3.6
- tensorflow - 1.14.0
- Keras - 2.3.1
- jupyterlab - 2.1.0


### 快速使用预训练模型（也可参考demo notebook文件）

```
import numpy as np
from glob import glob
from keras.preprocessing import image
from keras.models import load_model
from IPython.display import display

model = load_model('vgg19-256-512.h5')


def predict(img_path):
    angle = {
        0: 0, 
        1: 180, 
        2: 270, 
        3: 90
    }

    valid_image = image.load_img(img_path, target_size=(224, 224))
    image_arr = image.img_to_array(valid_image)
    idx = np.argmax(model.predict(np.expand_dims(image_arr, 0)),axis=1)
    print(f'{img_path}, orientation: {angle[idx[0]]}')
    display(image.load_img(img_path, target_size=(300, 300)))

for p in glob('./images/*.png'):
    predict(p)
```

### 效果展示

![result1](./images/result/result1.png)

![result2](./images/result/result2.png)

![result3](./images/result/result3.png)


### 训练自己的model
因为我的数据集是针对发票、账单类图片，在其他场景图片下泛化能力较弱，或者你想设置多种角度，那么模型就需要重新训练，可以参考detect_orientation notebook文件


由于我的图片数据及客户隐私，所以无法共享给大家，但我自己也从百度、必应下载了部分图片做训练集补充，这部分图片在：
https://aistudio.baidu.com/aistudio/datasetdetail/51231

这部分图片是爬虫的方式下载下来的，爬虫代码在：prepare_image.ipynb

训练集比较少，但是我们可以通过程序进行特定角度旋转来扩展我们的训练集（验证集也一样），另外训练的时候再用上keras中的数据增强功能进一步提升训练集数量。


