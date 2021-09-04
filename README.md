# DeepCS-AL

Experimental environment: 
TensorFlow 1.14 + Python3.7
### Datasets
Tiny ImageNet consists of 30 categories randomly selected from original ImageNet, ILSVRC 2012 database. In the experiments, 30 categories in the field of WNIDs include n02808304, n02326432, n03781244, n03950228, n07860988, n02974003, n03991062, n02447366, n03065424, n03967562, n02018795, n02119022, n07836838, n04522168, n02488702, n02971356, n04141975, n04465501, n01667114, n02437616, n04389033, n01498041, n07871810, n02699494, n02172182, n01601694, n02167151, n03461385, n03216828, n04536866, and the corresponding labels are 908, 128, 685, 982, 863, 562, 837, 15, 697, 380, 431, 61, 952, 873, 53, 748, 520, 288, 459, 185, 249, 445, 805, 676, 625, 395, 622, 702, 714, 342, respectively. In tiny ImageNet, there are 38766 training images and 1500 testing images in total. The names and labels of 1500 test images can be checked at datasets\imagenet\testData\validation_30.txt, and 1500 original images are stored at datasets\imagenet\testData\test.mat.
Note that all images in Cifar-10 and tiny ImageNet are performed gray processing. We use the gray images to train our adversarial models, so that the training process is consistent with those presented in Refs. relating to CSNet and BCSNet for a fair comparison.

### Model
The folder Model\Matrix stores five sampling matrixes pre-trained with CSNet at the sampling rates of 0.02, 0.05, 0.1, 0.15, and 0.2.
The folder Model\Recognizer stores the pre-trained recognition networks VGG, ResNet, and DenseNet with Cifar-10 and tiny ImageNet databases, respectively. 

### AT-CSNet-Ck
The folder AT-CSNet-Ck stores the AT-CSNet-VGG models separately trained with Cifar-10 and tiny ImageNet. By running AT_CSNet-Ck\AT_CSNet.py, the performance scores about AT-CSNet-VGG, shown in TABLE V can be achieved. One can easily get the performance scores with AT-CSNet-ResNet and AT-CSNet-DenseNet by only taking a small change to AT_CSNet\AT_CSNet.py. Here we omit the models of AT-CSNet-ResNet and AT-CSNet-DenseNet due to the excessive storage space. 

### AT-BCSNet-Ck
Similar to the folder AT-CSNet-Ck, the trained AT-BCSNet-VGG models are stored in folder AT-BCSNet-Ck.

### AT-G
These trained AT-CSNet and AT-BCSNet model are stored in folder AT-G. The corresponding source python codes are AT-CSNet.py and AT-BCSNet.py.

Any questions, please contact us at swzhou@hnu.edu.cn.
