# Denset121-SGD-restarts-Whale
Whale baby\
First implementation of kernel was taken from https://www.kaggle.com/matthewa313/resnet50 . Model Architecture is changed in this kernel This kernel uses Stochastic Gradient Descent with restarts to train model, it performed better than resnet101 trained without restarts for same number of epochs\
```
!pip install fastai==0.7.0 --no-deps
!pip install torch==0.4.1 torchvision==0.2.1
!pip install torchtext==0.2.3
```
fastai0.7.0应当是与pytorch0.3.1对应，这里用的是0.4.1会报不兼容的错误，但似乎不影响计算？\
```
df = pd.read_csv(LABELS).set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
print("Number of classes: {}".format(len(unique_labels)))
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]
```
这里采用字典的模式来将标签转化为数值，前面有见到enumerate做转化的，似乎比这种模式更为精巧。
```
class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['Id']

    def get_c(self):
        return len(unique_labels)
```
这里我们分析定义的DataSet集合，首先super()函数表示调用父类的方法，因此这里调用了FilesDataset的初始化方法，而FilesDataset的初始化方法又调用了BaseDataset的初始化方法，path指的是图片所在的位置，get_x()方法获取图片并且按照图片中最大的那一张图片进行resize，get_y()方法用于获取图片对应的标签，若数据集为测试集，则返回0，get_c()方法用于返回训练集中有的类别。
```
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b, self.c = b, c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  # add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1 / (c - 1) if c < 0 else c + 1
        x = lighting(x, b, c)
        return x
```
这个类用于数据增强中的明暗度调节，在父类中定义了一个全局变量，用于保存各线程中的特定变量，这里应该是后面需要用多线程处理图像，为了避免对每一个图像做的处理相同。
```
def get_data(sz, batch_size):
    """
    Read data and do augmentations
    """
    aug_tfms = [RandomRotateZoom(deg=20, zoom=2, stretch=1),
                RandomLighting(0.2, 0.2, tfm_y=TfmType.NO),
                RandomBlur(blur_strengths=3,tfm_y=TfmType.NO),
                RandomFlip(tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HWIDataset, (tr_n[:-(len(tr_n) % batch_size)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, TEST))
    md = ImageData("./", ds, batch_size, num_workers=num_workers, classes=None)
    return md
```
这里定义了四种数据增强的操作，第一种包括旋转，放大，拉伸，三个变量分别代表最大的幅度，第二种是调节明暗度，第三种模糊度，第四种随机翻转

```
image_size = 224
batch_size = 64
md = get_data(image_size, batch_size)
extra_fc_layers_size = []
learn = ConvLearner.pretrained(arch, md, xtra_fc=extra_fc_layers_size) 
learn.opt_fn = optim.Adam
```
这里设定图片大小为（224，224），extra_fc_layers代表的是在经典架构的顶层设置的输出结构，由于至少需要一层来将前面网络的输出转化成已知的输出，因此当设置xtra_fc为空集时，其代表的是通过一层将前面网络的输出转化为类别维度的输出。

```
base_lr = 5e-4 # lr for the backbone
fc_lr = 1e-3 # lr for the classifer

lrs = [base_lr, base_lr, fc_lr]
# Freeze backbone and train the classifier for 2 epochs
learn.fit(lrs=lrs, n_cycle=2, cycle_len=None)

# Unfreeze backbone and continue training for 9 epochs
learn.unfreeze()
learn.fit(lrs, n_cycle=3, cycle_len=1, cycle_mult=2)
learn.save('weights')
```
这里的学习率由find_lr找到，此kernel中直接对网络的前面结构进行unfreeze然后进行fine tuning、实际上这不是一个好的做法，因为最后一层由我们人为在顶端加上的层，其权重是随机生成的，而前面的网络权重则已经相对训练的较好了，混合的这样训练并不是一个好的事情。\

在训练中发现train loss要显著大于valid loss，这是由于在实际的计算过程中训练过程采取了drop out操作，而相应的valid过程则没有，一般而言我们只有可能将drop out比率上调。

```
best_th = 0.38
preds_t,y_t = learn.TTA(is_test=True,n_aug=8)
```
由于在图片的resize过程中一般是center crop的方式，因此在预测的时候通过对预测的图片进行随机的数据增强来提高模型的表现。
