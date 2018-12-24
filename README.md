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
