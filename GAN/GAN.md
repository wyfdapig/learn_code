## GAN的设计初衷

一句话来概括 GAN 的设计动机就是——自动化。

**人工提取特征——>自动提取特征**

![传统机器学习和深度学习的核心区别](https://easyai.tech/wp-content/uploads/2022/08/959e6-2019-06-06-butong.png)

**人工判断生成结果的好坏——自动判断和优化**

而 GAN 能自动完成这个过程，且不断的优化，这是一种效率非常高，且成本很低的方式。 



## GAN 的基本原理

==核心公式：MinMax博弈==

![image-20240912110720857](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/image-20240912110720857.png)

D是判别器，G是生成器



GAN由2个重要的部分构成：

1. **生成器(Generator**)：通过机器生成数据（大部分情况下是图像），目的是“骗过”判别器
2. **判别器(Discriminator**)：判断这张图像是真实的还是机器生成的，目的是找出生成器做的“假数据”

![生成对抗网络GANs由生成器和判别器构成](https://easyai.tech/wp-content/uploads/2022/08/4d3f8-2019-07-16-2bf-1.png)

下面详细介绍一下过程：

### 第一阶段：固定「判别器D」，训练「生成器G」

我们使用一个还 OK 的判别器，让一个「生成器G」不断生成“假数据”，然后给这个「判别器D」去判断。

一开始，「生成器G」还很弱，所以很容易被揪出来。

但是随着不断的训练，「生成器G」技能不断提升，最终骗过了「判别器D」。

到了这个时候，「判别器D」基本属于瞎猜的状态，因为判断是否为假数据的概率为50%。

![固定判别器，训练生成器](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/8f496-2019-07-16-g-tg.png)

### 第二阶段：固定「生成器G」，训练「判别器D」

当通过了第一阶段，继续训练「生成器G」就没有意义了。这个时候我们固定「生成器G」，然后开始训练「判别器D」。

「判别器D」通过不断训练，提高了自己的鉴别能力，最终他可以准确的判断出所有的假图片。

到了这个时候，「生成器G」已经无法骗过「判别器D」。

![固定生成器，训练判别器](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/e3628-2019-07-16-d-tg-1.png)

### 循环阶段一和阶段二

通过不断的循环，「生成器G」和「判别器D」的能力都越来越强。

最终我们得到了一个效果非常好的「生成器G」，我们就可以用它来生成我们想要的图片了。

![循环训练，2遍越来越强](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/ebb3c-2019-07-16-xh.png)

 

## GAN的优缺点

**3个优势**

1. 能更好建模数据分布（图像更锐利、清晰）
2. 理论上，GANs 能训练任何一种生成器网络。其他的框架需要生成器网络有一些特定的函数形式，比如输出层是高斯的。
3. 无需利用马尔科夫链反复采样，无需在学习过程中进行推断。

**2个缺陷**

1. 难训练，不稳定。生成器和判别器之间需要很好的同步，但是在实际训练中很容易D收敛，G发散。D/G 的训练需要精心的设计。
2. 模式缺失（Mode Collapse）问题。GANs的学习过程可能出现模式缺失，生成器开始退化，总是生成同样的样本点，无法继续学习。

 

## 10大典型的GAN算法

GAN 算法有数百种之多，大家对于 GAN 的研究呈指数级的上涨，目前每个月都有数百篇论坛是关于对抗网络的。

下图是每个月关于 GAN 的论文发表数量：

![关于GANs的论文呈指数级增长](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/6984a-2019-07-16-paper.png)

如果你对 GANs 算法感兴趣，可以在 「[GANs动物园](https://github.com/hindupuravinash/the-gan-zoo)」里查看几乎所有的算法。我们为大家从众多算法中挑选了10个比较有代表性的算法，技术人员可以看看他的论文和代码。

| 算法     | 论文                                           | 代码                                                         |
| :------- | :--------------------------------------------- | :----------------------------------------------------------- |
| GAN      | [论文地址](https://arxiv.org/abs/1406.2661)    | [代码地址](https://github.com/goodfeli/adversarial)          |
| DCGAN    | [论文地址](https://arxiv.org/abs/1511.06434)   | [代码地址](https://github.com/floydhub/dcgan)                |
| CGAN     | [论文地址](https://arxiv.org/abs/1411.1784)    | [代码地址](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras) |
| CycleGAN | [论文地址](https://arxiv.org/abs/1703.10593v6) | [代码地址](https://github.com/junyanz/CycleGAN)              |
| CoGAN    | [论文地址](https://arxiv.org/abs/1606.07536)   | [代码地址](https://github.com/mingyuliutw/CoGAN)             |
| ProGAN   | [论文地址](https://arxiv.org/abs/1710.10196)   | [代码地址](https://github.com/tkarras/progressive_growing_of_gans) |
| WGAN     | [论文地址](https://arxiv.org/abs/1701.07875v3) | [代码地址](https://github.com/eriklindernoren/Keras-GAN)     |
| SAGAN    | [论文地址](https://arxiv.org/abs/1805.08318v1) | [代码地址](https://github.com/heykeetae/Self-Attention-GAN)  |
| BigGAN   | [论文地址](https://arxiv.org/abs/1809.11096v2) | [代码地址](https://github.com/huggingface/pytorch-pretrained-BigGAN) |



## GAN 的13种实际应用

GAN 看上去不如「语音识别」「文本挖掘」那么直观。不过他的应用已经进入到我们的生活中了。下面给大家列举一些 GAN 的实际应用。

### **生成图像数据集**

人工智能的训练是需要大量的数据集的，如果全部靠人工收集和标注，成本是很高的。GAN 可以自动的生成一些数据集，提供低成本的训练数据。

![GANs生成人脸的矢量算法案例](https://easyai.tech/wp-content/uploads/2022/08/a4ce6-2019-07-10-gans-dataset.png)

### **生成人脸照片**

生成人脸照片是大家很熟悉的应用，但是生成出来的照片用来做什么是需要思考的问题。因为这种人脸照片还处于法律的边缘。

![2014年至2017年GANs能力进展的实例](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/3f5fd-2019-07-10-gans-renlian.png)

### **生成照片、漫画人物**

GAN 不但能生成人脸，还能生成其他类型的照片，甚至是漫画人物。

![GANs生成的照片](https://easyai.tech/wp-content/uploads/2022/08/4a82e-2019-07-10-gans-pic.png)

![GANs生成的漫画人物](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/3dda2-2019-07-10-gans-manhua.png)

### **图像到图像的转换**

简单说就是把一种形式的图像转换成另外一种形式的图像，就好像加滤镜一样神奇。例如：

- 把草稿转换成照片
- 把卫星照片转换为Google地图的图片
- 把照片转换成油画
- 把白天转换成黑夜

![用pix2pix从草图到彩色照片的示例](https://easyai.tech/wp-content/uploads/2022/08/b5137-2019-07-10-gans-caogao.png)

![GANs应用-照片到油画、马到斑马、冬天到夏天、照片到google地图](https://easyai.tech/wp-content/uploads/2022/08/86064-2019-07-10-piczh.png)

###  **文字到图像的转换**

在2016年标题为“ [StackGAN：使用 StackGAN 的文本到逼真照片的图像合成](https://arxiv.org/abs/1612.03242) ”的论文中，演示了使用 GAN，特别是他们的 StackGAN，从鸟类和花卉等简单对象的文本描述中生成逼真的照片。

![从StackGAN获取鸟类的文本描述和GAN生成照片的示例](https://easyai.tech/wp-content/uploads/2022/08/b33c5-2019-07-10-word-pic.png)

###  **语意 – 图像 – 照片 的转换**

在2017年标题为“ [高分辨率图像合成和带条件GAN的语义操纵](https://arxiv.org/abs/1711.11585) ”的论文中，演示了在语义图像或草图作为输入的情况下使用条件GAN生成逼真图像。

![语义图像和GAN生成的城市景观照片的示例](https://easyai.tech/wp-content/uploads/2022/08/c672a-2019-07-10-yuyi-pic.png)

###  **自动生成模特**

在2017年标题为“ [姿势引导人形象生成](https://arxiv.org/abs/1705.09368) ”的论文中，可以自动生成人体模特，并且使用新的姿势。

![GAN生成了新的模特姿势](https://easyai.tech/wp-content/uploads/2022/08/5463c-2019-07-10-mote.png)

###  **照片到Emojis**

GANs 可以通过人脸照片自动生成对应的表情（Emojis）。

![名人照片和GAN生成的表情符号示例](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/39d18-2019-07-10-emojis.png)

###  **照片编辑**

使用GAN可以生成特定的照片，例如更换头发颜色、更改面部表情、甚至是改变性别。

![使用IcGAN编辑照片的效果](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/890dd-2019-07-10-bianji.png)

###  **预测不同年龄的长相**

给一张人脸照片， GAN 就可以帮你预测不同年龄阶段你会长成什么样。

![用具有不同表观年龄的GAN生成的面部照片的示例](https://easyai.tech/wp-content/uploads/2022/08/ec901-2019-07-10-nianling.png)

###  **提高照片分辨率，让照片更清晰**

给GAN一张照片，他就能生成一张分辨率更高的照片，使得这个照片更加清晰。

![GANs在原始照片的基础上增加分辨率，使照片更清晰](https://easyai.tech/wp-content/uploads/2022/08/7a815-2019-07-10-qingxi.png)

###  **照片修复**

假如照片中有一个区域出现了问题（例如被涂上颜色或者被抹去），GAN可以修复这个区域，还原成原始的状态。

![遮住照片中间的一部分，GANs可以很好的修复](https://gitee.com/nie-shiqin/typora_pic/raw/master/img/27250-2019-07-10-xiufu-1.png)

###  **自动生成3D模型**

给出多个不同角度的2D图像，就可以生成一个3D模型。

![从2D图像到3D椅子模型的建立过程](https://easyai.tech/wp-content/uploads/2022/08/8a96d-2019-07-10-3d.png)