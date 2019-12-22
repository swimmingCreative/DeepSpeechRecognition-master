# 基于深度学习的中文语音识别系统


[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.7+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

如果觉得有用的话，小手给个star吧~

## 注意：本人于近期想对该项目进行翻新，tf现在已经将keras作为重要的一部分，因此可能将代码用TensorFlow2来进行修改。大家有什么建议可以在issue提一下。
## Note: I want to refurbish the project in the near future. tf now has keras as an important part, so the code may be modified with TensorFlow2. Any suggestions for everyone can be mentioned in the issue.

## 1. Introduction
该系统实现了基于深度框架的语音识别中的声学模型和语言模型建模，其中声学模型包括CNN-CTC、GRU-CTC、CNN-RNN-CTC，语言模型包含[transformer](https://jalammar.github.io/illustrated-transformer/)、[CBHG](https://github.com/crownpku/Somiao-Pinyin)，数据集包含stc、primewords、Aishell、thchs30四个数据集。

本系统更整体介绍：https://blog.csdn.net/chinatelecom08/article/details/82557715

本项目现已训练一个迷你的语音识别系统，将项目下载到本地上，下载[thchs数据集](http://www.openslr.org/resources/18/data_thchs30.tgz)并解压至data，运行`test.py`，不出意外能够进行识别，结果如下：

     the  0 th example.
    文本结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文汉字： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然
    识别结果： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然

若自己建立模型则需要删除现有模型，重新配置参数训练，具体实现流程参考本页最后。

## 2. 声学模型

声学模型采用CTC进行建模，采用CNN-CTC、GRU-CTC、FSMN等模型`model_speech`，采用keras作为编写框架。

- 论文地址：http://www.infocomm-journal.com/dxkx/CN/article/openArticlePDFabs.jsp?id=166970

- tutorial：https://blog.csdn.net/chinatelecom08/article/details/85013535


## 3. 语言模型

新增基于self-attention结构的语言模型`model_language\transformer.py`，该模型已经被证明有强于其他框架的语言表达能力。

- 论文地址：https://arxiv.org/abs/1706.03762。

- tutorial：https://blog.csdn.net/chinatelecom08/article/details/85051817

基于CBHG结构的语言模型`model_language\cbhg.py`，该模型之前用于谷歌声音合成，移植到该项目中作为基于神经网络的语言模型。

- 原理地址：https://github.com/crownpku/Somiao-Pinyin

- tutorial：https://blog.csdn.net/chinatelecom08/article/details/85048019


## 4. 数据集
包括stc、primewords、Aishell、thchs30四个数据集，共计约430小时, 相关链接：[http://www.openslr.org/resources.php](http://www.openslr.org/resources.php)


|Name | train | dev | test
|- | :-: | -: | -:
|aishell | 120098| 14326 | 7176
|primewords | 40783 | 5046 | 5073
|thchs-30 | 10000 | 893 | 2495
|st-cmd | 10000 | 600 | 2000


数据标签整理在`data`路径下，其中primewords、st-cmd目前未区分训练集测试集。

若需要使用所有数据集，只需解压到统一路径下，然后设置utils.py中datapath的路径即可。

与数据相关参数在`utils.py`中：
- data_type: train, test, dev
- data_path: 对应解压数据的路径
- thchs30, aishell, prime, stcmd: 是否使用该数据集
- batch_size: batch_size
- data_length: 我自己做实验时写小一些看效果用的，正常使用设为None即可
- shuffle：正常训练设为True，是否打乱训练顺序
```py
def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type = 'train',
        data_path = 'data/',
        thchs30 = True,
        aishell = True,
        prime = False,
        stcmd = False,
        batch_size = 1,
        data_length = None,
        shuffle = False)
      return params
```

## 5. 配置

使用train.py文件进行模型的训练。

声学模型可选cnn-ctc、gru-ctc，只需修改导入路径即可：

`from model_speech.cnn_ctc import Am, am_hparams`

`from model_speech.gru_ctc import Am, am_hparams`

语言模型可选transformer和cbhg:

`from model_language.transformer import Lm, lm_hparams`

`from model_language.cbhg import Lm, lm_hparams`

### 模型识别

使用test.py检查模型识别效果。
模型选择需和训练一致。


# 一个简单的例子


# 1. 声学模型训练

train.py文件


```python
import os
import tensorflow as tf
from utils import get_data, data_hparams


# 准备训练所需数据
data_args = data_hparams()
data_args.data_length = 10
train_data = get_data(data_args)


# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

epochs = 10
batch_num = len(train_data.wav_lst) // train_data.batch_size

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    #shuffle(shuffle_list)
    batch = train_data.get_am_batch()
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)

am.ctc_model.save_weights('logs_am/model.h5')

```

    get source list...
    load  thchs_train.txt  data...
    100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 236865.96it/s]
    load  aishell_train.txt  data...
    100%|██████████████████████████████████████████████████████████████████████| 120098/120098 [00:00<00:00, 260863.15it/s]
    make am vocab...
    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9986.44it/s]  
    make lm pinyin vocab...
    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9946.18it/s]
    make lm hanzi vocab...
    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9950.90it/s]
    Using TensorFlow backend.


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    the_inputs (InputLayer)      (None, None, 200, 1)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, None, 200, 32)     320       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, None, 200, 32)     128       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, None, 200, 32)     9248      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, None, 200, 32)     128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, None, 100, 32)     0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, None, 100, 64)     18496     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, None, 100, 64)     256       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, None, 100, 64)     36928     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, None, 100, 64)     256       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, None, 50, 64)      0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, None, 50, 128)     73856     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, None, 50, 128)     512       
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, None, 50, 128)     147584    
    _________________________________________________________________
    batch_normalization_6 (Batch (None, None, 50, 128)     512       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, None, 25, 128)     0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_7 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_8 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_9 (Batch (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    reshape_1 (Reshape)          (None, None, 3200)        0         
    _________________________________________________________________
    dense_1 (Dense)              (None, None, 256)         819456    
    _________________________________________________________________
    dense_2 (Dense)              (None, None, 230)         59110     
    =================================================================
    Total params: 1,759,174
    Trainable params: 1,757,254
    Non-trainable params: 1,920
    _________________________________________________________________
    load acoustic model...


# 2.语言模型训练


```python
# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams


lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm = Lm(lm_args)

epochs = 10
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    if os.path.exists('logs_lm/model.meta'):
        print('loading language model...')
        saver.restore(sess, 'logs_lm/model')
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        if (k+1) % 5 == 0:
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model')
    writer.close()
```

    loading language model...
    INFO:tensorflow:Restoring parameters from logs_lm/model


# 3. 模型测试
整合声学模型和语言模型

test.py文件


## 定义解码器


```python
import os
import tensorflow as tf
import numpy as np
from keras import backend as K

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text
```

## 准备测试数据


```python
# 0. 准备解码所需字典，需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_length = 10 # 重新训练需要注释该行
train_data = get_data(data_args)


# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。
data_args.data_type = 'train'
test_data = get_data(data_args)
am_batch = test_data.get_am_batch()
lm_batch = test_data.get_lm_batch()
```

    get source list...
    load  thchs_train.txt  data...
    100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 226097.06it/s]
    load  aishell_train.txt  data...
    100%|██████████████████████████████████████████████████████████████████████| 120098/120098 [00:00<00:00, 226827.96it/s]
    make am vocab...
    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9950.90it/s]
    make lm pinyin vocab...
    100%|██████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<?, ?it/s]
    make lm hanzi vocab...
    100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9953.26it/s]


## 加载声学模型和语言模型


```python
# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('logs_am/model.h5')

# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():
    saver.restore(sess, 'logs_lm/model')
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    the_inputs (InputLayer)      (None, None, 200, 1)      0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, None, 200, 32)     320       
    _________________________________________________________________
    batch_normalization_11 (Batc (None, None, 200, 32)     128       
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, None, 200, 32)     9248      
    _________________________________________________________________
    batch_normalization_12 (Batc (None, None, 200, 32)     128       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, None, 100, 32)     0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, None, 100, 64)     18496     
    _________________________________________________________________
    batch_normalization_13 (Batc (None, None, 100, 64)     256       
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, None, 100, 64)     36928     
    _________________________________________________________________
    batch_normalization_14 (Batc (None, None, 100, 64)     256       
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, None, 50, 64)      0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, None, 50, 128)     73856     
    _________________________________________________________________
    batch_normalization_15 (Batc (None, None, 50, 128)     512       
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, None, 50, 128)     147584    
    _________________________________________________________________
    batch_normalization_16 (Batc (None, None, 50, 128)     512       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, None, 25, 128)     0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_17 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_18 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_19 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, None, 25, 128)     147584    
    _________________________________________________________________
    batch_normalization_20 (Batc (None, None, 25, 128)     512       
    _________________________________________________________________
    reshape_2 (Reshape)          (None, None, 3200)        0         
    _________________________________________________________________
    dense_3 (Dense)              (None, None, 256)         819456    
    _________________________________________________________________
    dense_4 (Dense)              (None, None, 230)         59110     
    =================================================================
    Total params: 1,759,174
    Trainable params: 1,757,254
    Non-trainable params: 1,920
    _________________________________________________________________
    loading acoustic model...
    loading language model...
    INFO:tensorflow:Restoring parameters from logs_lm/model


## 使用语音识别系统


```python

for i in range(5):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, outputs = next(am_batch)
    x = inputs['the_inputs']
    y = inputs['the_labels'][0]
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, train_data.am_vocab)
    text = ' '.join(text)
    print('文本结果：', text)
    print('原文结果：', ' '.join([train_data.am_vocab[int(i)] for i in y]))
    with sess.as_default():
        _, y = next(lm_batch)
        text = text.strip('\n').split(' ')
        x = np.array([train_data.pny_vocab.index(pny) for pny in text])
        x = x.reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: x})
        got = ''.join(train_data.han_vocab[idx] for idx in preds[0])
        print('原文汉字：', ''.join(train_data.han_vocab[idx] for idx in y[0]))
        print('识别结果：', got)
sess.close()
```


     the  0 th example.
    文本结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文结果： lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de di3 se4 si4 yue4 de lin2 luan2 geng4 shi4 lv4 de2 xian1 huo2 xiu4 mei4 shi1 yi4 ang4 ran2
    原文汉字： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然
    识别结果： 绿是阳春烟景大块文章的底色四月的林峦更是绿得鲜活秀媚诗意盎然

     the  1 th example.
    文本结果： ta1 jin3 ping2 yao1 bu4 de li4 liang4 zai4 yong3 dao4 shang4 xia4 fan1 teng2 yong3 dong4 she2 xing2 zhuang4 ru2 hai3 tun2 yi4 zhi2 yi3 yi1 tou2 de you1 shi4 ling3 xian1
    原文结果： ta1 jin3 ping2 yao1 bu4 de li4 liang4 zai4 yong3 dao4 shang4 xia4 fan1 teng2 yong3 dong4 she2 xing2 zhuang4 ru2 hai3 tun2 yi4 zhi2 yi3 yi1 tou2 de you1 shi4 ling3 xian1
    原文汉字： 他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先
    识别结果： 他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先

     the  2 th example.
    文本结果： pao4 yan3 da3 hao3 le zha4 yao4 zen3 me zhuang1 yue4 zheng4 cai2 yao3 le yao3 ya2 shu1 di4 tuo1 qu4 yi1 fu2 guang1 bang3 zi chong1 jin4 le shui3 cuan4 dong4
    原文结果： pao4 yan3 da3 hao3 le zha4 yao4 zen3 me zhuang1 yue4 zheng4 cai2 yao3 le yao3 ya2 shu1 di4 tuo1 qu4 yi1 fu2 guang1 bang3 zi chong1 jin4 le shui3 cuan4 dong4
    原文汉字： 炮眼打好了炸药怎么装岳正才咬了咬牙倏地脱去衣服光膀子冲进了水窜洞
    识别结果： 炮眼打好了炸药怎么装岳正才咬了咬牙倏地脱去衣服光膀子冲进了水窜洞

     the  3 th example.
    文本结果： ke3 shei2 zhi1 wen2 wan2 hou4 ta1 yi1 zhao4 jing4 zi zhi1 jian4 zuo3 xia4 yan3 jian3 de xian4 you4 cu1 you4 hei1 yu3 you4 ce4 ming2 xian3 bu2 dui4 cheng1
    原文结果： ke3 shei2 zhi1 wen2 wan2 hou4 ta1 yi1 zhao4 jing4 zi zhi1 jian4 zuo3 xia4 yan3 jian3 de xian4 you4 cu1 you4 hei1 yu3 you4 ce4 ming2 xian3 bu2 dui4 cheng1
    原文汉字： 可谁知纹完后她一照镜子只见左下眼睑的线又粗又黑与右侧明显不对称
    识别结果： 可谁知纹完后她一照镜子知见左下眼睑的线右粗右黑与右侧明显不对称

     the  4 th example.
    文本结果： yi1 jin4 men2 wo3 bei4 jing1 dai1 le zhe4 hu4 ming2 jiao4 pang2 ji2 de lao3 nong2 shi4 kang4 mei3 yuan2 chao2 fu4 shang1 hui2 xiang1 de lao3 bing1 qi1 zi3 chang2 nian2 you3 bing4 jia1 tu2 si4 bi4 yi1 pin2 ru2 xi3
    原文结果： yi1 jin4 men2 wo3 bei4 jing1 dai1 le zhe4 hu4 ming2 jiao4 pang2 ji2 de lao3 nong2 shi4 kang4 mei3 yuan2 chao2 fu4 shang1 hui2 xiang1 de lao3 bing1 qi1 zi3 chang2 nian2 you3 bing4 jia1 tu2 si4 bi4 yi1 pin2 ru2 xi3
    原文汉字： 一进门我被惊呆了这户名叫庞吉的老农是抗美援朝负伤回乡的老兵妻子长年有病家徒四壁一贫如洗
    识别结果： 一进门我被惊呆了这户名叫庞吉的老农是抗美援朝负伤回乡的老兵妻子长年有病家徒四壁一贫如洗



# 相关内容

[我的github: https://github.com/audier](https://github.com/audier)

[我的github博客: audier.github.io](https://audier.github.io)

[我的csdn博客: https://blog.csdn.net/chinatelecom08](https://blog.csdn.net/chinatelecom08)
