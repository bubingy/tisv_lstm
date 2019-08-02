# tisv_lstm

此项目无耻抄袭了[ge2e的一个pytorch实现](https://github.com/HarryVolek/PyTorch_Speaker_Verification), 卑鄙的我只实现了端点检测.

尝试过[Mozilla Common Voice](https://voice.mozilla.org/zh-CN)数据集, 亲测有效...

最后谈一下端点检测的实现, 思路很简单:
  1. 对音频序列中所有元素求绝对值, 并转成0到1之间的浮点数;
  2. 用处理后的音频序列训练一个混合度为3的高斯混合模型;
  3. 把3个高斯成分的均值的最大值和最小值, 留下中值, 作为阈值.
  
  
