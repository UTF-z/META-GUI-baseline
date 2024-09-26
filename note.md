<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2::before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3::before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4::before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5::before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6::before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
    .italic {
        font-style: italic;
    }
    .normal {
        font-style: normal;
    }
    .oblique {
        font-style: oblique;
    }
    .text_decoration {
        text-decoration: underline red wavy;
    }
    .text_shadow {
        font-size: 5em;
        text-shadow:
            -1px -1px 1px #aaa,
            0px 4px 1px rgba(0, 0, 0, 0.5),
            4px 4px 5px rgba(0, 0, 0, 0.7),
            0px 0px 7px rgba(0, 0, 0, 0.4);
    }
    img {
      display: block;
      margin: 0 auto;
      align: center;
    }
</style>
# 代码笔记
## META-GUI数据集格式
### 处理后数据格式
一个大列表, 里面是一个个字典, 每个字典是某一轮中的某一步, 里面有如下字段:
1. screenshot_history: 当前对话轮次的截图历史, 是一个列表, 每个元素是一个字符串, 字符串是截图的文件名, **最后一个元素是当前页面**
2. action_history: 当前对话轮次的操作历史, 是一个列表, **最后一个元素是上一个操作**. 列表中每个元素是一个字典, 字典包含如下字段:
    - image: 该操作发生的页面的截图文件名
    - <span id="action_info">action_info: '\<split\>|\<type\>|\<x, y\>|\<intput text\>'</span>
        如果是swipe, 最后一个字段是lift-point. 此外, type有可能是read, 表示标注者在读文本, 不用管, 跳过即可, 参见`./src/processors.py`
    - items: 该页面中提取的元素列表, 每个元素是一个字典, 字典包含如下字段:
        - text: 元素的文本
        - type: 元素的类型, 比如‘android.widget.ImageButton'
        - border: 元素的边界框, 四个整数, 左上角x, 左上角y, 右下角x, 右下角y
    - target: 该动作点击的目标元素的id, 是一个列表, 如果没有点击, 则为null
3. dialog: 当前轮次的对话历史, 是一个列表, 每个元素是一个字符串, 字符串是对话的文本, 用户和agent交互对话
4. items: 当前页面中提取的元素列表, 每个元素是上面提及的元素字典
5. action: 当前操作的类型, 总共有`click, swipe, input, clear, enter, back, end, response`
6. response: 当前操作的响应, 如果当前操作是response, 则为响应文本, 否则为null
7. target: 当前操作的目标元素, 如果当前操作是click, 则为目标元素的id, 否则为null
8. category: 当前数据的domain, 比如订酒店就是`hotel`
9. input: 当前操作的输入, 如果当前操作是input, 则为输入文本, 否则为null
10. scroll: 当前操作的滑动方向, 如果当前操作是`swipe`, 则为一个整数, 代表方向, 否则是null
11. turn: 当前操作的轮次, 是该轮次原始数据的文件夹路径
### 原始数据格式
每一个对话(dialog)都保存在一个单独的文件夹中, 所谓对话其实就是一次任务执行. 一个对话包含若干轮次(turn), 每个turn就是dialogue的一个子文件夹.
dialog文件夹中有这些东西
1. dialog_id.txt: 对话标志符
2. dialog.json: 本次对话的对话历史, 是一个列表, 每个元素是一个字典, 字典包含如下字段:
    - isUser: 是否是用户发言, 是的话为true, 否则为false
    - text: 对话文本, 一个字符串
    - program: 对话文本的中文翻译, 这个字段可能根本不存在, 别用
3. category.txt: 对话的domain, 比如定酒店的, 那就是'hotel'
4. meta.json: 对话中每个轮次涉及的app, 是一个列表, 每个元素是一个字典, 字典包含如下字段:
    - turn: 轮次号, 从0开始
    - is_single: 是否只有一步操作
    - apps: 本轮涉及的app, 是一个字符串列表, 每个字符串就是app的activity name
5. turn_0, turn_1, ...: 对话的轮次, 每个轮次是一个文件夹, 里面有
    - actions.json: 该轮次的操作历史, 是一个字典, 字典的key是操作序号, 从0开始, 可以根据操作序号找到操作发生的页面截图以及VH, 字典的值是<a href=#action_info>action_info</a>
    - 0.png, 1.png, ...: 该轮次的截图, 文件名是序号
    - 0.xml, 1.xml, ...: 该轮次的view hierarchy, 文件名是序号

## 数据集加载
在`src/dataloader.py`中实现了两个dataloader, `reply_data_loader`和`action_data_loader`. 他们都是生成器
### reply_data_loader
每一次yield一个动作类型为‘response’的数据点, 在这个数据点中, 包括了历史对话信息, 动作历史信息, 当前全体页面元素信息以及截图的图像特征, 这些信息会作为模型的输入. 此外, 还会返回response的token序列, 这是模型训练的标签.

返回的元组(按顺序):
```python
input_ids # Tensor[1, seq_len] 把对话历史, 屏幕元素文本, 以及可选的前三个历史动作拼接在一起, 然后分词得到input_id. 注意最开始是[CLS]
image # Tensor[n, 3, 224, 224], 如果设置了multi_modal字段, 那么就会返回这个值, 代表预处理之后的当前页面的截图, 如果设置了history=all|screen, 那么还会拼接前三个历史页面
attention_masks # Tensor[1, seq_len], input_ids的attention mask, 1表示当前位置的token是有效的, 0表示当前位置的token是无效的
token_type_ids # Tensor[1, seq_len], 代表了序列的token_type_id. 0 是dialog和动作, 1是页面元素
bbox_s # Tensor[1, seq_len, 4], 代表了每个token对应的页面元素的包围框, 如果没有元素, 则为[0, 0, 0, 0], [CLS]的位置对应[0, 0, 1000, 1000]
reply_texts # Tensor[1, reply_seq_len], 代表每个reply的token序列, 注意如果是空字符串, 返回的是[[cls_id, sep_id]] 
```

### action_data_loader
对话历史, 动作历史仍然像reply_data_loader一样编码, 但是此时会padding到config中指定的dialog_seq_length. 此后才会逐个拼接页面元素文本

返回的元组(按顺序):
```python
Optional[screen_path] # str 当前页面的路径, 如果设置了train=True, 则不会返回这个值
input_ids # Tensor[1, seq_len] 把对话历史, 屏幕元素文本. 如果设置了history=all|action, 会把前三个历史动作拼接在dialog前面, 然后分词得到input_id. 注意最开始是[CLS]
Optional[image] # Tensor[n, 3, 224, 224], 如果设置了multi_modal字段, 那么就会返回这个值, 代表预处理之后的当前页面的截图, 如果设置了history=all|screen, 那么还会拼接前三个历史页面
attention_masks # Tensor[1, seq_len], input_ids的attention mask, 1表示当前位置的token是有效的, 0表示当前位置的token是无效的
token_type_ids # Tensor[1, seq_len], 代表了序列的token_type_id. 0 是dialog和动作, 1是页面元素
bbox_s # Tensor[1, seq_len, 4], 代表了每个token对应的页面元素的包围框, 如果没有元素, 则为[0, 0, 0, 0], [CLS]的位置对应[0, 0, 1000, 1000]
item_metrixes # Tensor[1, item_number, seq_len], 代表了每个item对应的token位置. 对应位置为1/item_token_len, 其余位置为0
actions # Tensor[1], 代表当前动作的id
starts # Tensor[1], 如果当前动作是input, 那么代表input在input_ids中的起始位置, 否则为-100
ends # Tensor[1], 如果当前动作是input, 那么代表input在input_ids中的结束位置, 否则为-100
target_items # Tensor[1], 数据点中的“target”字段内容, 如果是none, 则为-100
directions # Tensor[1], 如果当前动作是swipe, 则代表swipe的方向, 否则为-100
Optional[turn] # str 当前轮次原始数据文件夹的路径
```
## 模型结构

### ResponseModel
这个模型用于根据文本模态的input_id生成响应文本, 模型结构如下
- ResponseModel
    - encoder_model
    - reply_text_decoder

`forward`流程:
    1. 用encoder_model去编码input_id, 得到hidden_states
    2. 用reply_text_decoder在reply_text上面做next_token prediction, decode的时候每一层都和input_id在encoder的最后一层做cross-attention
    3. next-token交叉熵损失

`generate`流程:
- 如果没有启用beam_search, 那么使用贪婪生成. 
    1. 将文本模态信息生成的input_id通过encoder编码, 取最后一层hidden. 
    2. 然后往生成序列中先添加一个`[CLS]`token id, 然后用reply_text_decoder和hidden交叉注意力生成下一个token id, 
    3. 一直生成config中指定的reply_seq_length个token id, 中途遇到sep_token_id就break.
    4. 返回生成的token id序列.
- 如果启用beam_search, 则使用束搜索.
    1. 将文本模态信息生成的input_id通过encoder编码, 取最后一层hidden.
    2. 先往candidate中添加一个`Node([cls_token_id], 0, 0)`, 第一个参数是这个node代表的token_ids, 第二个参数是这个句子的对数概率, 第三个是平均每个词的对数概率.
    3. 首先把candidate中所有节点取出, 如果一个节点的文本以`sep_token_id`结尾或者长度达到`reply_seq_length`, 那么将它加入decoded_candidates. 否则对它进行扩展, 扩展方法是用每个节点的文本和hidden进行交叉注意力, 获得下一个token的概率分布, 取出概率最大的前k个token拼接在文本之后, 计算新文本的对数概率和平均每个词的对数概率, 然后把新的k个节点加入new_candidate中. 
    4. 从new_candidate中取出平均对数概率最大的k个节点, 加入candidate中, 重复3.
    5. 最后从decoded_candidate中取出概率最大的一个节点的token_ids, 返回.

### ActionModel
这个模型用于根据文本模态的input_id生成动作类型和参数, 模型结构如下
- ActionModel
    - encoder_model
    - action_type_classifier
    - typing_classifier
    - item_score_classfier
    - direction_classifier
    - 可学习的权重因子, action_loss_weight, type_loss_weight, item_loss_weight, direction_loss_weight

`forward`流程:
1. 用encoder_model去编码input_id, 得到hidden_states
> hidden_states其实有多个部分, 这里会用到两个, 集中解释如下
> - hidden_state.last_hidden_state: 最后一层的每个token的hidden_state, 用于做交叉注意力
> - hidden_state.pooler_output: 最后一层的hidden_state的pooler_output, 这是input_id第一个token, 即`[CLS]`经过线性和激活的结果, 可以理解为全局特征

2. 出动作: 用action_type_classifier和pooler_output做分类, 得到action_type_logits, 并根据输入的action标签做交叉熵损失.
3. 出start和end: 取出last_hidden_state的对话部分(由config.dialog_seq_length控制), 用typing_classifier出start和end, 计算交叉熵损失.
4. 出item: 将item_metrixes和encoder_outputs相乘, 然后得到item部分的embedding, 用item_score_classifier和出item的logits, 和target_items做交叉熵损失.
5. 出方向: 用direction_classifier和pooler_output做分类, 得到direction_logits, 并根据输入的direction标签做交叉熵损失.
6. 最后根据是否配置损失权重, 计算loss

### MultiModalResponseModelWithHistory
这个模型利用了视觉模态输入. 模型结构如下:
- MultiModalResponseModelWithHistory:
    - encoder_model (处理文本输入)
    - transform (GeneralizedRCNNTransform, 负责图像归一化和resize)
    - backbone (resnet_fpn_backbone, 输入图像, 输出特征金字塔)
    - box_roi_pool (MultiScaleRoIAlign, 输入特征金字塔和包围盒, 输出每个包围盒的特征)
    - box_head (MLP, 对包围盒特征进行重映射)
    - struc (若干层AttentionBlock, 拼接文本和图像特征, 进行self-attention)

`forward`流程

- 首先用encoder_model将文本信息`input_ids`编码, 并取出最后一层, 得到`encoder_outputs`
- 然后用resnet_fpn将截图和包围盒转换为每个包围盒一个局部feature
- 如果有历史信息, 对历史图片也提取局部feature, 然后用历史feature当KV, 当前feature当Q, 做一次attention计算
- 用AttentionBlock融合图像和文本信息
- 最后用`replay_text_decoder`做next-token prediction, 返回交叉熵损失.

`generate`流程
基本和上面一样, 先融合图像和文本信息, 然后用贪婪生成或者beam search

### MultiModalActionModelWithHistory

模型结构
- MultiModalActionModelWithHistory
    - encoder_model (处理文本输入)
    - transform (GeneralizedRCNNTransform, 负责图像归一化和resize)
    - backbone (resnet_fpn_backbone, 输入图像, 输出特征金字塔)
    - box_roi_pool (MultiScaleRoIAlign, 输入特征金字塔和包围盒, 输出每个包围盒的特征)
    - box_head (MLP, 对包围盒特征进行重映射)
    - struc (若干层AttentionBlock, 拼接文本和图像特征, 进行self-attention)
    - action_type_classifier (动作类别判别器)
    - typing_classifier (起止位置判别器)
    - item_score_classifier (target item 判别器)
    - direction_classifier (方向判别器)

`forward`流程
1. 出文本特征
2. 出包围盒局部feature
3. 融合文本和图像
4. 直接取出CLS token的feature, 过一次全连接, 经过action_type_classifier出动作, 交叉熵损失
5. 用对话部分token位置的特征, 经过typing_classifier来出起止位置, 交叉熵损失
6. item_matrix和encoder_outputs相乘, 然后用item_score_classifier来出target item, 交叉熵损失
7. 用CLS token的embedding, 经过direction_classfier来出方向, 交叉熵损失
8. 可选地, 多个损失加权求和, 权重可学习, 返回损失和预测结果

## 训练循环 
在这里要做的事情: 
1. 设置epoch进度条: `epoch_iter = tqdm(range(args.epoch), desc='epoch')`
2. 获取device, optimizer, scheduler, 为模型设置device.
3. 在每个epoch中:
    1. 根据配置拿data loader
    2. 设置data loader进度条`batch_iter = tqdm(dataloader, desc='iteration')`
    3. 迭代data loader, 每一次开头把模型设置为train模式
    4. 拆出batch, 设置好每个tensor的device, 过模型, 拿到loss
    5. 进度条设置信息展示, 一般是步数, loss, 一步时间`batch_iter.set_description(desc='iter: %d, loss: %.4f, time: %.4f" % (n_iter, loss, time))`
    6. `loss.backward()`, `optimizer.step()`, `scheduler.step()`三件套, 可以手动累积梯度, 详见代码

## 验证循环
验证会针对action model和response model分别进行
### Action model
先过一遍数据集, 在每个数据点上, 记录target和pred
最后计算如下指标:
1. 动作类别准确率
2. start 准确率
3. end 准确率
4. typing 内容的em
5. typing 内容的f1
6. item 准确率
7. direction 准确率
8. 动作完成率, 只有动作类别和参数都匹配才算完成
9. turn 完成率, 只有当前turn中每一步action type以及对应的参数都正确, 那么才认为这个turn complete

### Response model
每一个数据点都会用response model产生response, 记录下来
整个测试集过完之后, 跳过所有response target为空的数据点, 对于其他数据点, 计算bleu score, 返回平均bleu score
