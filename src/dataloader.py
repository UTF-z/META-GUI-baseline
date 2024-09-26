import json
import random
from PIL import Image
from DataProcessing import ActionToIdx
import torch
from config import Config
from transformers import AutoTokenizer, AutoProcessor
from transformers import ViTFeatureExtractor


fake_box = [[0, 0, 0, 0]]


def extract_image_feature(config, image, processor):
    fake_text = ["hello"]
    if "layoutlmv2" in config.encoder_model_type:
        image_feature = processor(image, fake_text, boxes=fake_box, return_tensors="pt")["image"]
    elif "clip" in config.encoder_model_type:
        image_feature = processor(text=fake_text, images=image, return_tensors="pt")["pixel_values"]
    else:
        image_feature = torch.tensor(processor(image)["pixel_values"][0]).unsqueeze(0)

    return image_feature


def get_start_end(tokenizer, utterance_split, text_split):
    """
    extract the start position and end position, which is the parameter of "input" action,  from the dialogue histories
    This function requires that the dialog histories are in reverse order, and there exists a "sep_token" between two
    different utterances.
    """

    start = -1
    end = -1
    for i in range(0, len(utterance_split) - len(text_split) + 3):
        turn_flag = True
        for j in range(0, len(text_split)-2):
            if utterance_split[i + j] == text_split[j+1]:
                continue
            else:
                turn_flag = False
                break
        if turn_flag:
            start = i
            end = i + len(text_split) - 3
            break
    if start == -1:
        print(utterance_split)
        print(text_split)
        raise ValueError("Can not extract the start postion and end postion!")
    return start, end


def reply_data_loader(batch_size, data_path, config: Config, train=True):
    '''
    返回每一步数据的对话历史, 动作历史, 当前页面上元素的信息, 以及截屏图像特征, 作为模型的输入
    还会返回response的token序列, 这是模型训练的标签
    返回的元组:
    (input_ids, Optional[image_feature] attention_masks, token_type_ids, bboxs, reply_texts)
    '''
    # 加载数据集
    with open(data_path, 'r') as reader:
        data = json.load(reader)
    if batch_size != 1:
        raise ValueError("only support batch_size=1!")

    if train:
        random.shuffle(data)

    # 加载选定的encoder所需的processor和tokenizer, 并且添加action tokens
    if config.multi_modal:
        if "layoutlmv2" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type, revision="no_ocr")
        elif "clip" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type)
        else:
            processor = ViTFeatureExtractor()

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_type)
    if config.history == "all" or config.history == "action":
        tokenizer.add_tokens(list(ActionToIdx.keys()))

    input_ids = []
    attention_masks = []
    token_type_ids = []
    reply_texts = []
    bbox_s = []

    # 对每一步数据进行处理
    for d in data:
        if d["action"] != "response":
            continue
        input_id = []
        token_type_id = []
        attention_mask = []
        bbox = []
        item_input_id = []
        item_attention_mask = []

        dialogue_text_list = d["dialog"].copy()
        dialogue_text_list.reverse()

        history_flag = False

        # 对dialog历史进行分词, 获取input_id, token_type_id, attention_mask. 如果配置中指定需要添加action, 那么把action历史的最后三个加入到输入中.
        # 每个token都对应一个bbox, 对于一轮对话, 第一个bbox是[0, 0, 1000, 1000], 之后每个token都是[0, 0, 0, 0]
        if config.history == "all" or config.history == "action":
            action_history = d["action_history"]

            if len(action_history) != 0:
                action_lists = []
                for action_history_ in action_history:
                    action_lists.append(f"[{action_history_['action_info'].split('|')[1]}]")
                action_lists = action_lists[-3:]    # 上三个动作?
                action_infos = " ".join(action_lists)

                dialogue_text_list = [action_infos] + dialogue_text_list

                dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
                if "layoutlmv2" in config.encoder_model_type:
                    dialog_token = tokenizer([dialogue_text], boxes=fake_box,
                                             max_length=config.dialog_seq_length, truncation=True)
                else:
                    dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length, truncation=True)

                input_id += dialog_token["input_ids"]
                token_type_id += dialog_token["token_type_ids"]
                attention_mask += dialog_token["attention_mask"]
                max_page_length = 512 - len(dialog_token["input_ids"])
                bbox += [[0, 0, 1000, 1000]]
                bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)
                history_flag = True

        if not history_flag:
            dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
            if "layoutlmv2" in config.encoder_model_type:
                dialog_token = tokenizer([dialogue_text], boxes=fake_box,
                                         max_length=config.dialog_seq_length, truncation=True)
            else:
                dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length, truncation=True)

            input_id += dialog_token["input_ids"]
            token_type_id += dialog_token["token_type_ids"]
            attention_mask += dialog_token["attention_mask"]
            max_page_length = 512 - len(dialog_token["input_ids"])
            bbox += [[0, 0, 1000, 1000]]
            bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)

        items = d["items"]
        text = d["response"]

        # items是当前页面上的元素. 把每个元素的text字段, 或者元素名称送入tokenizer进行分词, 分词之后每一个tokenid都对应这个元素的包围框. 
        # 最后把这个元素的input_id, attention_mask, token_type_id和bbox加入到当前轮次的总input_id, attention_mask, token_type_id, bbox中. 
        for item in items:
            item_text = item["text"]
            if "layoutlmv2" in config.encoder_model_type:
                item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
            else:
                item_text_token = tokenizer(item_text, add_special_tokens=False)
            if len(item_text_token["input_ids"]) == 0:
                item_text = item["type"].split(".")[-1]
                if "layoutlmv2" in config.encoder_model_type:
                    item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
                else:
                    item_text_token = tokenizer(item_text, add_special_tokens=False)
            item_input_id += item_text_token["input_ids"]
            item_attention_mask += item_text_token["attention_mask"]
            border = item["border"]
            resize_border = [int(border[0]*1000/1440), int(border[1]*1000/2560),
                             int(border[2]*1000/1440), int(border[3]*1000/2560)]
            bbox += [resize_border] * len(item_text_token["input_ids"])
            if len(item_input_id) > max_page_length:
                break
        # 输入token序列最长512, 否则截断
        if len(item_input_id) > max_page_length:
            item_input_id = item_input_id[:max_page_length]
            item_attention_mask = item_attention_mask[:max_page_length]
            bbox = bbox[:512]
        input_id += item_input_id
        attention_mask += item_attention_mask
        token_type_id += [1] * len(item_input_id)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        bbox_s.append(bbox)

        # 如果需要图像模态, 那么提取当前页面图片, 根据之前设置的processor抽取feature.
        if config.multi_modal:
            image_path = d["screenshot_history"][-1]
            image = Image.open(image_path).convert("RGB")
            image_feature = extract_image_feature(config, image, processor)

        # 如果要历史数据, 就把最多前两张历史页面截图feature拼接在当前页面feature的前面.
            if config.history == "all" or config.history == "screen":
                image_histories = d["screenshot_history"]
                if len(image_histories) == 2:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                elif len(image_histories) >= 3:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                    image_history = image_histories[-3]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)

        # 当前步的reply的token序列
        if "layoutlmv2" in config.encoder_model_type:
            reply_text_tokenized = tokenizer([text], boxes=fake_box, padding=True)["input_ids"]
        else:
            reply_text_tokenized = tokenizer(text, padding=True)["input_ids"]

        reply_texts.append(reply_text_tokenized)

        if config.multi_modal:
            yield torch.tensor(input_ids), \
                  image_feature, \
                  torch.tensor(attention_masks, dtype=torch.float), \
                  torch.tensor(token_type_ids), \
                  torch.tensor(bbox_s), \
                  torch.tensor(reply_texts),
        else:
            yield torch.tensor(input_ids), \
                  torch.tensor(attention_masks, dtype=torch.float), \
                  torch.tensor(token_type_ids), \
                  torch.tensor(bbox_s), \
                  torch.tensor(reply_texts),

        input_ids = []
        attention_masks = []
        token_type_ids = []
        bbox_s = []
        reply_texts = []


def action_data_loader(batch_size, data_path, config: Config, train=True):
    '''
    返回每一步数据的对话历史, 动作历史, 当前页面上元素的信息, 以及截屏图像特征, 作为模型的输入
    当前的动作type, 滑动方向, 目标item, 以及输入对应的start和end, 作为模型的输出标签
    返回的元组:
    (input_ids, Optional[image_feature] attention_masks, token_type_ids, bboxs, item_metrixes, actions, start, ends, target_items, directions)
    '''
    # 加载数据
    with open(data_path, 'r') as reader:
        data = json.load(reader)
    if batch_size != 1:
        raise ValueError("only support batch_size=1!")

    if train:
        random.shuffle(data)

    # 设置图像特征提取器
    if config.multi_modal:
        if "layoutlmv2" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type, revision="no_ocr")
        elif "clip" in config.encoder_model_type:
            processor = AutoProcessor.from_pretrained(config.encoder_model_type)
        else:
            processor = ViTFeatureExtractor()

    # 设置tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_type)
    if config.history == "all" or config.history == "action":
        tokenizer.add_tokens(list(ActionToIdx.keys()))

    input_ids = []
    attention_masks = []
    token_type_ids = []
    item_matrixes = []

    target_items = []
    actions = []
    starts = []
    ends = []
    directions = []

    bbox_s = []
    mat_length = 0

    for d in data:
        input_id = []
        token_type_id = []
        attention_mask = []
        item_matrix = []
        bbox = []
        item_input_id = []
        item_attention_mask = []

        dialogue_text_list = d["dialog"].copy()
        # 倒序是为了方便提取start和end
        dialogue_text_list.reverse()

        history_flag = False
        # 如果要动作历史特征, 那么把最多前三个动作拼接进入当前对话之前
        if config.history == "all" or config.history == "action":
            action_history = d["action_history"]
            if len(action_history) != 0:
                action_lists = []
                for action_history_ in action_history:
                    action_lists.append(f"[{action_history_['action_info'].split('|')[1]}]")
                action_lists = action_lists[-3:]
                action_infos = " ".join(action_lists)

                dialogue_text_list = [action_infos] + dialogue_text_list

                # 对对话以及动作历史进行分词
                if "clip" in config.encoder_model_type:
                    dialogue_text = " ".join(dialogue_text_list)
                else:
                    dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
                if "layoutlmv2" in config.encoder_model_type:
                    # 注意这里会padding到最大长度
                    dialog_token = tokenizer([dialogue_text], boxes=fake_box, max_length=config.dialog_seq_length,
                                             padding="max_length", truncation=True)
                else:
                    dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length,
                                             padding="max_length", truncation=True)

                mat_length += len(dialog_token["input_ids"])
                input_id += dialog_token["input_ids"]
                if "clip" not in config.encoder_model_type:
                    token_type_id += dialog_token["token_type_ids"]
                attention_mask += dialog_token["attention_mask"]
                max_page_length = 512 - len(dialog_token["input_ids"])
                # 每个token都有一个bbox
                bbox += [[0, 0, 1000, 1000]] # 这个对应[CLS]标签
                bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)
                history_flag = True

        # 不要动作历史, 或者动作历史为空, 那么就直接把对话分词
        if not history_flag:
            if "clip" in config.encoder_model_type:
                dialogue_text = " ".join(dialogue_text_list)
            else:
                dialogue_text = tokenizer.sep_token.join(dialogue_text_list)
            if "layoutlmv2" in config.encoder_model_type:
                dialog_token = tokenizer([dialogue_text], boxes=fake_box, max_length=config.dialog_seq_length,
                                         padding="max_length", truncation=True)
            else:
                dialog_token = tokenizer(dialogue_text, max_length=config.dialog_seq_length,
                                         padding="max_length", truncation=True)

            mat_length += len(dialog_token["input_ids"])
            input_id += dialog_token["input_ids"]
            if "clip" not in config.encoder_model_type:
                token_type_id += dialog_token["token_type_ids"]
            attention_mask += dialog_token["attention_mask"]
            # max_page_length -= len(dialog_token["input_ids"]) - 1
            max_page_length = 512 - len(dialog_token["input_ids"])
            bbox += [[0, 0, 1000, 1000]]
            bbox += [[0, 0, 0, 0]] * (len(dialog_token["input_ids"]) - 1)

        # 当前这一步的实际动作编号
        actions.append(ActionToIdx[d["action"]])

        # 如果是输入动作, 那么就从对话历史中找出这个input的start和end, 否则就是-100. 这里默认了输入动作的内容是来源于历史对话的.(奇奇怪怪)
        if d["input"] is not None:
            inputs_text = d["input"]
            if "layoutlmv2" in config.encoder_model_type:
                inputs_text_token = tokenizer([inputs_text], boxes=fake_box)
            else:
                inputs_text_token = tokenizer(inputs_text)
            start, end = get_start_end(tokenizer, dialog_token["input_ids"],
                                       inputs_text_token["input_ids"])
            starts.append(start)
            ends.append(end)
        else:
            starts.append(-100)
            ends.append(-100)

        # 如果是滑动动作, 就记录当前步的滑动方向, 否则是-100
        if d["scroll"] is not None:
            directions.append(d["scroll"])
        else:
            directions.append(-100)

        # 当前步的目标item编号
        target_item = d["target"]
        if target_item is None:
            target_items.append(-100)
        else:
            target_items.append(d["target"])

        items = d["items"]
        for item in items:
            # 开始处理页面元素, 每个元素先根据元素文本内容或者元素type来分词, 每个token的位置都对应该元素的reshaped bbox
            item_text = item["text"]
            if "layoutlmv2" in config.encoder_model_type:
                item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
            else:
                item_text_token = tokenizer(item_text, add_special_tokens=False)
            if len(item_text_token["input_ids"]) == 0:
                item_text = item["type"].split(".")[-1]
                if "layoutlmv2" in config.encoder_model_type:
                    item_text_token = tokenizer([item_text], boxes=fake_box, add_special_tokens=False)
                else:
                    item_text_token = tokenizer(item_text, add_special_tokens=False)
            item_input_id += item_text_token["input_ids"]
            item_attention_mask += item_text_token["attention_mask"]
            border = item["border"]
            resize_border = [int(border[0] * 1000 / 1440), int(border[1] * 1000 / 2560),
                             int(border[2] * 1000 / 1440), int(border[3] * 1000 / 2560)]
            item_token_length = len(item_text_token["input_ids"])
            bbox += [resize_border] * item_token_length

            # 每个item都对应item_matrix的一行, 这一行的长度就是整个这一步全部文本模态输入的input_id的长度, 这一行只在该item的token位置上有值, 其余位置为0
            item_matrix.append([0.0]*mat_length + [1/item_token_length]*item_token_length)
            mat_length += item_token_length
            # 处理截断
            if len(item_input_id) > max_page_length:
                break
        if len(item_input_id) > max_page_length:
            item_input_id = item_input_id[:max_page_length]
            item_attention_mask = item_attention_mask[:max_page_length]
            bbox = bbox[:512]
            mat_length = 512
        input_id += item_input_id
        attention_mask += item_attention_mask
        token_type_id += [1] * len(item_input_id)
        for i, item_mat in enumerate(item_matrix):
            if len(item_mat) < mat_length:
                item_mat += [0] * (mat_length - len(item_mat))
            else:
                item_matrix[i] = item_mat[:mat_length]
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        if "clip" not in config.encoder_model_type:
            token_type_ids.append(token_type_id)
        bbox_s.append(bbox)
        item_matrixes.append(item_matrix)
        mat_length = 0

        # 当前这一步的图像信息
        if config.multi_modal:
            image_path = d["screenshot_history"][-1]
            image = Image.open(image_path).convert("RGB")
            image_feature = extract_image_feature(config, image, processor)

            if config.history == "all" or config.history == "screen":
                image_histories = d["screenshot_history"]
                if len(image_histories) == 2:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                elif len(image_histories) >= 3:
                    image_history = image_histories[-2]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)
                    image_history = image_histories[-3]
                    image = Image.open(image_history).convert("RGB")
                    image_feature = torch.cat([extract_image_feature(config, image, processor), image_feature],
                                              dim=0)

        if train:
            if config.multi_modal:
                yield torch.tensor(input_ids), \
                      image_feature, \
                      torch.tensor(attention_masks, dtype=torch.float), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions)
            else:
                yield torch.tensor(input_ids), \
                      torch.tensor(attention_masks, dtype=torch.float), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions)
        else:
            if config.multi_modal:
                yield d["screenshot_history"][-1], \
                      torch.tensor(input_ids), \
                      image_feature, \
                      torch.tensor(attention_masks, dtype=torch.float), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions), \
                      d["turn"]
            else:
                yield d["screenshot_history"][-1], \
                      torch.tensor(input_ids), \
                      torch.tensor(attention_masks, dtype=torch.float), \
                      torch.tensor(token_type_ids), \
                      torch.tensor(bbox_s), \
                      torch.tensor(item_matrixes), \
                      torch.tensor(actions), \
                      torch.tensor(starts), \
                      torch.tensor(ends), \
                      torch.tensor(target_items), \
                      torch.tensor(directions), \
                      d["turn"]

        # 清空返回列表
        input_ids = []
        attention_masks = []
        token_type_ids = []
        bbox_s = []
        item_matrixes = []
        actions = []
        starts = []
        ends = []
        target_items = []
        directions = []
