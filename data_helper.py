
import os
import json
import random
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from albert import tokenization


class TrainData(object):
    def __init__(self, config):

        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        texta_list = []
        textb_list = []
        labels = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text_a, text_b, label = line.strip().split("\t")[1:]
                    texta_list.append(text_a)
                    textb_list.append(text_b)
                    labels.append(label)
                except:
                    continue

        return texta_list, textb_list, labels

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_len):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_len:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def trans_to_index(self, text_as, text_bs):
        """
        将输入转化为索引表示
        :param text_as: 输入
        :param text_bs:
        :return:
        """
        # 创建词字典
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        # 所有行的one-hot/mask/句子索引列表
        input_ids = []
        input_masks = []
        segment_ids = []
        for text_a, text_b in zip(text_as, text_bs):
            text_a = tokenization.convert_to_unicode(text_a)
            text_b = tokenization.convert_to_unicode(text_b)
            # tokenizer只是将句子分割成字，不是分词
            # [为 何 我 无 法 申 请 ...]
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)

            # 裁剪拼接后的左右句子到所配置的长度
            self._truncate_seq_pair(tokens_a, tokens_b, self._sequence_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            # [101, 2, 43, 23, 83, 321, 9 ... 102 ... 102]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            # 全为1，[1, 1, 1, 1, 1, 1 ...]
            input_masks.append([1] * len(input_id))
            # [CLS]--sentence1--[SEP]的索引为0，sentence2--[SEP]的索引为1
            # [0 0 0 0 0 0 ...0 1 1 1 1 1 1 1 ...1]
            segment_ids.append([0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1))

        return input_ids, input_masks, segment_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))
            pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
            pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training
        :return:
        """

        # 1，读取原始数据
        # [ [为何我无法申请开通花呗信用卡收款], [花呗分期付款会影响使用吗], ...]
        # [ [支付宝开通信用卡花呗收款不符合条件怎么回事], [花呗分期有什么影响吗], ... ]
        texta_list, textb_list, labels = self.read_data(file_path)
        print("read file finished")

        label_to_index = {"0": 0, "1": 1}
        # 如果训练-写入label_to_index.json，否则读取
        # label_to_index：{"0":0, "1":1}
        # if is_training:
        #     label_type = list(set(labels))
        #     # {"0":0, "1":1}或{"0":1, "1":0}
        #     label_to_index = dict(zip(label_type, list(range(len(label_type)))))
        #     with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as fw:
        #         json.dump(label_to_index, fw, indent=0, ensure_ascii=False)
        # else:
        #     with open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf8") as fr:
        #         label_to_index = json.load(fr)

        # 2，输入转索引，很耗费时间
        # 注：input_mask全是1，即没有进行mask
        # inputs_ids：[101, 2769, 4638, 2582, 720, 2458, 6858, 5709, ...]
        # input_masks：[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,..., 1, 1, 1, 1, 1, 1, 0, 0,...0]
        # segment_ids：[0, 0, 0, 0, 0, 0,..., 0, 0, 1, 1, 1, 1, 1 ... 1, 1, 0, 0,...0]
        inputs_ids, input_masks, segment_ids = self.trans_to_index(texta_list, textb_list)
        print("index transform finished")

        # 对三个列表，即每行的索引进行padding，在最后面补0，0对应的字符是'PAD'
        # [101, 2, 43, 23, 83, 321, 9 ... 102 ... 102  0 0 0 0 0 0 ...]
        inputs_ids, input_masks, segment_ids = self.padding(inputs_ids, input_masks, segment_ids)

        # 3，标签字符串转为整型
        # [1,1,0,1,0,0,1,0 ...]
        labels_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        for i in range(5):
            print("line {}: *****************************************".format(i))
            print("text_a: ", texta_list[i])
            print("text_b: ", textb_list[i])
            print("input_id: ", inputs_ids[i])
            print("input_mask: ", input_masks[i])
            print("segment_id: ", segment_ids[i])
            print("label_id: ", labels[i])

        return inputs_ids, input_masks, segment_ids, labels_ids, label_to_index

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids))
        random.shuffle(z)
        input_ids, input_masks, segment_ids, label_ids = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_label_ids = label_ids[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids=batch_label_ids)

