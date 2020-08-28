import os
import re
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from albert import modeling
from albert import optimization


class AlbertSentencePair(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        # 动态指定albert的配置json文件
        albert_config = "albert_config.json"
        for file in os.listdir(config["bert_model_path"]):
            albert_config = re.search(".*\.json", file)
            if albert_config:
                albert_config = albert_config.group(0)
                break
        self.__bert_config_path = os.path.join(config["bert_model_path"], albert_config)

        # self.__num_classes = config["num_classes"]
        self.__num_classes = [0, 1]
        self.__learning_rate = config["learning_rate"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids")

        self.built_model()
        self.init_saver()

    def built_model(self):
        # 创建bert的配置对象，将config文件中配置赋值给配置类
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        self.model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)
        # TODO 获取transformer的输出层
        # (batch_size, 312)
        output_layer = self.model.get_pooled_output()
        # print(output_layer)

        hidden_size = output_layer.shape[-1].value
        if self.__is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        with tf.name_scope("output"):
            print("------output------")
            # 定义分类层权重参数，参与训练
            output_weights = tf.get_variable(
                "output_weights", [len(self.__num_classes), hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            print(output_weights)   # (2, 312)

            # output_bias = tf.get_variable(
            #     "output_bias", [self.__num_classes], initializer=tf.zeros_initializer())
            # [0, 1]
            output_bias = tf.get_variable(
                "output_bias", len(self.__num_classes), initializer=tf.zeros_initializer())
            print(output_bias)      # (2,)

            # (batch_size, 312)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            # (batch_size, 2)
            self.logits = tf.nn.bias_add(logits, output_bias)

            # if self.__num_classes == 1:
            # 大于等于0为1，小于0为0
            # self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), dtype=tf.int32, name="predictions")
            # (batch_size, 2)
            self.predict = tf.nn.softmax(logits, axis=-1)
            print("------predictions------")
            print(self.predict)

            self.predictions = tf.argmax(self.predict, 1)

            # else:
            #     self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        if self.__is_training:
            with tf.name_scope("loss"):
                # if self.__num_classes == 1:
                # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.logits, [-1]),
                #                                                      labels=tf.cast(self.label_ids, dtype=tf.float32))
                # else:
                #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_ids)
                # self.loss = tf.reduce_mean(losses, name="loss")

                # depth:类别数量
                one_hot_labels = tf.one_hot(self.label_ids, depth=2, dtype=tf.float32)

                per_example_loss = -tf.reduce_sum(one_hot_labels * logits, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)

            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        print(f'0标签的个数：{list(batch["label_ids"]).count(0)}，1标签的个数：{list(batch["label_ids"]).count(1)}')
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"]}

        # 训练模型
        # 可以获取到各层的输出
        predict, predictions, _, loss, sequence_output = sess.run([self.predict, self.predictions, self.train_op, self.loss, self.model.sequence_output], feed_dict=feed_dict)
        print("-------sequence_output-------")
        print(predict[:10])
        print(predictions)
        print(f"0标签的个数：{list(predictions).count(0)}")

        # print(sequence_output.shape)
        # print(sequence_output[0][0])
        return loss, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"]}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"]}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
