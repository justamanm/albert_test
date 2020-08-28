import os
import json
import argparse
import random
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from albert import modeling
from model import AlbertSentencePair
from data_helper import TrainData
from metrics import mean, get_multi_metrics


class Trainer(object):
    def __init__(self):
        # config字典
        config_path = "./config/huabei_config.json"
        with open(config_path, "r") as fr:
            self.config = json.load(fr)

        # 数据处理对象
        self.data_obj = TrainData(self.config)

        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "albert_model.ckpt")

        # 加载训练数据集，很耗时
        self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids, lab_to_idx = self.data_obj.gen_data(
            self.config["train_data"])
        # 加载验证数据集，很耗时
        self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_lab_ids, lab_to_idx = self.data_obj.gen_data(
            self.config["eval_data"], is_training=False)

        print("训练集大小: {}".format(len(self.t_lab_ids)))
        print("验证集大小: {}".format(len(self.e_lab_ids)))
        # [0, 1]
        self.label_list = [value for key, value in lab_to_idx.items()]
        print("类别数量: ", len(self.label_list))

        # 所有epoch的训练batch次数
        num_train_steps = int(
            len(self.t_lab_ids) / self.config["batch_size"] * self.config["epochs"])

        # 设置前10%的训练batch使用预热的学习率
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])

        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = AlbertSentencePair(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
        return model

    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            for epoch in range(self.config["epochs"]):
                print("---------- Epoch {}/{} ----------".format(epoch + 1, self.config["epochs"]))
                self.train_num = len(self.t_in_ids)
                print(f"总训练数据{self.train_num}条")
                for i, batch in enumerate(self.data_obj.next_batch(self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids)):
                    num_batches = len(self.t_lab_ids) / self.config["batch_size"]
                    print(f"---------- batch {i+1}/{num_batches} ----------")
                    loss, predictions = self.model.train(sess, batch)
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["label_ids"],
                                                                  labels=self.label_list)
                    print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1

                    # 根据配置，隔一定数量的batch后保存一次模型
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.data_obj.next_batch(self.e_in_ids, self.e_in_masks,
                                                                   self.e_seg_ids, self.e_lab_ids):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                          true_y=eval_batch["label_ids"],
                                                                          labels=self.label_list)
                            eval_accs.append(acc)
                            eval_recalls.append(recall)
                            eval_precs.append(prec)
                            eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

                # TODO 打乱数据
                self.shuffle_data()

            end = time.time()
            print("total train time: ", end - start)

    def shuffle_data(self):
        index_list = [i for i in range(self.train_num)]
        for _ in range(10):
            random.shuffle(index_list)
        self.t_in_ids = [self.t_in_ids[x] for x in index_list]
        self.t_in_masks = [self.t_in_masks[x] for x in index_list]
        self.t_seg_ids = [self.t_seg_ids[x] for x in index_list]
        self.t_lab_ids = [self.t_lab_ids[x] for x in index_list]


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", help="config path of model")
    # args = parser.parse_args()
    trainer = Trainer()
    trainer.train()
