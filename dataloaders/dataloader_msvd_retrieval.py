from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor

class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path   # 数据集地址
        self.features_path = features_path   # 视频地址
        self.feature_framerate = feature_framerate   #  视频帧率 ？？？？？
        self.max_words = max_words   #  句子最大单词数
        self.max_frames = max_frames   # 视频最大帧数
        self.tokenizer = tokenizer    # 单词 转 token ID
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order  # ？？？ 视频排列方法
        assert self.frame_order in [0, 1, 2]    
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos   #  裁剪视频方法？
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset    # 训练集 验证集 测试集
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")    # 三个集的地址
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")    # 视频描述地址

        with open(video_id_path_dict[self.subset], 'r') as fp:  
            video_ids = [itm.strip() for itm in fp.readlines()]   # 视频名称(.txt文件中记录的)  test:有670个视频

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)    # 视频描述  好像视频数量比video_ids更多

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])   # 得到视频名称(id)，这里是实际的视频文件名(.avi文件)
                if video_id_ not in video_ids:   # 判断 .txt文件中的视频名称(id),是否和.avi文件对应上   选取对应视频
                    continue
                file_path_ = os.path.join(root, video_file)    # 当视频的完整地址
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict   # test:选出670个视频  video_dict记录了670个视频的绝对路径

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:   # 遍历当前集合中的所有视频
            assert video_id in captions   # 判断该视频是否有对应的描述句子
            for cap in captions[video_id]:   # 遍历描述当前视频的每一个句子   cap 是一个列表,列表中的每一个元素是一个单词
                cap_txt = " ".join(cap)   # 将列表拼接成一个句子
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)   # 字典存储 视频名称和描述视频的句子
            self.cut_off_points.append(len(self.sentences_dict))   # 按顺序记录每个视频对应的描述句子的（视频索引+1）

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)   # 所有描述当前所有视频的句子总数量
            self.video_num = len(video_ids)  # 当前视频总数量
            assert len(self.cut_off_points) == self.video_num   # 判断是否记录了当前所有视频对应描述句子的分界
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)  # image_resolution 视频像素
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}   # 特殊 文本 token

    def __len__(self):
        return self.sample_len   # 返回总句子数量

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]   #  当视频名称(编号)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)   # shape = (1,32)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)   # shape = (1,32)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)  # shape = (1,32)

        for i, video_id in enumerate(choice_video_ids):   # video_id 是视频名称(编号)
            words = self.tokenizer.tokenize(caption)  # caption 是描述当前视频的句子(字符串)
            # 分词后的结果 是一个列表  words 长度不一定等于句子的单词数，可能会将某个单词进行拆分
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words   # 在开头添加一个特殊 token
            total_length_with_CLS = self.max_words - 1   #  最大分词数  
            if len(words) > total_length_with_CLS:   # 是否大于最大分词(token)数
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]   # 在结尾添加一个特殊 token

            input_ids = self.tokenizer.convert_tokens_to_ids(words)  # 将token 转为 token ID
            input_mask = [1] * len(input_ids)   # 掩码
            segment_ids = [0] * len(input_ids)  # 断句
            while len(input_ids) < self.max_words:   # 填充
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids  #  前三个张量形状(1,32)  choice_video_ids 是一个列表，记录当前视频名称(编号)

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)   # shape (1,12)
        max_video_length = [0] * len(choice_video_ids)  # [0]

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)   # shape (1,12,1,3,224,224)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]   # 当前视频的完整绝对路径

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']   # shape (采样数，3，224，224)

            if len(raw_video_data.shape) > 3:    # len(raw_video_data.shape)  = 4    若 (采样数，3，224，224)
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)   # shape (采样数,1,3,224,224)
                if self.max_frames < raw_video_slice.shape[0]:   # 如果最大帧数 比 当前采样数小
                    if self.slice_framepos == 0:   # 选择前self.max_frames帧
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # 选择后self.max_frames帧
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)  # 该代码使用NumPy的linspace函数创建一个均匀分布的整数数组，用于从视频中采样特定数量的帧。
                        video_slice = raw_video_slice[sample_indx, ...]   # 实现了"视频帧均匀采样"策略
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)  # 帧排序，默认情况不改变帧顺序

                slice_len = video_slice.shape[0]   # 当前视频帧数
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len   # max_video_length[i] 为当前视频帧数
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask   # video.shpae = (1, 12, 1, 3, 224, 224)  video_mask.shape = (1, 12)   两个变量都是np类型

    def __getitem__(self, idx):
        # print("idx",idx)
        video_id, caption = self.sentences_dict[idx]   # 获取当前的视频名称(编号)和句子描述， 这里只获取一个句子
        # 如果是一个视频对应对个描述句子的情况，如果当前批次中有n个句子描述了一个视频，那么这n个样本的视频都是相同的，但是n个句子不同，切分点就是不同视频的分界

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption) # #  前三个np形状(1,32)  choice_video_ids 是一个列表，记录当前视频名称(编号)
        video, video_mask = self._get_rawvideo(choice_video_ids)   # # video.shpae = (1, 12, 1, 3, 224, 224)  video_mask.shape = (1, 12)   两个变量都是np类型
        return pairs_text, pairs_mask, pairs_segment, video, video_mask   
