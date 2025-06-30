import json
import torch

from torch.utils.data import Dataset
from torchvision import transforms, io
from typing import List, Union

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

w2v_dict = {'0': 0, '0_to_10': 1, '10_to_20': 2, '20_to_30': 3, '30_to_40': 4, '40_to_50': 5, '50_to_60': 6,
            '60_to_70': 7, '70_to_80': 8, '80_to_90': 9, '90_to_100': 10, 'NVG_surface': 11, 'buildings': 12,
            'low_vegetation': 13, 'playgrounds': 14, 'trees': 15, 'water': 16, 'no': 17, 'yes': 18, 'green_house': 19,
            'road': 20, 'bridge': 21, 'others': 22}

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def make_answer_vec(answer_str):
    answer_vec = torch.zeros(23, dtype=torch.float32)
    answer_vec[w2v_dict[answer_str]] = 1
    return answer_vec


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class DatasetNew(Dataset):
    def __init__(self, json_path, img_path, mask_path):
        self.len = 0
        self.img_path = img_path
        self.img_question_answer_set = []

        self.img1_dict = {}
        self.img2_dict = {}

        transform = transforms.Compose([
            ZeroOneNormalize(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.question_json = json.load(open(json_path, 'r'))['question']

        self.mask_path = mask_path

        for question in self.question_json:
            file_name = question['img_id']

            if self.img1_dict.get(file_name) is None:
                im1 = io.read_image(self.img_path + 'im1/' + file_name)
                im2 = io.read_image(self.img_path + 'im2/' + file_name)
                self.img1_dict[file_name] = transform(im1)
                self.img2_dict[file_name] = transform(im2)

            q_str = question['question']
            type_str = question['type']
            a_str = question['answer']
            id = question['id']
            answer_vec = make_answer_vec(a_str)
            q_str = tokenize(q_str, 40, True).squeeze(0)

            self.img_question_answer_set.append([file_name, q_str, type_str, answer_vec, id])
            self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_name, q_str, type_str, answer_vec, a_id = self.img_question_answer_set[idx]

        img1 = self.img1_dict[file_name]
        img2 = self.img2_dict[file_name]

        mask = io.read_image(self.mask_path + str(a_id) + '.png')
        mask = mask.div(255)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return img1, img2, q_str, type_str, answer_vec, mask, str(a_id)
