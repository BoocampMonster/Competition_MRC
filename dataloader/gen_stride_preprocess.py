from transformers import PreTrainedTokenizerFast
from datasets import arrow_dataset
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
class gen_stride_preprocess:
    def __init__(self, tokenizer:PreTrainedTokenizerFast, max_length:int, max_answer_length:int, stride:int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_length = max_answer_length 
        self.stride = stride
        
    def train(self, train_data:arrow_dataset.Batch) -> arrow_dataset.Dataset:
        """
        max_length = 384를 넘어가는 문장이 들어오게 되면 stride 길이 만큼 중첩해서 문장을 쪼개는 방식의 전처리 함수입니다.
        truncation = only_second로 고정시키며, 이는 첫번째로 들어오는 sentence는 고정시키고 반복적으로 넣어줍니다.
        그리고 second 문장인 context의 길이가 max_length를 넘어가게 되면 max_length만큼 짤라서 나눠서 tokenizer에 넣게 됩니다.

        question + context 문장이 384를 넘어가게 되면 384(max_length) + 128(stride) 길이만큼 토크나이징한 후에,
        question + context[나머지 길이 stride(128) + remainder(ex 124)]만큼 토크나이징을 진행합니다.

        return: 다음과 같은 키 밸류값을 가집니다.
            {
            'input_ids'(List[int]) : 토큰들을 id값으로 반환한 리스트
            'token_type_ids'(List[int]) : 문장을 구분해주는 리스트, BERT에서는 필수이지만, 나머지에서는 Optional합니다. ex) [0,0,0,1,1,1,2,2,2]
            'attention_mask'(List[int]) : attention을 적용할 문장일 경우 1, pad토큰일 경우 0으로 반환하는 리스트 ex) [1,1,1,1,1,0,0,0,0]
            'start_positions'(List[int]) : answer에 해당하는 토큰의 시작 인덱스를 반환하는 리스트
            'end_positions'(List[int]) : answer에 해당하는 토큰의 끝 인덱스를 반환하는 리스트
            }
        """

        assert isinstance(train_data, arrow_dataset.Batch), "huggingface datasets를 이용해 Batch 데이터를 불러와주세요"
        if 't5' in self.tokenizer.name_or_path:
            # T5가 QA에 입력으로 받는 형식으로 만들어줍니다.
            question = [f"question: {q} </s>" for q in train_data["question"]]
            context = [f"context: {c} </s>" for c in train_data["context"]]

            tokenized_sentences = self.tokenizer(
                question,
                context,
                truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
                max_length=self.max_length,
                stride=self.stride,
                add_special_tokens=True,
                return_overflowing_tokens=True, # (List[int]) : 여러개로 쪼갠 문장들이 하나의 같은 context라는 것을 나타내주는 리스트, batch 단위일때 사용합니다.
                return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
                padding="max_length",
            )

        # example 하나가 여러 sequence에 대응하는 경우를 위해 매핑이 필요함.
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")
        # offset_mappings으로 토큰이 원본 context 내 몇번째 글자부터 몇번째 글자까지 해당하는지 알 수 있음.
        offset_mapping = tokenized_sentences.pop("offset_mapping")

        # 정답지를 만들기 위한 리스트
        targets = []

        for i, offsets in enumerate(offset_mapping):
            # sequence가 속하는 example을 찾는다.
            example_index = overflow_to_sample_mapping[i]
            targets.append(f"{train_data['answers'][example_index]['text'][0]} </s>")
        
        labels = self.tokenizer(
            text_target = targets,
            max_length = self.max_answer_length,
            padding = "max_length",
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True
        )

        tokenized_sentences['labels'] = labels['input_ids']
        # tokenized_sentences['decoder_attention_mask'] = labels["attention_mask"]

        return tokenized_sentences

    def valid(self, valid_data:arrow_dataset.Batch) -> arrow_dataset.Dataset:
        assert isinstance(valid_data, arrow_dataset.Batch), "huggingface datasets를 이용해 Batch 데이터를 불러와주세요"
        if 't5' in self.tokenizer.name_or_path:
            question = [f"question: {q} </s>" for q in valid_data["question"]]
            context = [f"context: {c} </s>" for c in valid_data["context"]]

            tokenized_sentences = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.max_length,
                stride=self.stride,
                add_special_tokens=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
        
        # example 하나가 여러 sequence에 대응하는 경우를 위해 매핑이 필요함.
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")
        # offset_mappings으로 토큰이 원본 context 내 몇번째 글자부터 몇번째 글자까지 해당하는지 알 수 있음.
        offset_mapping = tokenized_sentences["offset_mapping"]

        # 정답지를 만들기 위한 리스트
        tokenized_sentences["example_id"] = []
        targets = []

        for i, offsets in enumerate(offset_mapping):
            # 해당 example에 해당하는 sequence를 찾음.
            sequence_ids = tokenized_sentences.sequence_ids(i)
            
            # sequence가 속하는 example을 찾는다.
            example_index = overflow_to_sample_mapping[i]
            tokenized_sentences["example_id"].append(valid_data["id"][example_index])
            
            # question과 special token을 제외한 offset_mapping으로 교체
            context_index = 1
            tokenized_sentences["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_sentences["offset_mapping"][i])
            ]
            targets.append(f"{valid_data['answers'][example_index]['text'][0]} </s>")

        labels = self.tokenizer(
            text_target = targets,
            max_length = self.max_answer_length,
            padding = "max_length",
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True
        )
        
        tokenized_sentences['labels'] = labels['input_ids']
        # tokenized_sentences['decoder_attention_mask'] = labels["attention_mask"]

        return tokenized_sentences 

    def test(self, test_data:arrow_dataset.Batch) -> arrow_dataset.Dataset:
        '''
        tokenize된 question + context 문장 중에서
        question에 해당하는 offset_mapping을 None값으로 바꿔주고
        example_id라는 id값을 추가하여 반환합니다.

        return: 다음과 같은 키 밸류값을 가집니다.
            {
            'input_ids'(List[int]) : 토큰들을 id값으로 반환한 리스트
            'token_type_ids'(List[int]) : 문장을 구분해주는 리스트, BERT에서는 필수이지만, 나머지에서는 Optional합니다. ex) [0,0,0,1,1,1,2,2,2]
            'attention_mask'(List[int]) : attention을 적용할 문장일 경우 1, pad토큰일 경우 0으로 반환하는 리스트 ex) [1,1,1,1,1,0,0,0,0]
            'offset_mapping'(List[Tuple[int, int]]) : 변환된 토큰들이 실제 문장에 어디에 위치해 있는지를 나타내는 (start, end) 인덱스를 반환한 리스트 ex) [(0, 2), (3, 5), ...]
            'example_id'(List[int]) : 해당 질문의 id를 의미합니다.
            }
        '''

        assert isinstance(test_data, arrow_dataset.Batch), "huggingface datasets를 이용해 Batch 데이터를 불러와주세요"
        question = [f"question: {q} </s>" for q in test_data["question"]]
        context = [f"context: {c} </s>" for c in test_data["context"]]

        tokenized_sentences = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        overflow_to_sample_mapping = tokenized_sentences.pop("overflow_to_sample_mapping")

        tokenized_sentences["example_id"] = []

        for i in range(len(tokenized_sentences["input_ids"])):
            sequence_ids = tokenized_sentences.sequence_ids(i)
            context_index = 1

            example_index = overflow_to_sample_mapping[i]
            tokenized_sentences["example_id"].append(test_data["id"][example_index])

            tokenized_sentences["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_sentences["offset_mapping"][i])
            ]

        return tokenized_sentences