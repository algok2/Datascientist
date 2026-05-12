# 한글 맞춤법 교정기
# HuggingFace T5 based j5ng/et5-typos-corrector Model

# 관련 라이브러리를 호출합니다.
import os
import shutil
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import T5Config
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
    
# 한글 구어체 맞춤법 교정을 위한 언어 모델명을 설정합니다.
model_name = 'j5ng/et5-typos-corrector'

# 트랜스포머 모델을 저장할 폴더를 생성합니다.
download_dir = os.path.join(os.path.expanduser(path = '~'), 'Downloads')
model_dir = os.path.join('HuggingFace', model_name.replace('/', '-'))
local_model_dir = os.path.join(download_dir, model_dir)

# 모델을 저장하거나 저장된 모델을 불러와서 초기화합니다.
if not os.path.exists(path = local_model_dir):
    print('모델을 다운로드합니다...')
    
    # T5 기반 언어 모델과 토크나이저를 초기화합니다.
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path = model_name, 
        cache_dir = local_model_dir, 
        use_safetensors = True
    )
    
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path = model_name, 
        cache_dir = local_model_dir
    )
    
    # 모델을 로컬 폴더에 저장합니다.
    model.save_pretrained(save_directory = local_model_dir)
    tokenizer.save_pretrained(save_directory = local_model_dir)
    print(f'모델을 "{local_model_dir}"에 저장했습니다.')
    
    # 모델을 임시로 저장했던 폴더를 삭제합니다.
    delete_folder = os.path.join(local_model_dir, f'models--{model_name.replace("/", "--")}')
    if os.path.exists(path = delete_folder):
        shutil.rmtree(path = delete_folder)
    
else:
    print('저장된 모델을 불러옵니다...')

    # 직접 내려받은 bin 파일을 safetensors 파일로 저장합니다.
    bin_path = os.path.join(local_model_dir, 'pytorch_model.bin')

    if os.path.exists(path = bin_path):
        config = T5Config.from_pretrained(pretrained_model_name_or_path = local_model_dir)
        model = T5ForConditionalGeneration(config = config)

        state_dict = torch.load(f = bin_path, map_location = 'cpu')
        model.load_state_dict(state_dict = state_dict)

        model.save_pretrained(save_directory = local_model_dir, safe_serialization = True)
        os.remove(bin_path)
    
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path = local_model_dir, 
        use_safetensors = True, 
        local_files_only = True
    )
    
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path = local_model_dir
    )

print('모델 로딩을 완료했습니다!')

# 사용 가능한 디바이스를 확인합니다.
# [참고] MacOS M1에서 실행할 때 'cuda:0'을 'mps:0'으로 대신합니다.
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 모델을 사용 가능한 디바이스로 이동시킵니다.
model = model.to(device)

# 한글 구어체 맞춤법 검사기를 생성합니다.
def et5_typo_corrector(text, seed = 1):
    
    # 프롬프트를 설정합니다.
    prompt = '원본 내용을 보존하면서 맞춤법을 고쳐주세요: '

    # 입력 텍스트를 모델에 입력할 PyTorch 텐서 형식으로 인코딩합니다.
    input_encoding = tokenizer(
        text = prompt + text,
        return_tensors = 'pt'
    )

    # 모델의 generate 함수를 사용하여 텍스트를 생성합니다.
    # input_ids와 attention_mask를 선택한 디바이스(cpu 또는 gpu)로 옮깁니다.
    # max_length: 출력 텍스트의 최대 토큰 수를 설정합니다.
    # num_beams: Beam Search 기법에서 사용할 빔의 개수를 지정합니다.
    # [참고] Beam Search는 여러 후보 문장에서 가장 확률 높은 문장을 선택합니다.
    # early_stopping: 최적 문장이 확정되면 조기에 탐색을 종료합니다.
    # length_penalty: 출력 텍스트의 길이를 줄이거나 늘리지 않도록 설정합니다.
    # [참고] 0.8은 짧은 문장 선호, 1.0은 기본값, 1.2는 긴 문장을 선호합니다.
    # do_sample: False는 확률적 샘플링, True는 결정적 탐색을 수행합니다.(창의적 문장 방지)
    # no_repeat_ngram_size: 같은 n-gram을 반복하지 않도록 설정합니다.
    # repetition_penalty: 반복할 때 페널티를 부여하도록 설정합니다.(보통 1.0보다 큰 값 사용)
    # top_k, top_p: 완전한 무작위 샘플링 대신 상위 k개의 후보 또는 누적 확률 p 범위 내에서 
    # 샘플링하는 방법을 사용하면 다양성을 유지하면서 결과의 품질과 일관성을 높일 수 있습니다.
    # [참고] do_sample = True일 때 top_k와 top_p 매개변수는 적용되지 않습니다.
    with torch.no_grad():
        output_encoding = model.generate(
            input_ids = input_encoding.input_ids.to(device),
            attention_mask = input_encoding.attention_mask.to(device),
            max_length = len(text) + 20,
            num_beams = 5,
            early_stopping = False,
            length_penalty = 1.5,
            do_sample = False, 
            no_repeat_ngram_size = 3,
            repetition_penalty = 1.5,
            # top_k = 50,
            # top_p = 0.95
        )

    # 모델이 생성한 토큰 시퀀스를 사람이 읽을 수 있는 텍스트로 변환합니다.
    # skip_special_tokens = True 옵션은 [PAD], [EOS] 등의 특수 토큰을 제거합니다.
    output_text = tokenizer.decode(
        token_ids = output_encoding[0],
        skip_special_tokens = True
    )

    # 결과 텍스트를 반환합니다.
    return output_text

# 한글 맞춤법 검사기를 실행하는 재귀함수를 생성합니다.
def correct(text, show_process = True):
    
    # 입력 문장의 클래스에 맞게 한글 맞춤법 검사를 실행합니다.
    if isinstance(text, str):
        return et5_typo_corrector(text = text)
    
    elif isinstance(text, pd.Series):
        iterator = tqdm(text) if show_process else text
        return type(text)((correct(text = item, show_process = False) for item in iterator), index = text.index)
    
    elif isinstance(text, (list, tuple, np.ndarray)):
        iterator = tqdm(text) if show_process else text
        return type(text)(correct(text = item, show_process = False) for item in iterator)
    
    else:
        raise TypeError('문자열, 리스트, 튜플, 또는 시리즈를 입력하세요!')
        return None


# End of Document