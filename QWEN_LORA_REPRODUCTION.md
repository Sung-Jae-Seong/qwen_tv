# Qwen LoRA 실행 재현 가이드

이 문서는 다른 사용자가 이 저장소에서 `finetune.py`와 `inference_lora.py`를 실행할 수 있도록, 가상환경 재현부터 실행 조건까지 한국어로 정리한 문서다.

기준 환경은 아래와 같다.

- Docker 컨테이너: `mjstrong01`
- 베이스 이미지: `mltooling/ml-workspace-minimal:0.13.2`
- Conda 환경: `qwen_test`
- Python 버전: `3.10.20`

## 1. 먼저 알아둘 점

이 저장소는 Python 패키지만 맞춘다고 바로 실행되는 구조가 아니다. 아래 조건이 함께 맞아야 한다.

- Linux x86_64 환경
- NVIDIA GPU
- Hugging Face 모델 다운로드 가능
- `Qwen/Qwen3-VL-8B-Instruct` 접근 가능
- 데이터셋 파일과 비디오 파일이 현재 저장소 기준 상대경로에 존재

현재 수정된 `inference_lora.py`는 보이는 GPU를 자동 감지한다. 따라서 단일 GPU 환경에서도 실행 가능하다. 다만 Qwen3-VL-8B 특성상 VRAM이 부족하면 메모리 부족으로 실패할 수 있다.

## 2. 가상환경 재현 방법

### 방법 A. 가장 정확한 재현

Conda 패키지 빌드까지 최대한 맞추려면 아래 순서를 사용한다.

```bash
cd /home/milab/cvpr2026cctv/team01/minseok/qwen_tv

conda create -n qwen_test --file conda_explicit_qwen_lora.txt
conda activate qwen_test
pip install -r requirements_qwen_lora.txt
```

이 방법은 아래 파일을 사용한다.

- `conda_explicit_qwen_lora.txt`
- `requirements_qwen_lora.txt`

### 방법 B. 더 간단한 재현

```bash
cd /home/milab/cvpr2026cctv/team01/minseok/qwen_tv

conda env create -f environment_qwen_lora.yml
conda activate qwen_test
```

이 방법은 편하지만, 채널 상태나 호스트 환경 차이에 따라 완전 동일 재현이 안 될 수 있다.

## 3. 환경 생성 후 확인

아래 명령으로 Python 버전과 주요 패키지 import가 되는지 확인한다.

```bash
python --version
python -c "import torch, transformers, peft, bitsandbytes, cv2, pandas; print('imports ok')"
```

기대하는 Python 버전은 `3.10.20`이다.

GPU가 정상적으로 보이는지도 확인하는 편이 좋다.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4. Hugging Face 인증

`finetune.py`와 `inference_lora.py`는 저장소 루트의 `.env`를 읽고 `HF_TOKEN`이 있으면 로그인한다.

저장소 루트에 `.env` 파일을 만들고 아래처럼 넣는다.

```env
HF_TOKEN=your_huggingface_token
```

인터넷이 막혀 있거나 토큰이 없으면 모델 다운로드 단계에서 실패할 수 있다.

## 5. 데이터 준비

### 학습에 필요한 파일

기본 경로는 아래다.

- `dataset/sim_dataset/train_sft.jsonl`
- `dataset/sim_dataset/val_sft.jsonl`

또한 각 JSONL 내부의 `video` 경로가 실제 비디오 파일을 가리켜야 한다.

### 추론에 필요한 파일

기본 경로는 아래다.

- `dataset/test_video_path.txt`

또는 `--video_path`, `--video_list`로 직접 지정할 수 있다.

## 6. 학습 실행 방법

현재 `finetune.py`는 기본적으로 LoRA + QLoRA 경로를 사용한다.

- 베이스 모델: `Qwen/Qwen3-VL-8B-Instruct`
- 4-bit quantization: 사용
- LoRA adapter: 사용

또한 현재 수정된 `finetune.py`는 아래 기능을 지원한다.

- `--train_file`, `--val_file` 기본값을 저장소 기준 경로로 자동 사용
- JSONL 내부의 비디오 경로를 현재 작업 디렉터리, 저장소 루트, JSONL 위치 기준으로 자동 보정
- `--output_dir`를 명시하지 않으면 저장소의 `output/qwen3_vl_lora`를 기본 출력 경로로 사용
- 비디오 경로가 깨진 샘플 수를 시작 시 경고로 출력

예시:

```bash
cd /home/milab/cvpr2026cctv/team01/minseok/qwen_tv

nohup python3 finetune.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --train_file dataset/sim_dataset/train_sft.jsonl \
  --val_file dataset/sim_dataset/val_sft.jsonl \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_steps 50 \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_target_modules q_proj v_proj \
  --output_dir ./output/qwen3_vl_8b_lora_qproj_vproj_e3_lr1e4_w50 \
  > finetuning_new.log 2>&1 &
```

학습이 끝나면 LoRA adapter는 아래 경로에 저장된다.

```text
./output/qwen3_vl_8b_lora_qproj_vproj_e3_lr1e4_w50/final_lora_model
```

이 디렉터리 안에 아래 파일이 있어야 한다.

- `adapter_config.json`
- `adapter_model.safetensors` 또는 `adapter_model.bin`

## 7. 추론 실행 방법

`inference_lora.py`는 이제 아래 기능을 지원한다.

- `--lora_model_path`를 생략하면 `output/*/final_lora_model` 중 가장 최근 디렉터리를 자동 탐색
- `--gpu_ids`를 생략하면 현재 보이는 GPU를 자동 사용
- 저장소 루트가 아니어도 상대경로를 최대한 자동 보정

그래도 재현성을 위해 `--lora_model_path`를 직접 넘기는 편이 가장 안전하다.

예시:

```bash
cd /home/milab/cvpr2026cctv/team01/minseok/qwen_tv

CUDA_VISIBLE_DEVICES=0,1 python3 inference_lora.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --lora_model_path ./output/qwen3_vl_8b_lora_qproj_vproj_e3_lr1e4_w50/final_lora_model \
  --gpu_ids 0 1 \
  --source test \
  --experiment_name qwen3_vl_8b_lora_qproj_vproj_test
```

단일 GPU 환경 예시는 아래와 같다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_lora.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --lora_model_path ./output/qwen3_vl_8b_lora_qproj_vproj_e3_lr1e4_w50/final_lora_model \
  --gpu_ids 0 \
  --source test \
  --experiment_name qwen3_vl_8b_lora_single_gpu
```

특정 비디오 하나만 돌리려면 아래처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_lora.py \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --lora_model_path ./output/qwen3_vl_8b_lora_qproj_vproj_e3_lr1e4_w50/final_lora_model \
  --gpu_ids 0 \
  --video_path path/to/video.mp4 \
  --experiment_name single_video_test
```

추론 결과는 `result/` 아래에 저장된다.

## 8. 현재 코드 기준 실행 조건

다른 사용자가 실행하려면 아래 조건을 만족해야 한다.

- `HF_TOKEN` 설정
- 학습용 JSONL과 실제 비디오 파일 존재
- 추론용 LoRA adapter 디렉터리 존재
- GPU 1개 이상이 보이는 환경

저장소 루트에서 실행하는 것이 가장 단순하지만, 현재 수정된 `finetune.py`와 `inference_lora.py`는 저장소 루트가 아니어도 주요 상대경로를 최대한 자동 보정한다.

`inference_lora.py`는 현재 아래 방식으로 동작한다.

- `--gpu_ids`를 주지 않으면 현재 보이는 GPU를 전부 사용한다.
- `--max_memory_per_gpu`를 주면 GPU별 메모리 상한을 설정할 수 있다.
- `--lora_model_path`를 주지 않으면 `output/*/final_lora_model`을 자동 탐색한다.

즉 다른 사용자가 실행할 때는 최소한 아래는 직접 맞춰줘야 한다.

- `CUDA_VISIBLE_DEVICES`
- 필요 시 `--gpu_ids`
- 필요 시 `--lora_model_path`
- `--video_path` 또는 `--video_list`

`finetune.py`는 현재 아래 방식으로 동작한다.

- `--train_file`, `--val_file`를 주지 않으면 저장소 기본 JSONL 경로를 사용한다.
- `--output_dir`를 주지 않으면 저장소의 기본 `output/` 경로를 사용한다.
- JSONL 내부 비디오 경로가 깨져 있으면 시작 시 경고를 출력하고, 해당 샘플은 dummy frame으로 학습된다.

## 9. 다른 사람이 자주 실패하는 지점

### `conda env create -f environment_qwen_lora.yml`는 되는데 실행이 안 되는 경우

주로 아래 원인이다.

- GPU 드라이버 또는 CUDA 런타임 불일치
- `bitsandbytes` 로딩 실패
- Hugging Face 인증 누락
- JSONL 안 비디오 경로 불일치
- LoRA adapter 경로 오타

### `finetune.py`는 도는데 실제로는 영상 없이 학습되는 경우

JSONL 파일은 읽히지만 내부 `video` 경로가 깨져 있으면 해당 샘플은 dummy frame으로 대체된다.

현재 수정된 스크립트는 시작 시 누락된 비디오 개수를 경고로 보여주므로, 이 경고가 나오면 먼저 데이터 경로를 수정해야 한다.

### 단일 GPU에서 메모리 부족이 나는 경우

이제 스크립트 자체는 단일 GPU도 지원한다. 다만 모델 크기 때문에 VRAM이 부족하면 실패할 수 있다.

이 경우 아래를 먼저 시도한다.

- `CUDA_VISIBLE_DEVICES=0`
- `--gpu_ids 0`
- `--max_memory_per_gpu 20GiB` 같은 상한 지정
- 더 큰 GPU 사용 또는 다중 GPU 사용

## 10. 다른 사람에게 전달할 때 같이 주면 좋은 파일

다른 사람에게 전달할 때는 아래 파일을 함께 주는 것이 좋다.

- `environment_qwen_lora.yml`
- `conda_explicit_qwen_lora.txt`
- `requirements_qwen_lora.txt`
- `.env` 예시 파일
- 실행 예시 명령
- 사용할 데이터 경로 규칙

가장 안전한 전달 순서는 아래다.

1. 같은 OS와 NVIDIA GPU 환경 준비
2. `conda_explicit_qwen_lora.txt`로 conda base 환경 생성
3. `requirements_qwen_lora.txt` 설치
4. `.env`에 `HF_TOKEN` 설정
5. 데이터와 LoRA adapter 경로 배치
6. 저장소 루트에서 `finetune.py` 또는 `inference_lora.py` 실행

## 11. 정리

`environment_qwen_lora.yml`만으로는 "패키지 설치"는 재현되지만, 실행 성공까지 자동으로 보장되지는 않는다.

다른 사용자가 실제로 실행 가능하게 하려면 아래 네 가지가 같이 필요하다.

- 재현 가능한 conda/pip 환경
- GPU와 CUDA 조건
- Hugging Face 인증
- 데이터와 LoRA adapter 경로 정합성

이 문서 기준으로 준비하면 현재 저장소 상태에서 다른 사용자가 실행할 가능성을 가장 높일 수 있다.
