# 데이터셋 경로 처리
- utils.py 를 본인 워크스페이스의 데이터셋 경로로 수정하여 실행
- 데이터셋 경로에 test_video_path.txt, train_video_path.txt 생성됨
- 이 두 파일이 있어야지 추론 시 데이터 경로를 잘 잡음

# qwen3.5 9b 실행

## 목표 및 실행 예시는 qwen35_README.md 참고

### conda 환경 설정
- setup_qwen_moe_host.sh 실행
    - 관련 파일들 (참고용)
        - QWEN_INFERENCE_EXECUTION_CHECKLIST.md
        - QWEN_MOE_HOST_ENV_SETUP.md
        - RUN_OPENAI_INFER_SETUP.txt
        - environment_qwen_moe_host.yml


### 9b 추론 코드
- inference_qwen35_9b_base_video_classification.py
- qwen35_README.md 참고

# qwen3 8b 실행
### conda 환경 설정
- setup_qwen_test.sh 실행
    - 관련 파일들 (참고용)
        - INFERENCE_PY_QWEN_TEST_SETUP.md
        - environment_qwen_test.yml

### 8b-Instruct 추론 코드
- inference.py

* 만약 hf_token 요구하면 example.env를 참고하여 .env 파일 만들기 & 본인 hf_token 기입하기