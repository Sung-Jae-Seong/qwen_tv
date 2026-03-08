import os
import ast
import json
import time
import multiprocessing as mp
from datetime import datetime

import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText
from safetensors.torch import load_file

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def parse_in_json(llm_response, video_path):
    import re

    default = {
        "time": 0.0,
        "coordinate": [[0, 0], [0, 0]],
        "type": "head-on",
        "why": "invalid response format"
    }

    temp = None
    raw = llm_response

    # 1) markdown 코드블록 제거: ```json ... ``` 또는 ``` ... ```
    cleaned = llm_response.strip()
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if code_block:
        cleaned = code_block.group(1).strip()

    # 2) JSON 객체 추출: 첫 번째 { ... } 매칭
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # 시도 1: json.loads
        try:
            temp = json.loads(json_str)
        except json.JSONDecodeError:
            pass
        # 시도 2: ast.literal_eval
        if temp is None:
            try:
                temp = ast.literal_eval(json_str)
            except Exception:
                pass

    # 3) 원본 그대로 시도
    if temp is None:
        try:
            temp = json.loads(llm_response)
        except Exception:
            pass
    if temp is None:
        try:
            temp = ast.literal_eval(llm_response)
        except Exception:
            pass

    if not isinstance(temp, dict):
        temp = default

    # raw 응답도 함께 저장 (디버깅용)
    save_dir = os.path.join("result", "parsed_json")
    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, f"{video_name}.json")
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"failed to save parsed json for {video_path}: {e}", flush=True)

    # raw 응답 별도 저장
    raw_dir = os.path.join("result", "raw_responses")
    os.makedirs(raw_dir, exist_ok=True)
    try:
        with open(os.path.join(raw_dir, f"{video_name}.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
    except Exception:
        pass

    return temp


def save_submission(submission, experiment_name, description_text=""):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join("result", f"{experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    submission_path = os.path.join(save_dir, "submission.csv")
    description_path = os.path.join(save_dir, "description.txt")

    submission.to_csv(submission_path, index=False, lineterminator="\n")

    with open(description_path, "w", encoding="utf-8") as f:
        f.write(description_text)

    print(f"saved to {save_dir}")


def make_submission(results):
    rows = []

    for result in results:
        try:
            video_path = result.get("video_path", "")
            path = "videos/" + video_path.split("/videos/")[-1] if "/videos/" in video_path else video_path

            accident_time = result.get("time", 0.0)
            coordinate = result.get("coordinate", [[0, 0], [0, 0]])
            (x1, y1), (x2, y2) = coordinate

            # 좌표를 1000으로 나눠서 0~1 비율로 변환
            center_x = round(((x1 + x2) / 2) / 1000, 3)
            center_y = round(((y1 + y2) / 2) / 1000, 3)
            accident_type = result.get("type", "unknown")

            rows.append({
                "path": path,
                "accident_time": round(accident_time, 2),
                "center_x": center_x,
                "center_y": center_y,
                "type": accident_type
            })
        except Exception as e:
            print(f"failed to make submission for result: {result}, error: {e}")
            continue

    submission = pd.DataFrame(
        rows,
        columns=["path", "accident_time", "center_x", "center_y", "type"]
    )
    return submission


class VideoInferenceVLM:
    def video_inference(self, video_path, prompt, max_new_tokens=128):
        raise NotImplementedError("Subclasses should implement this method.")


class Qwen3VLLoRAInference(VideoInferenceVLM):
    def __init__(
        self, 
        model_id="Qwen/Qwen3-VL-8b-Instruct",
        lora_model_path="output/final_lora_model/final_lora_model",
        gpu_ids=None
    ):
        if gpu_ids is None:
            gpu_ids = [0, 1]
        self.lora_model_path = lora_model_path
        
        # GPU 0, 1 메모리 분배 설정
        max_memory = {i: "22GiB" for i in gpu_ids}
        print(f"Loading base model from: {model_id} on GPUs {gpu_ids}")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            print("✓ Base model loaded successfully with flash_attention_2!")
        except Exception as e:
            print(f"Error loading model with flash_attention_2: {e}")
            print("Retrying without flash attention...")
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    trust_remote_code=True
                )
                print("✓ Base model loaded successfully (without flash attention)!")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
        
        # 모델의 첫 번째 파라미터가 있는 디바이스를 input device로 사용
        self.device = next(self.model.parameters()).device
        print(f"Input device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # 수동으로 LoRA 가중치 병합
        self._apply_lora_manually()

    def _apply_lora_manually(self):
        print(f"Applying LoRA weights manually from: {self.lora_model_path}")
        config_path = os.path.join(self.lora_model_path, "adapter_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LoRA config not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            lora_config = json.load(f)

        r = lora_config.get("r", 8)
        lora_alpha = lora_config.get("lora_alpha", 8)
        scaling = lora_alpha / r

        # 가중치 파일 탐색 (safetensors 우선)
        safetensors_path = os.path.join(self.lora_model_path, "adapter_model.safetensors")
        bin_path = os.path.join(self.lora_model_path, "adapter_model.bin")

        if os.path.exists(safetensors_path):
            lora_state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            lora_state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("LoRA weights (.safetensors or .bin) not found.")

        # LoRA A, B 행렬 분류
        lora_A = {}
        lora_B = {}
        for key, tensor in lora_state_dict.items():
            # PEFT 저장 시 추가되는 접두사 제거
            base_key = key.replace("base_model.model.", "")
            
            if "lora_A" in base_key:
                module_name = base_key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                lora_A[module_name] = tensor
            elif "lora_B" in base_key:
                module_name = base_key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                lora_B[module_name] = tensor

        model_state_dict = self.model.state_dict()
        updated_count = 0

        # W_new = W_base + (B @ A) * scaling
        with torch.no_grad():
            for module_name in lora_A.keys():
                if module_name in lora_B:
                    target_weight_key = f"{module_name}.weight"
                    
                    if target_weight_key in model_state_dict:
                        W = model_state_dict[target_weight_key]
                        
                        # 연산을 위해 동일한 데이터 타입과 장치로 이동 (행렬 연산 시 float32로 캐스팅하여 정밀도 유지)
                        A = lora_A[module_name].to(device=W.device, dtype=torch.float32)
                        B = lora_B[module_name].to(device=W.device, dtype=torch.float32)
                        
                        delta_W = (B @ A) * scaling
                        
                        # 계산된 가중치를 기존 가중치에 더함 (원래 dtype으로 복원)
                        W.add_(delta_W.to(W.dtype))
                        updated_count += 1
                    else:
                        print(f"Warning: Target layer '{target_weight_key}' not found in base model.")

        print(f"✓ Successfully merged {updated_count} LoRA weight matrices into base model.")


    def video_inference(self, video_path, prompt, max_new_tokens=128):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text


def main():
    mp.set_start_method("spawn", force=True)

    instruction = (
        "return the time when the collision occurs, "
        "and return the collision bounding box with left-top and right-bottom coordinates. "
        "And return the collision type. The collision type includes: head-on, rear-end, sideswipe, single, "
        "and t-bone collisions. "
        "Head-on is defined as a collision where the front ends of two vehicles hit each other. "
        "Rear-end is defined as a collision where the front end of one vehicle hits the rear end of another vehicle. "
        "Sideswipe is defined as a slight collision where the sides of two vehicles hit each other. "
        "Single is defined as an accident that involves only one vehicle, such as a vehicle hitting a stationary object or a vehicle losing control and crashing without colliding with another vehicle. "
        "T-bone is defined as a collision where the front end of one vehicle hits the side of another vehicle, forming a 'T' shape."
    )

    return_format = """
please return the result in JSON format only, not markdown.
here is the JSON format:
{
    "time": exact time in seconds, the temporal location of the video where collision occured,
    "coordinate": left-top and right-bottom, the position of bounding box on the video frame that contains the collision,
    "type": choose and return one of the following [head-on, rear-end, sideswipe, single, t-bone],
    "why": explain why did you return that time, coordinate and type.
}
---
example:
{
    "time": seconds,
    "coordinate": [
        [x1, y1],
        [x2, y2]
    ],
    "type": "one of the following [head-on, rear-end, sideswipe, single, t-bone]",
    "why": "..."
}
"""

    test_path_file = "dataset/test_video_path.txt"
    # 절대 경로로 지정된 LoRA 모델 경로 적용
    lora_model_path = "/workspace/minseok/qwen_tv/output/final_lora_model/final_lora_model"
    prompt = instruction + return_format

    with open(test_path_file, "r", encoding="utf-8") as f:
        video_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(video_paths)} videos")
    print(f"Processing first 4 videos with LoRA model...")
    print(f"LoRA path: {lora_model_path}\n")
    
    # Initialize inference once
    try:
        inference = Qwen3VLLoRAInference(
            lora_model_path=lora_model_path,
            gpu_ids=[0, 1]
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    results = []
    total_start_time = time.time()

    # Process videos sequentially
    for i, video_path in enumerate(video_paths): #video_paths[:4] --- IGNORE ---
        try:
            print(f"\n{'='*60}")
            print(f"Processing {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
            print(f"{'='*60}")
            output = inference.video_inference(
                video_path,
                prompt,
                max_new_tokens=128
            )
            output_json = parse_in_json(output[0], video_path)
            output_json["video_path"] = video_path
            results.append(output_json)
            print(f"\n✓ Result:")
            print(json.dumps(output_json, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"\n✗ Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()

    total_end_time = time.time()
    total_elapsed_seconds = total_end_time - total_start_time

    print(f"\n{'='*60}")
    print(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds")
    print(f"Successfully processed {len(results)} videos")
    print(f"{'='*60}")

    # 결과를 submission CSV로 저장
    if results:
        submission = make_submission(results)
        save_submission(submission, "lora_inference", f"LoRA model: {lora_model_path}\nTotal videos: {len(results)}\nElapsed: {total_elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()