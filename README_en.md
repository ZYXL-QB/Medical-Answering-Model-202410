# JiYiChat (MedQA)

[中文 README](README.md)

## Environment

Activate the Conda environment used by this project. The complete dependency list is in [requirements.txt](requirements.txt).

```bash
conda activate llama_factory
```

### Reference hardware and system software

| Component | Version or configuration | Notes |
| --- | --- | --- |
| Operating system | Ubuntu 20.04.1 LTS | |
| CPU | Intel Xeon Gold 6338 at 2.00 GHz | Two processors |
| RAM | Samsung DDR4-2933, 64 GB per module | 16 modules, 1,024 GB total |
| GPU | NVIDIA A100-SXM4-80GB | Eight-GPU cluster, with one GPU per training or inference task |
| NVIDIA driver | 535.104.05 | Installation requires administrator access |
| CUDA | 12.2 | Reference installer `cuda_12.2.2_535.104.05_linux.run` |
| cuDNN | 8.9.2.26 | Reference archive `cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz` |
| NCCL | 2.18.3 | Reference archive `nccl_2.18.3-1+cuda12.2_x86_64.txz` |

### Core package versions

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.10.14 | Runtime |
| PyTorch | 2.3.0 | Model training and inference |
| Transformers | 4.40.2 | Model and tokenizer loading |
| Datasets | 2.19.1 | Dataset loading and processing |
| Accelerate | 0.30.0 | Device management |
| PEFT | 0.10.0 | Parameter-efficient fine-tuning |
| TRL | 0.8.6 | Dependency of the `llmtuner` training stack |
| DeepSpeed | 0.14.0 | Training infrastructure |
| bitsandbytes | 0.43.1 | Low-precision model support |
| FlashAttention | 2.5.8 | Attention kernels. Install separately with `pip install flash-attn==2.5.8` if required. |
| vLLM | 0.4.2 | Deployment engine. Install separately with `pip install vllm==0.4.2` if required. |
| llmtuner | 0.7.1.dev0 | LLaMA-Factory training and prediction CLI. The original environment may require installation from source. |

[vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation.html) · [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) · [Original LLaMA-Factory installation walkthrough](https://zhuanlan.zhihu.com/p/695287607)

### API credentials for explanation generation

The API utilities are used to generate explanations with services such as Azure OpenAI, API2D, and Kimi. Set the environment variables for the services you intend to call in `~/.bashrc`. Reload the shell configuration after editing. [MedQA API utilities](data/MedQA/utils/MultiProcessingLLM) · [Moonshot AI](https://platform.moonshot.cn) · [API2D](https://api2d.com)

```bash
# Credentials for Azure OpenAI.
# https://zhishenggpt.openai.azure.com/
export GPT35_AZURE_OPENAI_KEY='xxxx'
# https://zhishenggpt40.openai.azure.com/
export GPT4_AZURE_OPENAI_KEY='xxxx'

# https://openai.api2d.net/v1
export API2D_OPENAI_KEY='xxxx'

# https://dev.iai007.cloud/ai/api/v1
export HEFEI_OPENAI_KEY='xxxx'

# https://platform.moonshot.cn/console/api-keys
export KIMI_OPENAI_KEY='xxxx'
```

```bash
source ~/.bashrc
```

### Download model weights

Use the enterprise ModelScope repositories as the primary source for the following models. The Qwen repository uses separate subdirectories for each version. The Llama repository stores its model files at the root.

Run the commands in this section from the repository root to download the weights directly into this repository's `model/` directory. Each model uses the named subdirectory shown below. The ModelScope commands and Hugging Face alternatives already specify these destinations.

| Model | Enterprise ModelScope URL | Repository subdirectory | Local model directory |
| --- | --- | --- | --- |
| Qwen1.5-14B (base model) | [Browse model files](https://modelscope.cn/models/zyxl2026/Qwen/files) · `https://modelscope.cn/models/zyxl2026/Qwen/files` | `Qwen1.5-14B/` | `model/Qwen1.5-14B/` |
| Qwen2-7B-Instruct | [Browse model files](https://modelscope.cn/models/zyxl2026/Qwen/files) · `https://modelscope.cn/models/zyxl2026/Qwen/files` | `Qwen2-7B-Instruct/` | `model/Qwen2-7B-Instruct/` |
| Meta-Llama-3-8B-Instruct | [Browse model files](https://modelscope.cn/models/zyxl2026/Llama/files) · `https://modelscope.cn/models/zyxl2026/Llama/files` | Repository root | `model/Meta-Llama-3-8B-Instruct/` |

Once the complete weights have been published to these locations, run the relevant commands below from the repository root. If an enterprise repository still contains only its initialization files, use the Hugging Face alternative below.

```bash
python -m pip install modelscope-hub==0.7.1
MODEL_DIR="$PWD/model"
mkdir -p "$MODEL_DIR"

ms-hub download zyxl2026/Qwen --revision master \
    --local-dir "$MODEL_DIR" --include 'Qwen1.5-14B/*'
test -f "$MODEL_DIR/Qwen1.5-14B/config.json" && \
    test -f "$MODEL_DIR/Qwen1.5-14B/model.safetensors.index.json"

ms-hub download zyxl2026/Qwen --revision master \
    --local-dir "$MODEL_DIR" --include 'Qwen2-7B-Instruct/*'
test -f "$MODEL_DIR/Qwen2-7B-Instruct/config.json" && \
    test -f "$MODEL_DIR/Qwen2-7B-Instruct/model.safetensors.index.json"

ms-hub download zyxl2026/Llama --revision master \
    --local-dir "$MODEL_DIR/Meta-Llama-3-8B-Instruct"
test -f "$MODEL_DIR/Meta-Llama-3-8B-Instruct/config.json" && \
    test -f "$MODEL_DIR/Meta-Llama-3-8B-Instruct/model.safetensors.index.json"
```

#### Alternative download from Hugging Face

Use the `huggingface-hub` package included in the project dependencies to download the corresponding official weights. The model pages are [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B) (`https://huggingface.co/Qwen/Qwen1.5-14B`), [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) (`https://huggingface.co/Qwen/Qwen2-7B-Instruct`), and [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (`https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`).

Before downloading Llama, request access and accept the license on the official model page. Run `huggingface-cli login` with the account that has been granted access.

```bash
MODEL_DIR="$PWD/model"
mkdir -p "$MODEL_DIR"

huggingface-cli download Qwen/Qwen1.5-14B \
    --local-dir "$MODEL_DIR/Qwen1.5-14B"
huggingface-cli download Qwen/Qwen2-7B-Instruct \
    --local-dir "$MODEL_DIR/Qwen2-7B-Instruct"
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir "$MODEL_DIR/Meta-Llama-3-8B-Instruct"
```

The quick test uses `Qwen1.5-14B-Chat` and downloads it from the official ModelScope repository into `model/Qwen1.5-14B-Chat/` by default. The enterprise `Qwen1.5-14B/` directory is designated for the base model. See [Quick Test](#quick-test) for Chat downloads and the Hugging Face alternative. [ModelScope Qwen1.5-14B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files) · `https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files`

## MedQA Dataset

The project uses the MedQA question bank. Select the training, validation, and test splits for your experiment before retrieval. [MedQA source repository](https://github.com/jind11/MedQA) · [Dataset archive](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing) · [Chinese introduction](https://zhuanlan.zhihu.com/p/679590312)

## Data Preparation

The commands below document the preprocessing pipeline used for the prepared data.

### Convert textbook text to JSON

Run `txt2json.py` from `data/MedQA/utils` to convert the paragraph-level Chinese, sentence-level Chinese, and English textbooks. The commands split text into chunks using the configured minimum and maximum lengths.

```bash
conda activate llama_factory
cd ./data/MedQA/utils

# Convert source text to JSON.
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_paragraph' --output_json_dir '../data_clean/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_sentence' --output_json_dir '../data_clean/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/en' --output_json_dir '../data_clean/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

### Build vector indexes

Convert the corresponding JSON collections into vector stores. Run these commands from `data/MedQA/utils`.

```bash
# Embed the JSON documents and build vector indexes.
python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_sentence_json' --store_top_path '../data_clean/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_paragraph_json' --store_top_path '../data_clean/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../data_clean/textbooks/en_json' --store_top_path '../data_clean/vector_stores/en'
```

### Retrieve knowledge for MedQA splits

Set `model_path` in the embedding configuration to the model location on your machine. Use `--device_ids="0"` and `--num_process=1` for a single GPU. Set `--query_key_name` to `question` for question-only retrieval or `question_with_options` to include the answer choices. The final preprocessing configuration uses `question_with_options`. [Embedding configuration](data/MedQA/utils/config.py)

```bash
conda activate llama_factory
cd ./data/MedQA/utils

# Retrieve knowledge for the question files.
# Chinese dataset, question-only queries.
# Training split.
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

# Validation split.
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"


# Chinese dataset, question and answer choices.
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../data_clean/vector_stores/zh_paragraph" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"


# English dataset, question-only queries.
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"

python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/test.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question"


# 英文数据集------检索仅使用"question_with_options"
python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/train" \
    --output_json_file="train.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"



python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"


python generate_question_with_knowledges.py \
    --num_process=8 \
    --index_key_name="id" \
    --topk_knowledge=10 \
    --knowledge_threshold=0.65 \
    --input_jsonl_path="../data_clean/questions/US/test.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../data_clean/vector_stores/en" \
    --output_json_dir="../data_clean/questions_with_knowledge/US/test" \
    --output_json_file="test.json" \
    --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"
```

### Generate explanations for the Chinese samples

The following commands use Kimi to add an explanation to each retrieved item. Run the command for each split you want to regenerate after configuring API credentials.

```bash
conda activate llama_factory
cd ./data/MedQA/utils

# Training split.
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train.json" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95

# Test split.
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test.json" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95


# Validation split.
python ./MultiProcessingLLM/multiprocess_using_chatgpt_input_with_prompt_and_json_data.py --do_check \
    --num_process=30 \
    --model_name="moonshot-v1-32k_kimi" \
    --prompt_config_path="./MultiProcessingLLM/prompt_config_of_generate_explain.json" \
    --use_load_raw_json_data_with_process \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/dev/retrive_use_question_with_options/stella-base-zh-v2/dev.json" \
    --output_json_dir="../data_clean/questions_with_knowledge/Mainland/dev/retrive_use_question_with_options/stella-base-zh-v2/dev_with_explain_using_moonshot-v1-32k_kimi" \
    --list_placeholder="list_placeholder" \
    --llm_output_key="chatgpt_explain" \
    --index_key_name="id" \
    --temperature=0.7 \
    --max_tokens=4096 \
    --top_p=0.95
```

### Format and register the Chinese datasets

Convert the prepared training and test records to the LLaMA-Factory conversation format, then register them in `data/dataset_info.json`. Run the following commands from `data/MedQA/utils`.

```bash
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path="../../dataset_info.json" \
    --dataset_name="RAG_MedQA_Mainland_train" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset

# Register the test split.
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path="../../dataset_info.json" \
    --dataset_name="RAG_MedQA_Mainland_test" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset
```

## Fine-Tuning and Evaluation

The training, prediction, and result-processing entry points are in [train_model](train_model).

### Fine-tune a model

Set the variables in the single-GPU training script for your environment. The dataset must be registered in `data/dataset_info.json` with the `sharegpt` format and its `messages` column. The bundled `RAG_MedQA_Mainland_train_500(example)` entry provides an example of the role mapping for `system`, `user`, and `assistant`. [Training script](train_model/run_single_gpu_train_model.sh) · [Dataset registry](data/dataset_info.json)

| Setting | Purpose |
| --- | --- |
| `ROOT_BASE_DIR` | Project or LLaMA-Factory root containing the configured data and training directories. |
| `MODEL_NAME_OR_PATH` | Location of the base model weights. |
| `DATASET` | Name of the registered training dataset. |
| `RESULT_BASE_SAVE_DIR` | Directory for training outputs and checkpoints. |
| `NUM_TRAIN_EPOCHS` | Number of training epochs. |
| `--template` | Conversation template for the selected base model. |
| `--cutoff_len` | Sequence length limit, which affects GPU memory usage. |
| `--per_device_train_batch_size` | Per-device batch size, which affects GPU memory usage. |

Use `llamafactory-cli train -h` to review the other framework options.

```bash
conda activate llama_factory
cd ./train_model
# The first argument is the LoRA rank. Higher ranks require more GPU memory.
# The second argument is the learning rate, such as 0.0001, 0.0005, 0.00005, or 0.00001.
chmod +x run_single_gpu_train_model.sh
CUDA_VISIBLE_DEVICES=0 ./run_single_gpu_train_model.sh 8 0.0001
```

### Run checkpoint-based prediction

The bundled fine-tuning results are stored in `finetune_result/checkpoint-4500/`. This checkpoint contains a LoRA adapter trained on `Llama-3.1-8B-Instruct`. To run inference with `checkpoint-4500`, retrieve the weight files through Git LFS, point `ADAPTER_NAME_OR_PATH` to this directory, set `MODEL_NAME_OR_PATH` to the matching base model weights, and select the corresponding `--template` conversation template. [checkpoint-4500](finetune_result/checkpoint-4500) · [Adapter configuration](finetune_result/checkpoint-4500/adapter_config.json)

Configure `ROOT_BASE_DIR`, `MODEL_NAME_OR_PATH`, `ADAPTER_NAME_OR_PATH`, `DATASET`, and `RESULT_BASE_SAVE_DIR` in the single-GPU prediction script. Set `--template` for the base model. `ADAPTER_NAME_OR_PATH` can include `checkpoint-${checkpoint_steps}` to select a training checkpoint, or it can point to a specific adapter directory. [Prediction script](train_model/run_single_gpu_predict_model.sh)

Register the intended evaluation dataset in `data/dataset_info.json` with the same `sharegpt` message mapping. The historical example registers `RAG_MedQA_Mainland_test_300.json` under `RAG_MedQA_Mainland_test_300`.

```bash
conda activate llama_factory
cd ./train_model
chmod +x run_single_gpu_predict_model.sh
# The first argument is the sampling temperature.
# The second argument is top-p.
# The third argument selects the checkpoint step.
CUDA_VISIBLE_DEVICES=0 ./run_single_gpu_predict_model.sh 0.3 0.7 2000
```

### Export predictions and score accuracy

Pass the evaluation input JSON and the `generated_predictions.jsonl` file produced by LLaMA-Factory to the result formatter. `--output_xlsx_result_path` controls the Excel output location. The paths below show the original deployment layout. Use the corresponding paths for your run. [Result formatter](train_model/format_json_dataset_for_eval_from_jsonl.py)

```bash
conda activate llama_factory
cd ./train_model
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="/home/phd-chen.yirong/scutcyr/LLaMA-Factory/data/MedQA/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"
```

### Evaluate multiple configurations

Set `ROOT_BASE_DIR`, `MODEL_NAME`, and `DATASET` in the launcher to match the single-GPU script. Configure `CUDA_VISIBLE_DEVICES_LIST`, `temperatures`, `top_ps`, and `checkpoints` for the grid of runs you want to evaluate. [Batch prediction script](train_model/run_multiple_gpu_predict_model.sh)

```bash
conda activate llama_factory
cd ./train_model
chmod +x run_single_gpu_predict_model.sh
chmod +x run_multiple_gpu_predict_model.sh
./run_multiple_gpu_predict_model.sh
```

To summarize multiple runs, point the formatter at the prediction directory and select matching subdirectories by prefix.

```bash
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="/home/phd-chen.yirong/scutcyr/LLaMA-Factory/data/MedQA/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/" \
    --input_jsonl_subdir_startwiths="predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train_checkpoint" \
    --output_xlsx_result_path="./results/output_xlsx_result" \
    --output_xlsx_total_result_path="./results/output_xlsx_result/0_all_scores.xlsx"
```

## Quick Test

With the `llama_factory` environment active, run the following two blocks in order from the repository root. The test uses one NVIDIA A800 80GB GPU to run Qwen1.5-14B-Chat on the first 300 prepared MedQA Mainland examples with retrieved knowledge. The test command uses the fixed local path `model/Qwen1.5-14B-Chat/`. [Prepared MedQA data](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json)

### 1. Download the model

Download the official Chat weights from ModelScope into `model/Qwen1.5-14B-Chat/` in this repository. The initial download is approximately 28.35 GB. Rerun the same block to resume an interrupted download. [ModelScope model files](https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files) · `https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files`

```bash
(
set -e
python -m pip install modelscope-hub==0.7.1
mkdir -p "model/Qwen1.5-14B-Chat"
ms-hub download Qwen/Qwen1.5-14B-Chat --revision master \
    --local-dir "$PWD/model/Qwen1.5-14B-Chat"
test -s "$PWD/model/Qwen1.5-14B-Chat/config.json"
test -s "$PWD/model/Qwen1.5-14B-Chat/model.safetensors.index.json"
)
```

### 2. Run the quick test

After the download completes, run the entire block below from the same repository root.

```bash
(
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
QWEN_MODEL="$PWD/model/Qwen1.5-14B-Chat"
if [ ! -s "$QWEN_MODEL/config.json" ] || [ ! -s "$QWEN_MODEL/model.safetensors.index.json" ]; then
    printf '%s\n' 'Download Qwen1.5-14B-Chat in Step 1 above before running the quick test.' >&2
    exit 1
fi
DATASET='RAG_MedQA_Mainland_train_500(example)'
SAMPLE_COUNT=300

if ! command -v llamafactory-cli >/dev/null; then
    printf '%s\n' 'Activate the project llama_factory environment first.' >&2
    exit 1
fi

mkdir -p results
RUN_DIR="$(mktemp -d results/quick_eval_XXXXXX)"

python - "$RUN_DIR/samples.json" "$SAMPLE_COUNT" <<'PY'
import json
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("Select a CUDA GPU for the Qwen1.5-14B-Chat quick test.")

source = Path("data/MedQA/RAG_MedQA_Mainland_train_500(example).json")
sample_count = int(sys.argv[2])
samples = json.loads(source.read_text(encoding="utf-8"))[:sample_count]
if len(samples) != sample_count:
    raise SystemExit(f"The prepared MedQA dataset needs at least {sample_count} examples.")
Path(sys.argv[1]).write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
PY

llamafactory-cli train \
    --model_name_or_path "$QWEN_MODEL" \
    --stage sft --do_predict --predict_with_generate \
    --finetuning_type full --template qwen \
    --dataset "$DATASET" --dataset_dir "$PWD/data" --max_samples "$SAMPLE_COUNT" \
    --cutoff_len 32000 --per_device_eval_batch_size 2 \
    --temperature 0.3 --top_p 0.7 --top_k 20 --max_new_tokens 4096 \
    --output_dir "$RUN_DIR/predictions"

python train_model/format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path "$RUN_DIR/samples.json" \
    --input_jsonl_data_path "$RUN_DIR/predictions/generated_predictions.jsonl" \
    --output_xlsx_result_path "$RUN_DIR/results.xlsx"

python - "$RUN_DIR/results.xlsx" "$SAMPLE_COUNT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

workbook = Path(sys.argv[1])
sample_count = int(sys.argv[2])
if not workbook.is_file():
    raise SystemExit("The result formatter did not create results.xlsx. Check its output above.")
rows = pd.read_excel(workbook, keep_default_na=False)
if len(rows) != sample_count or "LLM原始输出" not in rows.columns:
    raise SystemExit(f"Expected {sample_count} prediction rows with model output.")
if rows["LLM原始输出"].fillna("").astype(str).str.strip().eq("").any():
    raise SystemExit("At least one prediction is empty. Check the prediction log above.")
print(f"Quick test passed: {workbook}")
PY
)
```

Each run saves the 300 input examples, prediction JSONL, and workbook in a new `results/quick_eval_*` directory. `Quick test passed` confirms that the workbook contains 300 non-empty predictions. The samples come from the training examples and serve as a workflow check.

### Alternative download from Hugging Face

If ModelScope downloads are unavailable, download the same Chat version into the same directory, then run Step 2. [Hugging Face model files](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) · `https://huggingface.co/Qwen/Qwen1.5-14B-Chat`

```bash
huggingface-cli download Qwen/Qwen1.5-14B-Chat \
    --local-dir "$PWD/model/Qwen1.5-14B-Chat"
```

### Download result workbooks

The `test_result/` directory contains 18 Excel workbooks with 300 records per file. Downloads are grouped by the model and configuration identifiers in their filenames. [Browse result files](test_result)

Filename prefixes are target percentages. With 300 records, the percentage calculated from the score column can differ because the number of correct answers is an integer.

`basic` denotes the base-model group, `inference` adds retrieval-augmented inference, and `inference_finetuning` also includes fine-tuning. MedQA Mainland is the Chinese dataset; MedQA USMLE is the English dataset and is named `MedQA_USS` in the files.

| Model identifier | Configuration | MedQA Mainland (300 records) | MedQA USMLE (300 records) |
| --- | --- | --- | --- |
| Qwen1.5-14B-Chat | Base model (`basic`) | [Download Excel](test_result/73.29_predict_MedQA_Mainland_test_use_model_Qwen1.5-14B-Chat_stage_basic.xlsx?raw=true) | [Download Excel](test_result/45.01_predict_MedQA_USS_test_use_model_Qwen1.5-14B-Chat_stage_basic.xlsx?raw=true) |
| Qwen1.5-14B-Chat | Retrieval-augmented inference (`inference`) | [Download Excel](test_result/71.98_predict_RAG_MedQA_Mainland_test_use_model_Qwen1.5-14B-Chat_stage_inference.xlsx?raw=true) | [Download Excel](test_result/52.47_predict_RAG_MedQA_USS_test_use_model_Qwen1.5-14B-Chat_stage_inference.xlsx?raw=true) |
| Qwen1.5-14B-Chat | Retrieval-augmented inference with fine-tuning (`inference_finetuning`) | [Download Excel](test_result/80.58_predict_RAG_MedQA_Mainland_test_use_model_Qwen1.5-14B-Chat_stage_inference_finetuning.xlsx?raw=true) | [Download Excel](test_result/54.20_predict_RAG_MedQA_USS_test_use_model_Qwen1.5-14B-Chat_stage_inference_finetuning.xlsx?raw=true) |
| Qwen2-7B-Instruct | Base model (`basic`) | [Download Excel](test_result/70.58_predict_MedQA_Mainland_test_use_model_Qwen2-7B-Instruct_stage_basic.xlsx?raw=true) | [Download Excel](test_result/42.11_predict_MedQA_USS_test_use_model_Qwen2-7B-Instruct_stage_basic.xlsx?raw=true) |
| Qwen2-7B-Instruct | Retrieval-augmented inference (`inference`) | [Download Excel](test_result/71.86_predict_RAG_MedQA_Mainland_test_use_model_Qwen2-7B-Instruct_stage_inference.xlsx?raw=true) | [Download Excel](test_result/46.74_predict_RAG_MedQA_USS_test_use_model_Qwen2-7B-Instruct_stage_inference.xlsx?raw=true) |
| Qwen2-7B-Instruct | Retrieval-augmented inference with fine-tuning (`inference_finetuning`) | [Download Excel](test_result/82.92_predict_RAG_MedQA_Mainland_test_use_model_Qwen2-7B-Instruct_stage_inference_finetuning.xlsx?raw=true) | [Download Excel](test_result/56.25_predict_RAG_MedQA_USS_test_use_model_Qwen2-7B-Instruct_stage_inference_finetuning.xlsx?raw=true) |
| Meta-Llama-3-8B-Instruct | Base model (`basic`) | [Download Excel](test_result/55.81_predict_MedQA_Mainland_test_use_model_Meta-Llama-3-8B-Instruct_stage_basic.xlsx?raw=true) | [Download Excel](test_result/42.97_predict_MedQA_USS_test_use_model_Meta-Llama-3-8B-Instruct_stage_basic.xlsx?raw=true) |
| Meta-Llama-3-8B-Instruct | Retrieval-augmented inference (`inference`) | [Download Excel](test_result/55.14_predict_RAG_MedQA_Mainland_test_use_model_Meta-Llama-3-8B-Instruct_stage_inference.xlsx?raw=true) | [Download Excel](test_result/57.97_predict_RAG_MedQA_USS_test_use_model_Meta-Llama-3-8B-Instruct_stage_inference.xlsx?raw=true) |
| Meta-Llama-3-8B-Instruct | Retrieval-augmented inference with fine-tuning (`inference_finetuning`) | [Download Excel](test_result/76.24_predict_RAG_MedQA_Mainland_test_use_model_Meta-Llama-3-8B-Instruct_stage_inference_finetuning.xlsx?raw=true) | [Download Excel](test_result/64.02_predict_RAG_MedQA_USS_test_use_model_Meta-Llama-3-8B-Instruct_stage_inference_finetuning.xlsx?raw=true) |

Each workbook retains the 37 original columns in `Sheet1`, including predictions, scores, generated responses, and ten retrieved-knowledge groups. `Sheet2` contains the correct count, incorrect count, total, and accuracy.

The workbooks do not record runtime metadata such as GPU hardware, numerical precision, `temperature`, `top_p`, `top_k`, LoRA rank, learning rate, or checkpoint revision. The catalog lists only the identifiers available in the filenames; the quick-test defaults above are not runtime records for these downloads.
