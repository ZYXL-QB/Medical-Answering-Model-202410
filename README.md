# JiYiChat（MedQA）



## 项目环境依赖
项目主要依赖于如下conda环境
```bash
conda activate llama_factory
```
完整的包依赖参考[./requirements.txt](./requirements.txt)

### 硬件与系统级驱动依赖

其中硬件与驱动依赖：
| 软件包/硬件        | 版本号/配置     | 备注   |
| ------------ | -------    | ------- | 
| 系统       | Ubuntu 20.04.1 LTS |    |
| CPU       | Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz | 2块 |
| 内存       |  Samsung, DRAM, 2933 MT/s, 64 GB        | 16条64GB内存条，共计1024GB      |
| GPU       | A100-SXM4-80GB | 8卡集群。每个训练/推理任务使用1卡      |
| NVIDIA Driver | 535.104.05 | NVIDIA-Linux-x86_64-535.104.05.run，需要管理员安装 |
| CUDA       | 12.2         | cuda_12.2.2_535.104.05_linux.run，需要管理员安装 |
| cudnn       | 8.9.2.26 | cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz，需要管理员安装 |
| nccl  | 2.18.3 | nccl_2.18.3-1+cuda12.2_x86_64.txz，需要管理员安装 |

### 软件包版本依赖

其中，核心软件包版本：
| 软件包        | 版本号     | 备注   |
| :------------ | :-------   | :------- |
| python       | 3.10.14   | 本项目必需   |
| torch        | 2.3.0     | 本项目必需    |
| transformers | 4.40.2    | 本项目必需    |
| datasets     | 2.19.1    | 本项目必需    |
| accelerate   | 0.30.0    | 本项目必需    |
| peft         | 0.10.0    | 本项目必需    |
| trl          | 0.8.6     | 本项目无关，为RLHF时使用，本项目没有用到，但是llmtuner库需要依赖于这个 | 
| deepspeed    | 0.14.0    | 本项目必需 |
| bitsandbytes | 0.43.1    | 本项目必需 |
| flash-attn   | 2.5.8     | 本项目必需，可能需要手动单独运行```pip install flash-attn==2.5.8```，安装非常耗时，且可能存在不成功的可能，安装难度较大  |
| vllm         | 0.4.2     | 本项目必需，核心部署库，可能需要单独运行```pip install vllm==0.4.2```，安装非常耗时，且可能存在不成功的可能，安装难度较大，安装请参考：[https://docs.vllm.ai/en/latest/getting_started/installation.html](https://docs.vllm.ai/en/latest/getting_started/installation.html) |
| llmtuner     | 0.7.1.dev0 | **本项目必需，核心训练库，安装时可能需要用git clone的方式安装，参考[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**，安装时可以参考教程[https://zhuanlan.zhihu.com/p/695287607](https://zhuanlan.zhihu.com/p/695287607) |


### 调用大模型接口的环境变量依赖
本项目调用大模型（gpt-4-1106-preview、kimichat等）的部分依赖于[./data/BigFiveCPED/utils/MultiProcessingLLM](./data/BigFiveCPED/utils/MultiProcessingLLM)。
你可以从[https://platform.moonshot.cn](https://platform.moonshot.cn)、[https://api2d.com](https://api2d.com)当中注册账号申请api并使用LLM

本部分主要用于调用KimiChat API时用到。需要调用那些大模型，就设置那些API。
在使用前，需要配置环境变量，如下所示：
* 设置环境变量
```bash
vim ~/.bashrc
```
在其中增加以下语句并且保存

```bash
# 设置访问https://portal.azure.com提供的ChatGPT接口服务的api_key变量
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
然后刷新环境配置
```bash
source ~/.bashrc
```


### 大模型参数下载(推荐)

优先使用企业魔搭仓库获取以下三个模型。Qwen 系列通过仓库内的子目录区分版本，Llama 权重位于单独仓库的根目录。

| 模型 | 企业魔搭地址 | 仓库内目录 | 本地模型目录 |
| --- | --- | --- | --- |
| Qwen1.5-14B（基座版） | [打开模型文件](https://modelscope.cn/models/zyxl2026/Qwen/files) · `https://modelscope.cn/models/zyxl2026/Qwen/files` | `Qwen1.5-14B/` | `models/Qwen1.5-14B/` |
| Qwen2-7B-Instruct | [打开模型文件](https://modelscope.cn/models/zyxl2026/Qwen/files) · `https://modelscope.cn/models/zyxl2026/Qwen/files` | `Qwen2-7B-Instruct/` | `models/Qwen2-7B-Instruct/` |
| Meta-Llama-3-8B-Instruct | [打开模型文件](https://modelscope.cn/models/zyxl2026/Llama/files) · `https://modelscope.cn/models/zyxl2026/Llama/files` | 仓库根目录 | `models/Meta-Llama-3-8B-Instruct/` |

企业目录发布完整权重后，可在仓库根目录执行下方命令，按需选择对应模型下载。若企业目录仍只有初始化文件，可使用后面的 Hugging Face 备选方案。

```bash
python -m pip install modelscope-hub==0.7.1
MODEL_DIR="$PWD/models"
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

#### Hugging Face 备选方案

使用环境依赖中已安装的 `huggingface-hub` 下载同一版本的官方权重。三个模型的地址分别为 [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B)（`https://huggingface.co/Qwen/Qwen1.5-14B`）、[Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)（`https://huggingface.co/Qwen/Qwen2-7B-Instruct`）和 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)（`https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`）。

下载 Llama 前，在官方模型页完成访问申请和协议确认，并运行 `huggingface-cli login` 登录获批账号。

```bash
MODEL_DIR="$PWD/models"
mkdir -p "$MODEL_DIR"

huggingface-cli download Qwen/Qwen1.5-14B \
    --local-dir "$MODEL_DIR/Qwen1.5-14B"
huggingface-cli download Qwen/Qwen2-7B-Instruct \
    --local-dir "$MODEL_DIR/Qwen2-7B-Instruct"
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir "$MODEL_DIR/Meta-Llama-3-8B-Instruct"
```

快速测试使用 `Qwen1.5-14B-Chat`，默认从魔搭官方仓库下载。企业目录 `Qwen1.5-14B/` 对应基座版。Chat 权重的下载及 Hugging Face 备选命令见[快速测试](#快速测试)。 [魔搭 Qwen1.5-14B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files) · `https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files`

## MedQA数据集
* 数据集来源：[https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)
* 下载链接：[https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing)
* 中文介绍：[https://zhuanlan.zhihu.com/p/679590312](https://zhuanlan.zhihu.com/p/679590312)


## 数据集预处理

### 将txt文本转换为json文本(已完成，无需重复操作)
```bash
conda activate llama_factory
cd ./data/MedQA/utils

# txt转换为json
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_paragraph' --output_json_dir '../data_clean/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/zh_sentence' --output_json_dir '../data_clean/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../data_clean/textbooks/en' --output_json_dir '../data_clean/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

### 读取json文本完成向量化(已完成，无需重复操作)

```bash
# 根据json文件转换为向量
python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_sentence_json' --store_top_path '../data_clean/vector_stores/zh_sentence' 
python -u vector_store.py --input_json_dir '../data_clean/textbooks/zh_paragraph_json' --store_top_path '../data_clean/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../data_clean/textbooks/en_json' --store_top_path '../data_clean/vector_stores/en'
```

### 执行训练集和测试集的检索，构建标准指令集(已完成，无需重复操作)

* 特别地，```--device_ids```参数指定使用的显卡，如果只有一张显卡，就设置```--device_ids="0"```,```--num_process=1```。
* 特别地，```--query_key_name```指定用于检索的query，可以选择"question"——仅用问题进行检索，或者"question_with_options"——问题与选项拼接在一起检索。最终方案使用了"question_with_options"。

* 特别地，需要先打开[./data/MedQA/utils/config.py](./data/MedQA/utils/config.py)修改"model_path"为服务器中检索Embedding模型的实际路径。


```bash
conda activate llama_factory
cd ./data/MedQA/utils

# 对json文件进行检索
# 中文数据集------检索仅使用"question"
# 训练集
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

# 验证集
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


# 中文数据集------检索使用"question_with_options"
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


# 英文数据集------检索仅使用"question"
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


### 生成中文数据集的含有解析的文本(已完成，无需重复操作)

```bash
conda activate llama_factory
cd ./data/MedQA/utils

# 训练集
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

# 测试集
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


# 验证集
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

### 构建中文数据集的标准数据集(已完成，无需重复操作)
```bash
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/train/retrive_use_question_with_options/stella-base-zh-v2/train_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_train" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset

# 构建测试集
python format_json_dataset_for_training_llm.py \
    --input_json_data_path="../data_clean/questions_with_knowledge/Mainland/test/retrive_use_question_with_options/stella-base-zh-v2/test_with_explain_using_moonshot-v1-32k_kimi" \
    --dataset_info_path=""../../dataset_info.json \
    --dataset_name="RAG_MedQA_Mainland_test" \
    --dataset_info_relative_dir="MedQA/" \
    --output_json_data_dir="../" \
    --do_register_dataset
```

## 微调与测试
微调与测试的指令主要存储在[./train_model](./train_model)下面。

### 微调

微调前，需要修改[./train_model/run_single_gpu_train_model.sh](./train_model/run_single_gpu_train_model.sh)的如下变量

| 变量名 | 含义或用途                                                                                                                                                                                              |
|:------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ROOT_BASE_DIR | 项目的根目录，目前为```/home/chenyirong/MedQA_Train```，请根据实际情况修改，其子目录包含data、train_model等                                                                                                                     |
| MODEL_NAME_OR_PATH | **重要**，为基座模型的路径，请修改为实际使用的基座模型的服务器中的存储路径！！！                                                                                                                                                         |
| DATASET | 训练所使用的数据集，在/home/zhuxiang001/llamafactory/LLaMA-Factory/data/ [./data/dataset_info.json](./data/dataset_info.json)当中定义，这是在调用```./data/MedQA/utils/format_json_dataset_for_training_llm.py```文件时注册的 |
修改格式为：
"RAG_MedQA_Mainland_train_500(example)": {
        "file_name": "RAG_MedQA_Mainland_train_500(example).json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }


| RESULT_BASE_SAVE_DIR | 训练结果的保存路径，请根据实际情况设置 |
| NUM_TRAIN_EPOCHS | 训练所设置的epoches数目，请根据实际情况设置 |
| **llamafactory-cli train的框架参数** | |
| --template | 需要根据基座模型修改，否则会报错，请参考项目[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)中的src/llmtuner/data/template.py文件设置。 |
| --cutoff_len | 截断长度，根据实际情况设置，这个由数据集决定，同时会线性影响显存占用量！ |
| --per_device_train_batch_size | 训练时的batch_size，这个会线性影响显存占用 |
| 其他参数 | 参考```llamafactory-cli train -h```或者详细查看[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |


```bash
conda activate llama_factory
cd ./train_model
# 第一个参数指定了lora_rank的值，为整数，值越大，占用的显存越大
# 第二个参数指定了微调时的学习率，一般取0.0001、0.0005、0.00005、0.00001等
chmod +x run_single_gpu_train_model.sh
CUDA_VISIBLE_DEVICES=0 ./run_single_gpu_train_model.sh 8 0.0001
```





### 测试

测试前，需要修改[./train_model/run_single_gpu_predict_model.sh](./train_model/run_single_gpu_predict_model.sh)的如下变量

| 变量名 | 含义或用途                                                                                                                                                                       |
|:------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ROOT_BASE_DIR | 项目的根目录，目前为```/home/chenyirong/MedQA_Train```，请根据实际情况修改，其子目录包含data、train_model等                                                                                              |
| MODEL_NAME_OR_PATH | **重要**，为基座模型的路径，请修改为实际使用的基座模型的服务器中的存储路径！！！                                                                                                                                  |
| ADAPTER_NAME_OR_PATH | Lora微调时保存模型参数的路径，特别地，其设置格式包含了```checkpoint-${checkpoint_steps}```，你可以设置为绝对路径从而忽略传入的checkpoint_steps                                                                         |
| DATASET | 测试所使用的数据集，需要在/home/zhuxiang001/llamafactory/LLaMA-Factory/data/  (./data/dataset_info.json)当中定义，这是在调用```./data/MedQA/utils/format_json_dataset_for_training_llm.py```文件时注册的 |
修改格式为：
"RAG_MedQA_Mainland_test_300": {
        "file_name": "RAG_MedQA_Mainland_test_300.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }

| RESULT_BASE_SAVE_DIR | 推理结果的保存路径，请根据实际情况设置                                                                                                                                                         |
| **llamafactory-cli train的框架参数** |                                                                                                                                                                             |
| --template | 需要根据基座模型修改，否则会报错，请参考项目[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)中的src/llmtuner/data/template.py文件设置。                              |
| 其他参数 | 参考```llamafactory-cli train -h```或者详细查看[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)                                                 |



```bash
conda activate llama_factory
cd ./train_model
chmod +x run_single_gpu_predict_model.sh
# 第一个参数指定了推理时的temperature
# 第二个参数指定了推理时的top_p
# 第三个参数指定了选用的checkpoint，根据实际情况选取
CUDA_VISIBLE_DEVICES=0 ./run_single_gpu_predict_model.sh 0.3 0.7 2000
```

### 测试结果标准化与统计准确率

参考如下命令，其中，--input_json_data_path指定对应的json数据集路径，与测试集对应；--input_jsonl_data_path指定推理时的结果保存路径；--output_xlsx_result_path指定输出的.xlsx结果保存路径
```bash
conda activate llama_factory
cd ./train_model
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="/home/phd-chen.yirong/scutcyr/LLaMA-Factory/data/MedQA/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20/generated_predictions.jsonl" \
    --output_xlsx_result_path="./results/predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train-checkpoint-4500_with_temperature_0.7_top_p_0.8_top_k_20.xlsx"

```


### 批量测试
主要调用[./train_model/run_multiple_gpu_predict_model.sh](./train_model/run_multiple_gpu_predict_model.sh)文件进行批量测试，需要修改ROOT_BASE_DIR、MODEL_NAME、DATASET与[./train_model/run_single_gpu_predict_model.sh](./train_model/run_single_gpu_predict_model.sh)对应；CUDA_VISIBLE_DEVICES_LIST、temperatures、top_ps、checkpoints需要根据实际情况设置。



```bash
conda activate llama_factory
cd ./train_model
chmod +x run_single_gpu_predict_model.sh
chmod +x run_multiple_gpu_predict_model.sh
./run_multiple_gpu_predict_model.sh
```


批量统计得分：
```bash
python format_json_dataset_for_eval_from_jsonl.py \
    --input_json_data_path="/home/phd-chen.yirong/scutcyr/LLaMA-Factory/data/MedQA/RAG_MedQA_Mainland_test_300.json" \
    --input_jsonl_data_path="./results/" \
    --input_jsonl_subdir_startwiths="predict_RAG_MedQA_Mainland_test_300_use_jiyichat_on_RAG_MedQA_Mainland_train_checkpoint" \
    --output_xlsx_result_path="./results/output_xlsx_result" \
    --output_xlsx_total_result_path="./results/output_xlsx_result/0_all_scores.xlsx"
```






[English README](README_en.md)

## 快速测试

在已激活的 `llama_factory` 环境中，从仓库根目录复制执行整段命令。测试使用单张 NVIDIA A800 80GB GPU（CUDA），以 Qwen1.5-14B-Chat 对预处理 MedQA Mainland 文件的前 300 条带知识样本运行预测。默认从魔搭官方仓库下载到 `models/Qwen1.5-14B-Chat/` 后，使用本地模型目录运行预测。已有本地 Chat 权重时，先将 `QWEN_MODEL` 设为该模型目录。 [魔搭 Qwen1.5-14B-Chat](https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files) · `https://modelscope.cn/models/Qwen/Qwen1.5-14B-Chat/files` · [预处理 MedQA 数据](data/MedQA/RAG_MedQA_Mainland_train_500%28example%29.json)

```bash
(
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

if [ -z "${QWEN_MODEL:-}" ]; then
    QWEN_MODEL="$PWD/models/Qwen1.5-14B-Chat"
    python -m pip install modelscope-hub==0.7.1
    ms-hub download Qwen/Qwen1.5-14B-Chat --revision master \
        --local-dir "$QWEN_MODEL"
fi
test -f "$QWEN_MODEL/config.json"
test -f "$QWEN_MODEL/model.safetensors.index.json"

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

每次运行都会在 `results/quick_eval_*` 下写入 300 条样本、预测 JSONL 和 Excel。输出 `Quick test passed` 表示 Excel 包含 300 行非空预测。这些样本来自训练示例，用于验证推理流程。

### Hugging Face 备选方案

魔搭下载不可用时，可先通过 Hugging Face 下载同一 Chat 版本并设置本地模型目录，再执行上面的整段快速测试命令。 [Hugging Face Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) · `https://huggingface.co/Qwen/Qwen1.5-14B-Chat`

```bash
huggingface-cli download Qwen/Qwen1.5-14B-Chat \
    --local-dir "$PWD/models/Qwen1.5-14B-Chat"
export QWEN_MODEL="$PWD/models/Qwen1.5-14B-Chat"
```
