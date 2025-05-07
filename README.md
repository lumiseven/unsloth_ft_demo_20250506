# 使用 Unsloth 微调 Qwen3-8B 详细教程

本教程将详细介绍如何使用 Unsloth 库对 Qwen3-8B 大语言模型进行高效微调。Unsloth 是一个专为大语言模型优化的微调工具，可以显著提高训练速度并降低显存占用。

## 目录

- [环境准备](#环境准备)
- [Unsloth 简介](#unsloth-简介)
- [模型加载](#模型加载)
- [LoRA 参数配置](#lora-参数配置)
- [数据集准备](#数据集准备)
- [训练配置](#训练配置)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [模型保存](#模型保存)
- [常见问题](#常见问题)

## 环境准备

首先，确保您已安装必要的依赖库：

```bash
pip install unsloth transformers trl torch datasets pandas
```

对于 Qwen3-8B 模型，建议使用至少 16GB 显存的 GPU。如果显存有限，可以考虑使用 4 位量化加载模型。

## Unsloth 简介

Unsloth 是一个专为大语言模型优化的微调工具，具有以下特点：

1. **更快的训练速度**：比标准 LoRA 训练快 2-5 倍
2. **更低的显存占用**：使用 "unsloth" 梯度检查点可减少 30% 的 VRAM 使用
3. **支持多种模型**：包括 Llama、Mistral、Qwen、Phi 等主流大语言模型
4. **优化的 LoRA 实现**：提供了更高效的 LoRA 微调方法

## 模型加载

使用 Unsloth 的 `FastLanguageModel` 加载 Qwen3-8B 模型：

```python
from unsloth import FastLanguageModel
import torch

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen3-8B",  # 模型路径，可以是本地路径或Hugging Face模型ID
    dtype=torch.float16,    # 使用float16数据类型以减少内存占用
    max_seq_length=2048,    # 设置最大序列长度
    load_in_4bit=True       # 使用4位量化加载模型，进一步减少显存占用
)
```

参数说明：
- `model_name`：模型名称或路径
- `dtype`：模型权重的数据类型，通常使用 `torch.float16` 或 `torch.bfloat16`
- `max_seq_length`：模型支持的最大序列长度，影响注意力计算的上下文窗口大小
- `load_in_4bit`：是否使用 4 位量化加载模型，可大幅减少显存占用

## LoRA 参数配置

配置 LoRA (Low-Rank Adaptation) 参数以进行高效微调：

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                   # LoRA的秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",],  # 目标模块
    lora_alpha=32,          # LoRA的缩放参数
    lora_dropout=0,         # LoRA的dropout率
    bias="none",            # 偏置项处理方式
    use_gradient_checkpointing="unsloth",  # 使用Unsloth梯度检查点
    random_state=3407,      # 随机种子
    use_rslora=False,       # 是否使用秩稳定化LoRA
    loftq_config=None,      # LoftQ配置
)
```

参数详解：

- **r**：LoRA 的秩，决定了可训练参数的数量。推荐值为 8、16、32、64、128。秩越高，可训练参数越多，效果可能更好但需要更多资源。
- **target_modules**：指定应用 LoRA 的层。对于 Qwen3 模型，通常包括注意力层和 MLP 层的投影矩阵。
- **lora_alpha**：LoRA 的缩放参数，最佳选择是等于秩或秩的两倍。
- **lora_dropout**：LoRA 的 dropout 率，0 是优化选择，但可以设置为 0.1 等值以增加正则化。
- **bias**："none" 是优化选择，也可以是 "all" 或 "lora_only"。
- **use_gradient_checkpointing**："unsloth" 是 Unsloth 特有的优化，可减少 30% 的 VRAM 使用。
- **random_state**：随机种子，确保结果可复现。
- **use_rslora**：是否使用秩稳定化 LoRA，可以提高训练稳定性。
- **loftq_config**：LoftQ 配置，用于进一步优化量化训练。

## 数据集准备

数据集准备是微调的关键步骤。以下是一个混合推理和对话数据集的示例：

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd

# 定义对话格式转换函数
def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ])
    return {"conversations": conversations}

# 加载推理数据集和对话数据集
reasoning_dataset = load_dataset("path/to/reasoning/dataset", split="train")
chat_dataset = load_dataset("path/to/chat/dataset", split="train")

# 转换为对话格式
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
    tokenize=False,
)

# 标准化对话数据集格式
chat_dataset = standardize_sharegpt(chat_dataset)
chat_conversations = tokenizer.apply_chat_template(
    chat_dataset["conversations"],
    tokenize=False,
)

# 设置数据集比例并合并
reasoning_ratio = 0.75
chat_subset = pd.Series(chat_conversations).sample(
    int(len(reasoning_conversations) * (1.0 - reasoning_ratio)),
    random_state=42,
)

# 合并数据集
data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(chat_subset)
])
data.name = "text"

# 转换为Hugging Face数据集格式并打乱
from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed=42)
```

数据集准备的关键点：

1. **数据格式**：确保数据符合对话格式，包含用户和助手的角色标识
2. **数据多样性**：混合不同类型的数据（如推理、对话）可以提高模型的通用能力
3. **数据平衡**：根据任务需求调整不同类型数据的比例
4. **聊天模板**：使用 `apply_chat_template` 应用模型特定的聊天模板

## 训练配置

使用 `SFTTrainer` 配置训练参数：

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,  # 或使用 num_train_epochs
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)
```

参数详解：

- **per_device_train_batch_size**：每个设备的训练批次大小，根据显存大小调整
- **gradient_accumulation_steps**：梯度累积步数，用于模拟更大的批次大小
- **warmup_steps**：学习率预热步数，有助于稳定训练
- **max_steps** / **num_train_epochs**：训练步数或轮数，根据数据集大小和训练需求选择
- **learning_rate**：学习率，对于长时间训练可以降低到 2e-5
- **logging_steps**：日志记录间隔步数
- **optim**：优化器，"adamw_8bit" 是 8 位精度的 AdamW 优化器，可减少显存占用
- **weight_decay**：权重衰减率，用于正则化
- **lr_scheduler_type**：学习率调度器类型，"linear" 表示线性衰减
- **seed**：随机种子，确保结果可复现
- **report_to**：报告工具，可以设置为 "wandb" 等进行训练监控

## 模型训练

启动训练过程：

```python
# 开始训练
trainer.train()
```

训练过程中，您可以观察以下指标：

- **loss**：训练损失，应该随着训练进行而降低
- **learning_rate**：学习率，根据调度器设置变化
- **epoch**：当前训练轮数
- **step**：当前训练步数

## 模型推理

训练完成后，可以使用以下代码进行模型推理测试：

```python
# 准备测试输入
messages = [
    {"role": "user", "content": "Solve (x + 2)^2 = 0."}
]

# 普通聊天模式（不启用思考过程）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # 生成时必须添加
    enable_thinking=False,  # 禁用思考过程
)

from transformers import TextStreamer
# 生成回答
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7, top_p=0.8, top_k=20,  # 普通聊天模式参数
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# 推理模式（启用思考过程）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # 启用思考过程
)

# 生成回答
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=1024,
    temperature=0.6, top_p=0.95, top_k=20,  # 推理模式参数
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

推理参数说明：

- **普通聊天模式**：`temperature=0.7, top_p=0.8, top_k=20`
- **推理任务模式**：`temperature=0.6, top_p=0.95, top_k=20`
- **max_new_tokens**：控制生成文本的最大长度
- **enable_thinking**：是否启用思考过程，对于复杂推理任务建议启用

## 模型保存

有两种方式保存微调后的模型：

1. **保存 LoRA 适配器**（推荐用于进一步微调）：

```python
# 本地保存
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 在线保存到 Hugging Face Hub
model.push_to_hub("your_name/lora_model", token="...")
tokenizer.push_to_hub("your_name/lora_model", token="...")
```

2. **保存合并后的模型**（推荐用于部署）：

```python
# 保存合并后的16位精度模型
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
```

保存方法说明：

- **save_pretrained**：仅保存 LoRA 适配器权重，体积小，但使用时需要原始模型
- **save_pretrained_merged**：将 LoRA 权重合并到原始模型中，体积大，但可独立使用
- **save_method**：
  - "merged_16bit"：16 位精度，平衡大小和精度
  - "merged_8bit"：8 位精度，更小的体积
  - "merged_4bit"：4 位精度，最小的体积

## 常见问题

### 1. 显存不足怎么办？

- 减小 `max_seq_length`
- 减小 `per_device_train_batch_size` 并增加 `gradient_accumulation_steps`
- 使用 `load_in_4bit=True` 加载模型
- 减小 LoRA 的秩 `r`
- 使用 `use_gradient_checkpointing="unsloth"` 优化显存使用

### 2. 训练速度慢怎么办？

- 确保使用了 Unsloth 的优化功能
- 减小 `max_seq_length`
- 增大 `per_device_train_batch_size`（如果显存允许）
- 使用更快的 GPU 或多 GPU 训练

### 3. 模型效果不好怎么办？

- 增加训练数据量和多样性
- 增大 LoRA 的秩 `r`
- 调整学习率，尝试更小的值如 2e-5
- 增加训练步数或轮数
- 尝试不同的 `target_modules` 组合

### 4. 如何在多 GPU 上训练？

在 `SFTConfig` 中添加以下参数：

```python
args = SFTConfig(
    # 其他参数...
    device_map="auto",  # 自动分配设备
    # 或明确指定
    # device_map={"": 0}  # 使用第一个GPU
)
```

### 5. 如何加载保存的模型？

加载 LoRA 适配器：

```python
from unsloth import FastLanguageModel
import torch

# 加载原始模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen3-8B",
    dtype=torch.float16,
    max_seq_length=2048,
    load_in_4bit=True
)

# 加载LoRA适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",],
    lora_alpha=32,
    # 其他参数...
)

# 加载保存的权重
model.load_adapter("lora_model")
```

加载合并后的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 直接加载合并后的模型
model = AutoModelForCausalLM.from_pretrained("model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("model")
```

---

通过本教程，您应该能够成功使用 Unsloth 对 Qwen3-8B 模型进行高效微调。如果遇到问题，请参考 [Unsloth 官方文档](https://github.com/unslothai/unsloth) 或相关社区获取更多帮助。
