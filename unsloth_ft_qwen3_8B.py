# 导入必要的库
from unsloth import FastLanguageModel  # 导入Unsloth的快速语言模型加载工具
from transformers import TrainingArguments  # 导入Transformers的训练参数配置
from trl import SFTTrainer, SFTConfig  # 导入TRL库的监督微调训练器和配置
import torch  # 导入PyTorch
from datasets import load_dataset  # 导入数据集加载工具

# 定义一个函数，将数据集转换为对话格式
def generate_conversation(examples):
    """
    将问题和解决方案转换为对话格式
    
    参数:
        examples: 包含'problem'和'generated_solution'字段的数据集样本
    
    返回:
        包含格式化对话的字典
    """
    problems  = examples["problem"]  # 获取问题列表
    solutions = examples["generated_solution"]  # 获取解决方案列表
    conversations = []  # 初始化对话列表
    for problem, solution in zip(problems, solutions):  # 遍历每对问题和解决方案
        conversations.append([
            {"role" : "user",      "content" : problem},  # 用户角色发送问题
            {"role" : "assistant", "content" : solution},  # 助手角色回复解决方案
        ])
    return { "conversations": conversations, }  # 返回对话格式的数据

# 加载模型 - 使用Unsloth的FastLanguageModel加载预训练的Qwen3-8B模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/mnt/data1/D2/unsloth/Qwen3-8B",  # 模型路径
    dtype=torch.float16,  # 使用float16数据类型以减少内存占用
    max_seq_length = 2048,  # 设置最大序列长度为2048
    load_in_4bit=True  # 使用4位量化加载模型，进一步减少显存占用
)

# 配置PEFT（参数高效微调）模型 - 使用LoRA方法
model = FastLanguageModel.get_peft_model(
    model,  # 基础模型
    r = 32,  # LoRA的秩，可以选择>0的任意数字，建议值为8、16、32、64、128。秩越高，可训练参数越多，效果可能更好但需要更多资源
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # 目标模块，指定应用LoRA的层
    lora_alpha = 32,  # LoRA的缩放参数，最佳选择是等于秩或秩的两倍
    lora_dropout = 0,  # LoRA的dropout率，0是优化选择
    bias = "none",  # 偏置项处理方式，"none"是优化选择
    use_gradient_checkpointing = "unsloth",  # 使用"unsloth"梯度检查点，可减少30%的VRAM使用，支持2倍大的批量大小
    random_state = 3407,  # 随机种子，确保结果可复现
    use_rslora = False,  # 是否使用秩稳定化LoRA，这里不使用
    loftq_config = None,  # LoftQ配置，这里不使用
)

# 数据集加载和处理部分

# # 加载小数据集（注释掉的代码）
# dataset = load_dataset("/mnt/data1/D2/imdb", split="train[:100]")

# 加载大数据集
# 1. 加载推理型数据集 - 数学推理数据
reasoning_dataset = load_dataset("/mnt/data1/D2/unsloth/OpenMathReasoning-mini", split = "cot")
# 2. 加载非推理型数据集 - 通用对话数据
non_reasoning_dataset = load_dataset("/mnt/data1/D2/mlabonne/FineTome-100k", split = "train")

# 将推理数据集转换为对话格式并应用聊天模板
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched = True)["conversations"],  # 使用前面定义的函数转换为对话格式
    tokenize = False,  # 不进行分词，只应用模板
)

# 导入ShareGPT格式标准化工具
from unsloth.chat_templates import standardize_sharegpt
# 标准化非推理数据集格式
dataset = standardize_sharegpt(non_reasoning_dataset)

# 将非推理数据集应用聊天模板
non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)

# 打印两个数据集的大小
print(len(reasoning_conversations))
print(len(non_reasoning_conversations))

# 设置推理数据在最终数据集中的比例
chat_percentage = 0.75  # 推理数据占75%

# 导入pandas进行数据处理
import pandas as pd

# 从非推理数据集中采样一部分，使得推理数据占最终数据集的75%
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations) * (1.0 - chat_percentage)),  # 计算需要的非推理数据量
    random_state = 2407,  # 设置随机种子确保可复现
)

# 合并两个数据集
data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"  # 设置数据列名为"text"

# 将pandas数据转换为Hugging Face数据集格式
from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
# 打乱数据集
combined_dataset = combined_dataset.shuffle(seed = 3407)

# 配置训练参数 - 使用SFTTrainer进行监督微调
trainer = SFTTrainer(
    model=model,  # 要训练的模型
    tokenizer=tokenizer,  # 分词器
    train_dataset=combined_dataset,  # 训练数据集
    dataset_text_field="text",  # 数据集中文本字段的名称
    max_seq_length=512,  # 训练时使用的最大序列长度
    args = SFTConfig(  # SFT配置参数
        dataset_text_field = "text",  # 数据集中文本字段的名称
        per_device_train_batch_size = 2,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 4,  # 梯度累积步数，用于模拟更大的批次大小
        warmup_steps = 5,  # 学习率预热步数
        # num_train_epochs = 1,  # 训练轮数，这里注释掉了，使用max_steps代替
        max_steps = 30,  # 最大训练步数
        learning_rate = 2e-4,  # 学习率，对于长时间训练可以降低到2e-5
        logging_steps = 1,  # 日志记录间隔步数
        optim = "adamw_8bit",  # 优化器，使用8位AdamW
        weight_decay = 0.01,  # 权重衰减率
        lr_scheduler_type = "linear",  # 学习率调度器类型
        seed = 3407,  # 随机种子
        report_to = "none",  # 报告工具，可以设置为"wandb"等
    ),
)

# 开始训练
trainer.train()

"""
推理测试

下面使用Unsloth原生推理功能测试模型。根据`Qwen-3`团队的建议：
- 对于推理任务，推荐参数设置为：`temperature = 0.6, top_p = 0.95, top_k = 20`
- 对于普通聊天任务，推荐参数设置为：`temperature = 0.7, top_p = 0.8, top_k = 20`
"""
# 测试1：普通聊天模式（不启用思考过程）
messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}  # 用户提问：求解(x + 2)^2 = 0
]
# 应用聊天模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,  # 不进行分词
    add_generation_prompt = True,  # 添加生成提示，这是生成时必须的
    enable_thinking = False,  # 禁用思考过程
)

# 导入文本流式输出工具
from transformers import TextStreamer
# 生成回答
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),  # 将输入文本转换为模型输入格式并移至GPU
    max_new_tokens = 256,  # 最大生成256个新token，可以增加以获得更长的输出
    temperature = 0.7, top_p = 0.8, top_k = 20,  # 普通聊天模式的参数设置
    streamer = TextStreamer(tokenizer, skip_prompt = True),  # 使用流式输出，跳过提示部分
)

# 测试2：启用思考过程的推理模式
messages = [
    {"role" : "user", "content" : "Solve (x + 2)^2 = 0."}  # 同样的问题
]
# 应用聊天模板，但这次启用思考过程
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,  # 添加生成提示
    enable_thinking = True,  # 启用思考过程
)

# 生成回答，使用推理任务的参数设置
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024,  # 增加最大生成token数，因为思考过程可能更长
    temperature = 0.6, top_p = 0.95, top_k = 20,  # 推理任务的参数设置
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

"""
保存模型

有两种方式保存最终的LoRA适配器：
1. 使用Huggingface的`push_to_hub`进行在线保存
2. 使用`save_pretrained`进行本地保存
"""
# 定义保存路径，使用具有辨识度的文件夹名称
import os
from datetime import datetime

# 创建一个具有高辨识度的文件夹名称，包含模型名称、微调方法、数据特点和日期
save_dir_base = "/mnt/data1/D2/unsloth/"
model_name = "Qwen3-8B-Unsloth-ReasonChat-" + datetime.now().strftime("%Y%m%d")
lora_save_path = os.path.join(save_dir_base, model_name + "-LoRA")
merged_save_path = os.path.join(save_dir_base, model_name + "-Merged")

# 确保保存目录存在
os.makedirs(lora_save_path, exist_ok=True)
os.makedirs(merged_save_path, exist_ok=True)

# 本地保存LoRA适配器
print(f"正在将LoRA模型保存到: {lora_save_path}")
model.save_pretrained(lora_save_path)  # 保存模型
tokenizer.save_pretrained(lora_save_path)  # 保存分词器

# 在线保存（注释掉的代码）
# model.push_to_hub("your_name/lora_model", token = "...") # 在线保存模型
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # 在线保存分词器

"""
合并为16位精度模型

将LoRA权重与原始模型合并，并以16位精度保存，便于部署和推理
"""
# 保存合并后的16位精度模型
print(f"正在将合并后的模型保存到: {merged_save_path}")
model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)

print(f"""
模型保存完成！
- LoRA适配器保存在: {lora_save_path}
- 合并后的模型保存在: {merged_save_path}
""")
