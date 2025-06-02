---
title: MiniGPT4-Video：利用交织的视觉-文本token来提升多模态LLM的视频理解能力
tags:
  - 多模态
  - 大语言模型
  - 视频理解
createTime: 2024/10/13 19:54:22
permalink: /article/h1iwpcu5/
---
::: tip 提示
根据遗忘曲线：如果没有记录和回顾，6天后便会忘记75%的内容

阅读笔记正是帮助你记录和回顾的工具，不必拘泥于形式，其核心是：记录、翻看、思考
:::

::: info 信息
论文 [MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens](http://arxiv.org/pdf/2404.03413)     

主页 [https://vision-cair.github.io/MiniGPT4-video/](https://vision-cair.github.io/MiniGPT4-video/)

代码 [https://github.com/Vision-CAIR/MiniGPT4-video](https://github.com/Vision-CAIR/MiniGPT4-video)

摘要: 本文专为视频理解设计了多模态大型语言模型 (LLM)，MiniGPT4-video。该模型能够处理时间视觉和文本数据，使其擅长理解视频的复杂性。MiniGPT4-video 不仅考虑视觉内容，还考虑了文本对话，允许模型有效地回答涉及视觉和文本组件的查询。

:::


## 论文贡献
> 考虑视觉内容的同时，还结合了文本对话，使模型能够有效地回答涉及视觉和文本组件的查询。

## 模型架构
MA-LMM主要包含三个组成部分：1）冻结的视觉编码器提取视觉特征；2）可训练的 Q-Former 在时序上对齐视觉和文本嵌入空间；3）冻结 LLM 的文本解码。


### 训练流水线

#### 大规模图像-文本对预训练
在第一阶段，本文训练了一个线性层，该层将由视觉编码器（如EVACLIP）编码的视觉特征投射到 LLM 的文本空间，使用字幕损失进行训练。论文利用包括 LAION、Conceptual Captions 和 SBU 等图像的综合图像字幕数据集，将视觉特征与 LLM 的输入空间对齐。

#### 大规模视频-文本对预训练
在第二阶段，论文使模型能够理解视频，接受多帧输入。具体来说，论文从每个视频中采样最多 N 帧。在这个阶段，论文使用以下模板中预先定义的提示：

`<s>[INST]<Img><FrameFeature_1><Sub><Subtitle>`

`text_1>... <Img> <FrameFeature_N><Sub><Subtitle>`

`text_N><Instruction></INST>`

采样帧的数量取决于每个语言模型的上下文窗口大小。对于 Llama 2，上下文窗口为 4096 个 token，而Mistral 的上下文窗口为 8192 个 token。在论文的方法中，每个图像用 64 个 token 表示。因此，对于Llama 2，论文设 N=45 帧，视觉内容表示占用 2880 个 token。此外，论文为字幕分配 1000 个 token，剩余的 token 用于模型输出。类似地，对于 Mistral，由于上下文窗口加倍，论文相应地将 N 加倍为 90 帧，以确保与扩展的上下文窗口兼容。
在这个提示中，每个都被视觉骨干编码的采样视频帧替换。如果适用，表示相应帧的字幕，表示从论文预定义的指令集中随机抽取的指令，该指令集包含诸如“简要描述这些视频”等变体形式的指令。论文使用结合了 CMD 和 WebVid 的视频字幕数据进行大规模视频字幕训练。

#### 视频问答指令微调

在这个阶段，论文采用了与第二阶段相同的训练策略，但重点是利用高质量的视频问答数据集进行指令微调。这个微调阶段有助于增强模型理解输入视频并生成精确响应的能力。模板与第二阶段相同，但将 替换为源自 Video-ChatGPT 数据集的一般性问题。

### 实现细节
论文的视觉主干是 EVA-CLIP，权重被冻结。值得注意的是，论文训练了线性投影层，并使用 LoRA 对语言模型进行高效微调。整个模型都训练到了 224×224 像素的一致图像分辨率。

## 总结

MiniGPT4-Video 为视频问答提供了令人信服的解决方案，有效地融合了视频域内的视觉和对话理解。通过直接输入视觉和文本标记，MiniGPT4-Video 支持语言建模模型来掌握视频帧之间的复杂关系，显示出在理解视频内容中的时间动态方面有希望的熟练程度。

限制：尽管 MiniGPT4-Video 取得了显着成就，但它面临着 LLM 上下文窗口的限制。具体来说，当前版本需要 Llama 2 版本的视频长度为 45 帧（相当于每秒 0.5 帧的采样率小于90秒）和 Mistral 版本的 90 帧（相当于不到三分钟）。

未来方向：未来的研究努力将专注于扩展模型处理较长视频的能力从而解决上述限制，进一步提高其在现实场景中的适用性和有效性。