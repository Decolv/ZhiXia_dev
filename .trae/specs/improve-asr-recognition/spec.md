# ASR识别质量改进方案

## Why
当前ASR系统在语音识别时容易出错，主要表现为：
1. Whisper引擎禁用了VAD（语音活动检测），导致背景噪音和静音段被错误识别
2. FunASR引擎VAD和标点符号模型都设为None，影响识别准确率
3. 两个引擎都没有设置识别置信度，无法评估结果质量

## What Changes
- 启用Whisper引擎的VAD过滤功能
- 为FunASR引擎启用VAD和标点符号模型
- 为Whisper引擎添加置信度计算
- 为FunASR引擎添加置信度提取

## Impact
- Affected specs: ASR引擎配置
- Affected code: 
  - `zhixia/asr/whisper_engine.py`
  - `zhixia/asr/funasr_engine.py`
  - `zhixia/config/settings.py`

## ADDED Requirements

### Requirement: VAD支持
系统 SHALL 在Whisper引擎中启用语音活动检测。

#### Scenario: Whisper引擎启用VAD
- **WHEN** 调用Whisper引擎进行识别
- **THEN** 引擎应当使用VAD过滤静音和噪音段，仅识别语音部分

### Requirement: 标点符号支持
系统 SHALL 为FunASR引擎启用标点符号预测。

#### Scenario: FunASR使用标点模型
- **WHEN** 调用FunASR引擎进行识别
- **THEN** 识别结果应当包含正确的标点符号

### Requirement: 置信度评估
系统 SHALL 为识别结果提供置信度评分。

#### Scenario: Whisper返回置信度
- **WHEN** Whisper引擎完成识别
- **THEN** 结果中的confidence字段应当反映识别的平均置信度

#### Scenario: FunASR返回置信度
- **WHEN** FunASR引擎完成识别
- **THEN** 结果中的confidence字段应当根据识别结果计算置信度

## MODIFIED Requirements

### Requirement: Whisper引擎配置
**修改前**: `vad_filter=False`
**修改后**: `vad_filter=True`, `vad_model="silero_vad"`

### Requirement: FunASR引擎配置
**修改前**: `vad_model=None`, `punc_model=None`
**修改后**: `vad_model="fsmn-vad"`, `punc_model="ct-punc"`

### Requirement: ASRConfig配置结构
**修改前**: 缺少VAD相关配置项
**修改后**: 添加VAD配置项

```python
# 新增配置项
whisper_vad_filter: bool = True
whisper_vad_model: str = "silero_vad"
funasr_vad_model: str = "fsmn-vad"
funasr_punc_model: str = "ct-punc"
```

## REMOVED Requirements

无移除需求。
