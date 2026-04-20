# Tasks
- [x] Task 1: 扩展ASRConfig配置项
  - [x] SubTask 1.1: 添加whisper_vad_filter、whisper_vad_model配置项
  - [x] SubTask 1.2: 添加funasr_vad_model、funasr_punc_model配置项

- [x] Task 2: 改进Whisper引擎 - 启用VAD和置信度
  - [x] SubTask 2.1: 修改transcribe方法启用VAD过滤
  - [x] SubTask 2.2: 从segment中提取average_log_prob计算置信度

- [x] Task 3: 改进FunASR引擎 - 启用VAD、标点和置信度
  - [x] SubTask 3.1: 修改_ensure_model加载VAD和标点模型
  - [x] SubTask 3.2: 更新transcribe方法使用VAD和标点模型
  - [x] SubTask 3.3: 从结果中提取confidence字段

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
