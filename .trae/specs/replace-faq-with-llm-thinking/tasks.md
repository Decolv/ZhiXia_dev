# Tasks

- [x] Task 1: 扩展DisplayPayload支持思考内容
  - [x] SubTask 1.1: 在`zhixia/display/base.py`中添加`thinking_text`字段到DisplayPayload
  - [x] SubTask 1.2: 更新相关文档字符串

- [x] Task 2: 扩展回调系统支持思考播报
  - [x] SubTask 2.1: 在`zhixia/agent/callbacks.py`的BaseCallbackHandler中添加`on_thinking_start`和`on_thinking_end`方法
  - [x] SubTask 2.2: 在CallbackManager中添加对应的事件广播方法
  - [x] SubTask 2.3: 更新StreamingDisplayHandler以支持思考过程播报

- [x] Task 3: 重构CampusLifeGuideTool移除FAQ
  - [x] SubTask 3.1: 移除`_faq`硬编码字典
  - [x] SubTask 3.2: 保留校园生活知识作为上下文参考数据
  - [x] SubTask 3.3: 修改`_guide`方法使用LLM生成答案
  - [x] SubTask 3.4: 添加思考过程回调触发

- [x] Task 4: 更新NullDisplay支持思考状态
  - [x] SubTask 4.1: 更新`update_thinking`方法支持思考内容显示
  - [x] SubTask 4.2: 更新`show`方法处理thinking_text

- [x] Task 5: 在HostOrchestrator中集成思考播报
  - [x] SubTask 5.1: 创建支持思考播报的自定义CallbackHandler
  - [x] SubTask 5.2: 在`_run_agent`方法中集成思考播报回调
  - [x] SubTask 5.3: 将思考状态同步到Display

- [x] Task 6: 验证和测试
  - [x] SubTask 6.1: 测试思考播报功能正常工作
  - [x] SubTask 6.2: 验证CampusLifeGuideTool能正确生成答案
  - [x] SubTask 6.3: 确保无FAQ硬编码残留

# Task Dependencies
- Task 3 depends on Task 1, Task 2
- Task 5 depends on Task 2, Task 4
- Task 6 depends on Task 3, Task 5
