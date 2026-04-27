# Tasks

- [x] Task 1: 创建网络连通性检测模块
  - [x] SubTask 1.1: 在`zhixia/utils/`创建`network.py`，实现`check_internet_connectivity()`函数
  - [x] SubTask 1.2: 实现网络检测缓存机制，避免频繁检测
  - [x] SubTask 1.3: 添加单元测试验证网络检测功能

- [x] Task 2: 添加云端LLM配置
  - [x] SubTask 2.1: 在`zhixia/config/settings.py`的`LLMConfig`中添加云端API配置项
  - [x] SubTask 2.2: 添加配置项：enable_cloud_fallback、cloud_api_url、cloud_api_key、cloud_model_name

- [x] Task 3: 创建云端LLM引擎
  - [x] SubTask 3.1: 在`zhixia/llm/`创建`cloud_engine.py`，实现CloudLLMEngine类
  - [x] SubTask 3.2: 实现chat()方法支持非流式调用
  - [x] SubTask 3.3: 实现stream_chat()方法支持流式输出
  - [x] SubTask 3.4: 实现set_system_prompt()方法

- [x] Task 4: 实现LLM引擎自动切换
  - [x] SubTask 4.1: 在`zhixia/__main__.py`修改`create_llm_engine()`函数
  - [x] SubTask 4.2: 实现根据网络状态选择引擎的逻辑
  - [x] SubTask 4.3: 添加日志输出显示当前使用的引擎类型

- [x] Task 5: 更新配置文件示例
  - [x] SubTask 5.1: 在`localconfig/localconfig.json`中添加云端LLM配置示例

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
