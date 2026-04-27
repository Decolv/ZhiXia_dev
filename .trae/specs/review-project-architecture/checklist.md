# 架构审查验证清单

- [x] CardLoader卸载卡片时仅注销该卡片注册的工具，不影响其他卡片
- [x] CardLoader挂载失败后能重新检测到变化并重试
- [x] CardLoader卸载卡片后恢复host.card_root到正确值
- [x] PersonaHolder支持多张Skill卡人设叠加，后挂载不覆盖先挂载
- [x] PersonaHolder.clear_overlay(card_name)能正确清除指定卡片的人设
- [x] Live2D眼睛渲染线程读取共享状态时有锁保护
- [x] Live2D眼睛在空闲状态下能自动眨眼
- [x] Live2D渲染线程异常退出时不会导致stop()死锁
- [x] NavUI渲染器不持有锁执行渲染操作
- [x] NavDisplay展示导航时眼睛窗口被隐藏
- [x] pygame.init()和pygame.quit()由全局管理器统一处理
- [x] HostContext的response_processors列表操作是线程安全的
- [x] AgentConfigurator的配置读写是线程安全的
- [x] SkillCard.get_tools()返回List[Tool]而非ToolRegistry
- [x] 所有现有测试通过
- [x] 无新增代码警告或错误
