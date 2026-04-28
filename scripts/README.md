# Scripts 目录

本目录包含项目相关的辅助脚本和工具。

## 脚本说明

### mount_cards.py
ZhiXia 插卡模拟器，用于将 Skill 卡和 Knowledge 卡挂载到指定槽位。

**用法:**
```bash
# 插入 Skill 卡 + Knowledge 卡
python scripts/mount_cards.py --skill skills/your_skill --knowledge knowledge/your_knowledge

# 只插入 Skill 卡
python scripts/mount_cards.py --skill skills/your_skill

# 只插入 Knowledge 卡
python scripts/mount_cards.py --knowledge knowledge/your_knowledge

# 拔卡（清空所有槽位）
python scripts/mount_cards.py --eject
```

### install_fast_tts.sh
Piper TTS 安装脚本，自动安装 Piper TTS 引擎和中文语音模型。

**用法:**
```bash
bash scripts/install_fast_tts.sh
```

### asr_llm_tts_piper.py
> ⚠️ **已废弃** - 此脚本仅为向后兼容保留。

**推荐使用:** `python -m zhixia`

## 注意事项

- 所有脚本应在项目根目录下执行
- Shell 脚本适用于 Linux/ARM 平台（RK3588 等）
- Python 脚本需要 Python 3.9+ 环境
