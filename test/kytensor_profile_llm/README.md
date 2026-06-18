# kytensor_profile_llm
用于分析 Kytensor 大語言模型性能的工具

## 环境配置
1. 构建Python虚拟环境
```bash
python3 -m venv llm_env
source llm_env/bin/activate
```

2. 安装Python依赖包
```bash
pip3 install -r requirements.txt
```

## 运行案例
查询部署模型的名称，即`ls /opt/appdata/kylin-ai/model-repository/`. 测试模型，执行命令：
```bash
python3 llm_test.py -m <your-deployed-model>  # llm_Qwen-2.5-3b_1.0
```