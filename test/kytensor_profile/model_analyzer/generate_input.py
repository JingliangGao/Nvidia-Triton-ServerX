#!/usr/bin/env python3

import json
import numpy as np

# IN0 的形状: [1, 512, 512]
shape = [1, 512, 512]

# 生成随机 uint8 数据
data = np.random.randint(0, 256, size=shape, dtype=np.uint8).tolist()

# 创建 JSON 负载
payload = {
    "IN0": data
}

# 写入文件
with open('input_data.json', 'w') as f:
    json.dump(payload, f)

print("已创建 input_data.json，包含形状 [1, 512, 512] 的随机 uint8 数据")