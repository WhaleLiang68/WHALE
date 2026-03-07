import requests
import json
# 替换为你的实际API凭证
API_TOKEN = "你的SheerID访问令牌"
PROGRAM_ID = "你的项目ID"
BASE_URL = "https://services.sheerid.com/rest/v2/"

# 步骤1 — 初始化身份验证流程
def start_verification():
    # 请求头配置
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",  # 鉴权令牌
        "Content-Type": "application/json"       # 数据格式为JSON
    }
    # 请求参数
    payload = {
        "programId": PROGRAM_ID
    }
    # 发起POST请求，创建验证会话
    response = requests.post(f"{BASE_URL}verification", headers=headers, json=payload)
    data = response.json()
    print("验证流程已启动：", json.dumps(data, indent=2))
    return data

# 步骤2 — 提交用户信息（以学生身份验证为例）
def submit_info(submission_url, info):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    # 向返回的提交地址，上传用户信息
    response = requests.post(submission_url, headers=headers, json=info)
    print("信息提交响应：", response.json())
    return response.json()

# 代码示例调用
if __name__ == "__main__":
    # 启动验证，获取会话数据
    verification_data = start_verification()
    # 提取会话中的用户信息提交地址
    submission_url = verification_data["submissionUrl"]
    # 构造待提交的用户信息
    user_info = {
        "firstName": "艾丽斯",
        "lastName": "阮",
        "email": "alice@example.com",
        "university": "华盛顿大学"
    }
    # 提交用户信息
    submit_info(submission_url, user_info)