import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import time

# # 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "https://api-inference.modelscope.cn/v1/"),
    api_key=os.getenv("OPENAI_API_KEY")
)

'''docker run -d \
  -p 7860:7860 \
  -e OPENAI_API_BASE="https://api-inference.modelscope.cn/v1/" \
  -e OPENAI_API_KEY="c3aae464-e3cd-4254-8cbd-f24d42bc21b3" \
  --name pt-assistant \
  pt-assistant
'''

def stream_response(prompt, history):
    """
    流式响应生成函数
    """
    full_answer = ""
    reasoning_content = ""
    has_shown_reasoning = False
    
    # 创建流式请求
    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1',
        messages=[
            {'role': 'system', 'content': '你是一个乐于助人的助手'},
            {'role': 'user', 'content': prompt}
        ],
        stream=True
    )

    # 处理每个流式块
    for chunk in response:
        # 获取思考过程和回答内容
        reasoning = chunk.choices[0].delta.reasoning_content or ""
        answer = chunk.choices[0].delta.content or ""
        
        # 合并思考过程
        if reasoning:
            reasoning_content += reasoning
            if not has_shown_reasoning:
                yield f"[思考过程] {reasoning_content}"
                has_shown_reasoning = True
            else:
                yield f"[思考过程] {reasoning_content}\n\n=== 最终回答 ===\n{full_answer}"
        
        # 合并最终回答
        if answer:
            full_answer += answer
            if has_shown_reasoning:
                yield f"[思考过程] {reasoning_content}\n\n=== 最终回答 ===\n{full_answer}"
            else:
                yield full_answer
        
        # 模拟流式输出的延迟效果
        time.sleep(0.02)

# 创建 Gradio 界面
demo = gr.ChatInterface(
    fn=stream_response,
    title="PT小助手",
    description="输入您的问题",
    examples=["请解释量子计算", "如何做番茄炒蛋？"],
    css="""
    /* 主容器高度设置 */
    .gradio-container {
        height: 120vh !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    
    /* 聊天区域高度调整 */
    #component-0 {
        min-height: 70vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
    }
    
    /* 输入框区域间距 */
    #component-1 {
        padding-top: 2rem !important;
    }
    
    /* 消息气泡样式 */
    .message {
        max-width: 85% !important;
        margin: 1rem auto !important;
        padding: 1.2rem !important;
    }
    
    /* 隐藏底部信息 */
    footer {
        display: none !important;
    }
    """
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        height="800px"  # 设置初始高度
    )
