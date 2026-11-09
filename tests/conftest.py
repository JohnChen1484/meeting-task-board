"""测试永不加载真实 .env；入口模块初始化也只能使用内存数据库。"""
import os

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DB_PATH"] = ":memory:"
os.environ["AI_PROVIDER"] = "mock"
os.environ["COZE_API_TOKEN"] = ""
os.environ["COZE_BOT_ID"] = ""
