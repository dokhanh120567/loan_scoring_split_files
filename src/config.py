import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Thông tin kết nối MySQL
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'example')
DB_NAME = os.getenv('DB_NAME', 'loan_scoring_db')

# Tạo chuỗi kết nối SQLAlchemy
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
