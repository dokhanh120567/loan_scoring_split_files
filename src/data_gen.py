import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from faker import Faker
import sqlalchemy
from sqlalchemy import text
from config import DB_URL

# 👉 Fake data generator
fake = Faker()

# 👉 Kết nối DB
engine = sqlalchemy.create_engine(DB_URL)

# 👉 Tạo bảng mới (đã đổi tên thành loan_requests)
CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS loan_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    annual_income FLOAT,
    credit_score INT,
    age INT,
    gender VARCHAR(6),
    num_of_credit_accounts INT,
    num_of_late_payments INT,
    debt_to_income_ratio FLOAT,
    region VARCHAR(20),
    label INT
);
'''

# 👉 Câu lệnh insert
INSERT_SQL = '''
INSERT INTO loan_requests (
    annual_income, credit_score, age, gender,
    num_of_credit_accounts, num_of_late_payments,
    debt_to_income_ratio, region, label
) VALUES (
    :annual_income, :credit_score, :age, :gender,
    :num_of_credit_accounts, :num_of_late_payments,
    :debt_to_income_ratio, :region, :label
);
'''

# 👉 Các giá trị giả lập
REGIONS = ['Urban', 'Suburban', 'Rural']
GENDERS = ['Male', 'Female']
NUM_ROWS = 20000  # Tổng số record muốn sinh

def generate_record():
    annual_income = round(random.uniform(20000, 200000), 2)
    credit_score = random.randint(300, 850)
    age = random.randint(21, 70)
    gender = random.choice(GENDERS)
    num_of_credit_accounts = random.randint(1, 10)
    num_of_late_payments = random.randint(0, 5)
    debt_to_income_ratio = round(random.uniform(0.1, 0.8), 2)
    region = random.choice(REGIONS)
    label = int((credit_score > 600) and (debt_to_income_ratio < 0.5))
    return {
        'annual_income': annual_income,
        'credit_score': credit_score,
        'age': age,
        'gender': gender,
        'num_of_credit_accounts': num_of_credit_accounts,
        'num_of_late_payments': num_of_late_payments,
        'debt_to_income_ratio': debt_to_income_ratio,
        'region': region,
        'label': label
    }

def main():
    print(f"🔧 Connecting to DB: {DB_URL}")
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        print("✅ Created table loan_requests (if not exists)")

        batch_size = 1000
        rows = []

        for i in range(1, NUM_ROWS + 1):
            rows.append(generate_record())

            if i % batch_size == 0:
                conn.execute(text(INSERT_SQL), rows)
                print(f"📦 Inserted {i} rows...")
                rows = []

        # Phần dư còn lại chưa insert
        if rows:
            conn.execute(text(INSERT_SQL), rows)
            print(f"📦 Inserted final {len(rows)} rows.")

    print("🎉 Data generation completed.")

if __name__ == '__main__':
    main()
