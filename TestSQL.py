import pyodbc
from datetime import datetime
server = 'TL-MTD-JPC02\SQLEXPRESS'
username = 'sa'
password = '1234567890'
database = 'phuong'
now = datetime.now()
time = f"table{now.year}{now.month}{now.day}"
# Kết nối đến SQL Server
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
# Tạo đối tượng cursor
cursor = conn.cursor()
# Tên bảng mới
new_table = time
cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", new_table)
exists = cursor.fetchone()[0]
# Kiểm tra xem bảng đã tồn tại hay chưa
if exists:
    print("Bảng đã tồn tại.")
else:
    # Tạo bảng mới
    cursor.execute('''
        CREATE TABLE {} (
            id INT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            age INT,
            department VARCHAR(100)
        )
    '''.format(new_table))
    print("Bảng mới đã được tạo.")
query = f"insert into {new_table} ( id,name,age,department)values(3,'{time}',23,'abc')"
cursor.execute(query)
# Lưu các thay đổi và đóng kết nối
conn.commit()
conn.close()
# cursor.close()
