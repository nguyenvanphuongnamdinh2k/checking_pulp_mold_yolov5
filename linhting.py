import pyodbc
import pandas as pd
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=TL-MTD-JPC02\SQLEXPRESS;'
    'DATABASE=phuong;'
    'UID=sa;'
    'PWD=1234567890'
)
cursor = conn.cursor()

query = '''
        UPDATE test1
        SET department = 'alo'
        WHERE id = 1
        '''
cursor.execute(query)
conn.commit()
cursor.close()
conn.close()




# query = '''
#         INSERT INTO students    (
#                                   name,
#                                   gender,
#                                   city
#                                 )
#                 VALUES
#                     (
#                         'Phuong4444',
#                         'female',
#                         'nam dinh'
#                     );
#                     '''
# cursor.execute(query)
# conn.commit()
# cursor.close()
# conn.close()
