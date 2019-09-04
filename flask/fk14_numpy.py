# 모듈을 불러옵니다.
import pymssql as ms
import numpy as np

# 데잉터베이스에 연결합니다.
conn = ms.connect(server='localhost', user='bit', password='1234', database='bitdb')

# 커서를 만듭니다.
cursor = conn.cursor()

# 커서에 쿼리를 입력해 실행시킵니다.
cursor.execute('SELECT * FROM iris2;')

# 한행을 가져옵니다.
rows = cursor.fetchall()
## tuple to numpy
## -> numpy.asarray(tuple) / numpy.array(tuple)
np_rows = np.asarray(rows)

print(np_rows)
conn.close()

np.save('test_iris2.npy', np_rows)

