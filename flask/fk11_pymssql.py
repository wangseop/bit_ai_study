# 모듈을 불러옵니다.
import pymssql as ms
# import numpy as np

# 데잉터베이스에 연결합니다.
conn = ms.connect(server='localhost', user='bit', password='1234', database='bitdb')

# 커서를 만듭니다.
cursor = conn.cursor()

# 커서에 쿼리를 입력해 실행시킵니다.
cursor.execute('SELECT top(100) * FROM train;')

# 한행을 가져옵니다.
row = cursor.fetchone()
# print(type(row))  # tuple

# 행이 존재할 때까지, 하나씩 행을 증가시키면서 컬럼을 문자로 출력합니다.
while row:
    # print('첫컬럼=%s, 둘컬럼=%s' % (row[0], row[1]))
    print(row)
    row = cursor.fetchone()

# 연결을 닫습니다.
conn.close()