from flask import Flask

app = Flask(__name__)

@app.route('/')     # 바로 다음 나오는 정의된 함수와 연결됨(route의 인자로 경로값 설정)
def hello333():     # 함수명은 아무렇게나 해도 상관없다
    return "<h1>hello junseop world</h1>"

@app.route('/ping', methods=['GET'])
def ping():
    return "<h1>pong</h1>"

if __name__ == '__main__':
    app.run(host="192.168.0.154", port=5000, debug=False)   