from flask import Flask
from flask import redirect

app = Flask(__name__)

@app.route('/naver')
def index():
    return redirect('http://www.naver.com')

@app.route('/google')
def index2():
    return redirect('http://www.google.com')

@app.route('/daum')
def index3():
    return redirect('http://www.daum.net')
if __name__ =="__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)