from flask import Flask, render_template, request
from prediction import detect_mediapipe

app=Flask(__name__ )


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/mediapipe_detect")
def mediapipe_detect():
    detect_mediapipe()

    return render_template('index.html')

if __name__=='__main__':
    app.run(host="0.0.0.0", port="5000",debug=True)