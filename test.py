from flask import Flask, render_template, request
from chatbot import predict_class,get_response
import json

intents = json.loads(open( 'intents_copy.json' , encoding="utf8").read())


app = Flask(__name__)
app.static_folder ='static'

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    ints= predict_class(userText )
    return get_response(ints,intents)

if __name__ == "__main__":
    app.run()
    
