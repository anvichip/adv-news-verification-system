from flask import Flask,request,render_template
from verify_news import verify
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_list = [str(x) for x in request.form.values()]
    answer,final_article = verify(input_list)
    return render_template('conclusion.html',conclusion = answer,union = final_article)


if __name__ == "__main__":
    app.run(debug=True)