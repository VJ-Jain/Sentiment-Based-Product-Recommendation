from flask import Flask, request, render_template
from model import get_product_recommendations 

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_name = request.form.get('user_name')
    (product_recommendations, issue) = get_product_recommendations(user_name)

    if issue == "Error":
        return render_template('index.html', result_found = False, user_name = user_name)
    else:
        return render_template('index.html', result_found = True,  user_name = user_name, product_recommendations = product_recommendations)

if __name__ == "__main__":
    app.run(debug=True, port=5000)