from flask import Flask, render_template, request
from model import predict


app = Flask(__name__)


@app.route('/', methods=["POST"])
def home():
    if request.method == "POST":
        gender = request.form.get("gender")
        ht = request.form.get("ht")
        status = request.form.get("status")
        heart = request.form.get("heart")
        residence = request.form.get("residence")
        work = request.form.get("work")
        age = request.form.get("age")
        Glucose = request.form.get("Glucose")

        res = predict(gender=gender, heartdisease=heart, hypertension=ht, ever_married=status,
                      Resident_type=residence, work_type=work, age=age, glucose=Glucose)

        print(res)
        return render_template("result.html", value=res)

    return render_template('index.html')


@app.route("/result")
def result():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))
