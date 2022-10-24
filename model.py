import numpy as np
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("loaded the module successfully")

model = tf.keras.models.load_model("Models/Stroke-model2.h5")
print("model loaded")

encoder = OneHotEncoder(handle_unknown='ignore')
oldX = np.load("Models/train.npy", allow_pickle=True)
print("train file is loaded")


# encode columns with string data
ct = ColumnTransformer([
    ('encoder', encoder, [0, 4, 5, 6])
], remainder='passthrough')

X = ct.fit_transform(oldX)


def predict(gender, age, hypertension, heartdisease, ever_married, work_type, Resident_type, glucose):

    try:
        row = [gender, age, hypertension, heartdisease, ever_married,
               work_type, Resident_type, float(glucose)]

        row = ct.transform([row]).astype("float64")
        print(row)
        res = model.predict([row])
        accuracy = str(max(res[0])*100)
        print(accuracy)
        result = np.argmax(res)

        print(res)

        accuracy = str(max(res[0])*100)
        print(accuracy)
        result = np.argmax(res)
        if result == 0:
            return ["According to our prediction your are healthy and will not get attacked by stroke.", accuracy[:accuracy.index(".")+3]]
        elif result == 1:
            return ["According to our prediction you may likely  have a stroke. Please see a doctor", accuracy[:accuracy.index(".")+3]]
    except Exception as e:
        print(e)
        return ["please enter a valied input", 0]


