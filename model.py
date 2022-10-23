import numpy as np
from tflite_runtime.interpreter import Interpreter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown='ignore')
oldX = np.load("Models\\train.npy", allow_pickle=True)

interpreter = Interpreter(model_path='Models\\model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# encode columns with string data
ct = ColumnTransformer([
    ('encoder', encoder, [0, 4, 5, 6])
], remainder='passthrough')

X = ct.fit_transform(oldX)


def predict(gender, age, hypertension, heartdisease, ever_married, work_type, Resident_type, glucose):

    try:
        row = [gender, age, hypertension, heartdisease, ever_married,
               work_type, Resident_type, float(glucose)]

        row = ct.transform([row]).astype("float32")
        print(row)


        input_data = row

        # Invoke the model on the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the result 
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)




        accuracy = str(max(output_data[0])*100)
        print(accuracy)
        result = np.argmax(output_data)
        if result == 0:
            return ["According to our prediction your are healthy and will not get attacked by stroke.", accuracy[:accuracy.index(".")+3]]
        elif result == 1:
            return ["According to our prediction you may likely  have a stroke. Please see a doctor", accuracy[:accuracy.index(".")+3]]
    except Exception as e:
        print(e)
        return ["please enter a valied input", 0]


predict("Male", 19, 0, 0, "No", "children", "Urban", 139)
