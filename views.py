from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input
from keras.applications import ResNet50
import os
import numpy as np
import cv2
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt



global resnet_algorithm, X_train, X_test, y_train, y_test
class_labels = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 
    'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
    'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
]

severity_type=['High severe','Medium Severe','Low Severe']

def DiseasePrediction(request):
    if request.method == 'GET':
        return render(request, 'DiseasePrediction.html', {})

def load_resnet_model():
    with open('model/RESNET50_model.json', "r") as json_file:
        model_json = json_file.read()
    classifier = model_from_json(model_json)
    classifier.load_weights("model/RESNET50_model_weights.h5")
    return classifier

def load_r2unet_model():
    with open("model/r2unet.json", "r") as json_file:
        model_json = json_file.read()
    runet = model_from_json(model_json)
    runet.load_weights("model/r2unet_weights.h5")
    return runet

def preprocess_for_classification(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = np.array(img).reshape(1, 32, 32, 3)
    img = img.astype('float32') / 255.0
    return img

def preprocess_for_segmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1)
    img = (img - 127.0) / 127.0
    return img


def runRESNET50(request):
    if request.method == 'GET':
        global resnet_algorithm, X_train, X_test, y_train, y_test
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        print(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        if os.path.exists('model/RESNET50_model.json'):
            with open('model/RESNET50_model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                classifier = model_from_json(loaded_model_json)
            json_file.close()    
            classifier.load_weights("model/model_weights.h5")
            classifier._make_predict_function()       
        else:
            classifier = Sequential()
            classifier = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            classifier.add(ResNet50(weights='imagenet'))
            
            classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            classifier.add(Flatten())
            classifier.add(Dense(output_dim = 256, activation = 'relu'))
            classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
            print(classifier.summary())
            #now compiling the model
            classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            hist = classifier.fit(X, Y, batch_size=8, epochs=50, shuffle=True, verbose=2, validation_data=(X_test, y_test))
            classifier.save_weights('model/RESNET50_model_weights.h5')            
            model_json = classifier.to_json()
            with open("model/RESNET50_model.json", "w") as json_file:
                json_file.write(model_json)
            json_file.close()
        print(classifier.summary())    
        resnet_algorithm = classifier    
        #conf_matrix, output = RESNETtestPrediction("RESNET50 Skin Disease Classification",classifier,X_test,y_test)
        context= {'data':output}
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,9])
        plt.title("RESNET50 Skin Disease Classification Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        return render(request, 'ViewOutput.html', context)       



def DiseasePredictionAction(request):
    if request.method == 'POST' and request.FILES['t1']:
        uploaded_file = request.FILES['t1']
        fs = FileSystemStorage()
        file_path = fs.save('SkinDiseaseApp/static/samples/test.png', uploaded_file)
        full_file_path = os.path.join(fs.location, file_path)

        classifier = load_resnet_model()
        runet = load_r2unet_model()

        # Preprocess image for classification
        img_for_classification = preprocess_for_classification(full_file_path)
        preds = classifier.predict(img_for_classification)
        predicted_label = np.argmax(preds)
        disease_name = class_labels[predicted_label]

        # Preprocess image for segmentation
        img_for_segmentation = preprocess_for_segmentation(full_file_path)

        # Get segmented image
        segmented_img = runet.predict(img_for_segmentation)[0]
        segmented_img = cv2.resize(segmented_img, (300, 300), interpolation=cv2.INTER_CUBIC)
        binary_img = cv2.inRange(segmented_img, 1, 255)  # Convert to binary (1 for white, 0 for black)

        # Calculate affected area
        white_pixels = np.count_nonzero(binary_img)
        pixel_size_in_mm = 0.5
        area_in_mm2 = white_pixels * (pixel_size_in_mm ** 2)
        print("Affected Area (mm²):", area_in_mm2)

        # Load and annotate the original image
        original_img = cv2.imread(full_file_path)
        original_img = cv2.resize(original_img, (300, 300))

        cv2.putText(original_img, f'Disease: {disease_name}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if area_in_mm2 >= 2000:
            severity = 'High Severity'
            cv2.putText(original_img, f'Severity: {severity_type[0]}', (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            precautions_page = 'precautions1.html'
        elif 700 <= area_in_mm2 <= 2000:
            severity = 'Medium Severity'
            cv2.putText(original_img, f'Severity: {severity_type[1]}', (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            precautions_page = 'precautions2.html'
        else:
            severity = 'Low Severity'
            cv2.putText(original_img, f'Severity: {severity_type[2]}', (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            precautions_page = None  # No precautions page needed for low severity

        # Display images
        cv2.imshow("Original Image with Disease Name", original_img)
        cv2.imshow("Segmented Image (R2-U-Net)", segmented_img)
        cv2.waitKey(0)  # Wait for user to close the images
        cv2.destroyAllWindows()  # Close all image windows

        # Redirect to the precautions page after closing images
        if precautions_page:
            return render(request, precautions_page, {'data': f'Precautions for {severity}'})

        return render(request, 'DiseasePrediction.html', {'disease_name': disease_name})


        
        

# Keeping all login, registration, and user functions the same
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='skindisease', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username + " Username already exists"
                    break
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='skindisease', charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = f"INSERT INTO register(username,password,contact,email,address) VALUES('{username}','{password}','{contact}','{email}','{address}')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            if db_cursor.rowcount == 1:
                output = "success"
        if output == "success":
            return render(request, 'Register.html', {'data': 'Signup Process Completed'})
        else:
            return render(request, 'Register.html', {'data': 'Username already exists'})

def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='skindisease', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    index = 1
                    break
        if index == 1:
            return render(request, 'UserScreen.html', {'data': 'Welcome ' + username})
        else:
            return render(request, 'Login.html', {'data': 'Invalid login details'})
