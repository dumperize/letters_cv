import requests
import base64
import json
import io
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2

IMAGE_SHAPE = (70,70)
def preprocessing_fun(img):
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    border = 160
        
    ret,thresh = cv2.threshold(img,border,255,cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 2)
    img = cv2.bitwise_or(img, erode)
    

    gray = 255*(img < border).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    
    img = cv2.copyMakeBorder(rect,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
    img = cv2.resize(img, IMAGE_SHAPE, interpolation = cv2.INTER_NEAREST)
    

    img = np.where(img > 80, img - 70, img)
    img = np.where(img < 0, 0, img)
    img = img.reshape(*IMAGE_SHAPE,1)
    
    return img

# ФУНКЦИИ ПОЛУЧЕНИЯ И ОТПРАВКИ ДАННЫХ =========================================================================

# получить от сервера данные вида  { "response" : { "data" : { hash : img, ... } }, "error" : error, "code" : code }
def __getData(url, user_token):
    resp = requests.post(url, json={'token': user_token})
    jsn = None
    if resp.status_code == 200: jsn = resp.json()
    return jsn


# отправить на сервер данные вида  { "token": token, "data": { hash : label, ... } }
def __sendAnswer(url, user_token, data):
    data = json.dumps(data, ensure_ascii=False)
    resp = requests.post(url, json={"token": user_token, "data": data})
    jsn = None
    if resp.status_code == 200: jsn = resp.json()
    return jsn

# ============================ КЛАССИФИКАТОР ================================
# предсказание буквы по изображению - сейчас стоит случайный генератор меток
# !!! ЗДЕСЬ ДОЛЖНО БЫТЬ ОБРАЩЕНИЕ К ВАШЕМУ КЛАССИФИКАТОРУ!!!
# !!! ВЫ МОЖЕТЕ ПЕРЕДАВАТЬ НЕ ОТДЕЛЬНЫЕ ИЗОБРАЖЕНИЯ (как здесь), а СРАЗУ ВСЮ ПОРЦИЮ
# !!! ГЛАВНОЕ, ЧТОБЫ ВЕРНУЛИСЬ МЕТКИ ДЛЯ КАЖДОГО ИЗОБРАЖЕНИЯ
def predict(img, model):
    img = preprocessing_fun(np.array(img))
    prediction = model.predict(np.array([img]))
    labels = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё']
    ind = np.argmax(prediction)
    # print(labels[ind])
    return labels[ind]


# СЕАНС ТЕСТИРОВАНИЯ =========================================================================

get_url = "https://mooped.net/local/hackathon/api/data/get"
send_url = "https://mooped.net/local/hackathon/api/data/send"
# user_token = "<user_token>" # токен пользователя
user_token = "022886bb4d6ee778e07d466b3e42d7c3" # здесь должен быть ВАШ токен пользователя

k = 0
while True:
    jsn = __getData(get_url, user_token) # запрашиываем данные у сервера и читаем из JSON
    if jsn != None:

        file1 = open('json/json__' + str(k) + '.json', "w")
        data = json.dumps(jsn, ensure_ascii=False)
        file1.write(data)
        file1.close()
        k = k + 1

        data = jsn["response"]["data"]  # извлекаем данные из словаря
        # классифицируем изображения - формируем словарь меток predicted_labels
        predicted_labels = {}
        model = load_model('weight/20-epoch')
        
        for hash in data:
            img = base64.b64decode(data[hash])
            img = Image.open(io.BytesIO(img))
            # img = np.asarray(img) # тензор размерностью (128, 128, 3)
            predicted_label = predict(img, model)
            predicted_labels[hash] = predicted_label

        print('hey')
        # преобразуем словарь с метками в JSON и отправляем ответ серверу
        jsn = __sendAnswer(send_url, user_token, predicted_labels)

        if jsn == None: # проверка успешности
            print("ERROR")
            break
        else: # данные успешно доставлены - проверяем ответ от сервера
            status_code = jsn["code"]
            message = jsn["response"]["message"]
            print(message)
            if status_code == 0: break  # если все данные классифицированы, то завершить